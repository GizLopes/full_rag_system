# OBJETIVO PRINCIPAL
# Executar um agente ReAct para RAG clínico, escolhendo ferramentas de busca, rewrite e resposta com fontes.

import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_REGION = "us-east-1"

BASELINE_INDEX_PREFIX = "index/"
BASELINE_INDEX_FILE = "clinical_faiss.index"
BASELINE_METADATA_FILE = "clinical_faiss_metadata.pkl"

SEMANTIC_INDEX_PREFIX = "index_semantic/"
SEMANTIC_INDEX_FILE = "clinical_faiss_semantic.index"
SEMANTIC_METADATA_FILE = "clinical_faiss_semantic_metadata.pkl"

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_TOP_K = 5
DEFAULT_MAX_STEPS = 5
DEFAULT_MAX_TOKENS = 1400
DEFAULT_TEMPERATURE = 0.0


DOCUMENT_TYPE_HINTS = {
    "hemograma_e_bioquimica": [
        "creatinina",
        "glicemia",
        "hemoglobina",
        "leucocitos",
        "leucócitos",
        "plaquetas",
        "hemograma",
        "bioquimica",
        "bioquímica",
        "exame",
        "laboratorial",
    ],
    "consultas_ambulatoriais": [
        "consulta",
        "consultas",
        "ambulatorial",
        "queixa",
        "historia",
        "história",
        "conduta",
        "medicacoes",
        "medicações",
    ],
    "alta_hospitalar": [
        "alta",
        "hospitalar",
        "prescricao",
        "prescrição",
        "orientacoes",
        "orientações",
        "diagnostico",
        "diagnóstico",
    ],
    "parecer_cardiologista": [
        "cardiologista",
        "cardiologia",
        "parecer",
        "ecg",
        "pressao",
        "pressão",
        "risco cardiovascular",
    ],
    "ressonancia_coluna": [
        "ressonancia",
        "ressonância",
        "coluna",
        "lombar",
        "cervical",
        "toracica",
        "torácica",
        "laudo",
        "imagem",
    ],
}


class ArtifactStore:
    """Carrega e mantém índices FAISS baseline/semantic sob demanda."""

    def __init__(
        self,
        s3_client,
        bucket: str,
        use_local_cache: bool,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.use_local_cache = use_local_cache
        self.cache: Dict[str, Tuple[faiss.Index, List[Dict]]] = {}

    def get(self, index_mode: str) -> Tuple[faiss.Index, List[Dict]]:
        """Retorna índice e metadados para baseline ou semantic."""
        if index_mode in self.cache:
            return self.cache[index_mode]

        index_prefix, index_file, metadata_file = resolve_index_config(index_mode)

        index, metadata = load_faiss_artifacts(
            s3_client=self.s3_client,
            bucket=self.bucket,
            index_prefix=index_prefix,
            index_file=index_file,
            metadata_file=metadata_file,
            use_local_cache=self.use_local_cache,
        )

        self.cache[index_mode] = (index, metadata)

        return index, metadata


def download_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    local_file: Path,
    use_local_cache: bool = False,
) -> None:
    """
    Baixa arquivo do S3.

    Por padrão, sempre sobrescreve o arquivo local para evitar usar índice
    ou metadados antigos.
    """
    if use_local_cache and local_file.exists():
        print(f"Usando cache local: {local_file}")
        return

    if local_file.exists():
        print(f"Substituindo arquivo local antigo: {local_file}")

    print(f"Baixando do S3: s3://{bucket}/{s3_key}")

    try:
        s3_client.download_file(
            bucket,
            s3_key,
            str(local_file),
        )
    except ClientError as exc:
        raise RuntimeError(f"Erro ao baixar {s3_key} do S3: {exc}") from exc


def resolve_index_config(index_mode: str) -> Tuple[str, str, str]:
    """Resolve arquivos do índice baseline ou semantic."""
    if index_mode == "baseline":
        return BASELINE_INDEX_PREFIX, BASELINE_INDEX_FILE, BASELINE_METADATA_FILE

    if index_mode == "semantic":
        return SEMANTIC_INDEX_PREFIX, SEMANTIC_INDEX_FILE, SEMANTIC_METADATA_FILE

    raise ValueError(f"index_mode inválido: {index_mode}")


def load_faiss_artifacts(
    s3_client,
    bucket: str,
    index_prefix: str,
    index_file: str,
    metadata_file: str,
    use_local_cache: bool = False,
) -> Tuple[faiss.Index, List[Dict]]:
    """Carrega índice FAISS e metadados."""
    clean_prefix = index_prefix.strip("/")

    index_path = Path(index_file)
    metadata_path = Path(metadata_file)

    download_from_s3(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=f"{clean_prefix}/{index_file}",
        local_file=index_path,
        use_local_cache=use_local_cache,
    )

    download_from_s3(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=f"{clean_prefix}/{metadata_file}",
        local_file=metadata_path,
        use_local_cache=use_local_cache,
    )

    index = faiss.read_index(str(index_path))

    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    return index, metadata


def normalize_text(text: str) -> str:
    """Normaliza espaços sem destruir o conteúdo."""
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Normaliza texto para matching simples."""
    text = normalize_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_patient_id(question: str) -> Optional[str]:
    """Extrai patient_id quando explícito."""
    match = re.search(r"\bP\d{3}\b", question, flags=re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return None


def extract_patient_name(question: str) -> Optional[str]:
    """Extrai nome provável do paciente quando explícito."""
    patterns = [
        r"paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"da\s+paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"do\s+paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"de\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, question)

        if match:
            return match.group(1).strip(" ?.,;:")

    return None


def infer_document_types(question: str) -> List[str]:
    """Infere tipos documentais candidatos por palavras-chave."""
    normalized = normalize_for_matching(question)
    inferred = []

    for document_type, keywords in DOCUMENT_TYPE_HINTS.items():
        for keyword in keywords:
            if normalize_for_matching(keyword) in normalized:
                inferred.append(document_type)
                break

    return inferred


def extract_clinical_terms(question: str) -> List[str]:
    """Extrai termos clínicos simples a partir dos hints."""
    normalized = normalize_for_matching(question)
    terms = []

    for keywords in DOCUMENT_TYPE_HINTS.values():
        for keyword in keywords:
            if normalize_for_matching(keyword) in normalized:
                terms.append(keyword)

    return sorted(set(terms))


def build_local_rewrite(question: str) -> str:
    """Gera query reescrita por regras locais."""
    normalized_question = normalize_text(question)
    patient_id = extract_patient_id(normalized_question)
    patient_name = extract_patient_name(normalized_question)
    document_types = infer_document_types(normalized_question)
    clinical_terms = extract_clinical_terms(normalized_question)

    parts = []

    if patient_name:
        parts.append(f"paciente {patient_name}")

    if patient_id:
        parts.append(f"patient_id {patient_id}")

    if clinical_terms:
        parts.append(" ".join(clinical_terms))

    if document_types:
        parts.append(" ".join(document_types))

    parts.append(normalized_question)

    deduplicated = []

    for part in parts:
        if part and normalize_for_matching(part) not in [
            normalize_for_matching(item)
            for item in deduplicated
        ]:
            deduplicated.append(part)

    return " | ".join(deduplicated)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza vetor para usar inner product como similaridade por cosseno."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1

    return vector / norm


def embed_query(
    bedrock_client,
    query: str,
    model_id: str,
) -> np.ndarray:
    """Gera embedding da query usando Titan Embeddings."""
    payload = {
        "inputText": query,
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    vector = np.array([embedding], dtype="float32")

    return normalize_vector(vector)


def search_faiss(
    artifact_store: ArtifactStore,
    bedrock_client,
    query: str,
    index_mode: str,
    embedding_model_id: str,
    top_k: int,
) -> List[Dict]:
    """Executa busca FAISS em índice baseline ou semantic."""
    index, metadata = artifact_store.get(index_mode)

    query_vector = embed_query(
        bedrock_client=bedrock_client,
        query=query,
        model_id=embedding_model_id,
    )

    search_k = min(top_k, len(metadata))
    scores, positions = index.search(query_vector, search_k)

    results = []

    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        if position < 0:
            continue

        item = metadata[position].copy()
        item["rank"] = rank
        item["score"] = float(score)
        item["query_used"] = query
        item["index_mode"] = index_mode

        results.append(item)

    return results


def summarize_results_for_observation(results: List[Dict], max_chars: int = 900) -> List[Dict]:
    """Reduz resultados para caber no loop ReAct."""
    observations = []

    for item in results:
        text = normalize_text(item.get("text", ""))

        observations.append({
            "rank": item.get("rank"),
            "score": round(float(item.get("score", 0)), 4),
            "chunk_id": item.get("chunk_id"),
            "document_name": item.get("document_name"),
            "document_type": item.get("document_type"),
            "chunk_number": item.get("chunk_number"),
            "total_chunks": item.get("total_chunks"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "s3_uri": item.get("s3_uri"),
            "chunk_strategy": item.get("chunk_strategy"),
            "patient_id": item.get("patient_id"),
            "patient_name": item.get("patient_name"),
            "clinical_section": item.get("clinical_section"),
            "text_preview": text[:max_chars],
        })

    return observations


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extrai JSON válido retornado pelo LLM."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("A resposta do LLM não contém JSON válido.")

    return json.loads(match.group(0))


def invoke_claude_text(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoca Claude e retorna texto."""
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    content = response_body.get("content", [])

    if not content:
        return ""

    return content[0].get("text", "").strip()


def invoke_claude_json(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Invoca Claude e retorna JSON."""
    raw_text = invoke_claude_text(
        bedrock_client=bedrock_client,
        model_id=model_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if not raw_text:
        raise RuntimeError("Claude retornou texto vazio.")

    return extract_json_from_text(raw_text)


def build_react_prompt(
    question: str,
    trace: List[Dict[str, Any]],
    default_top_k: int,
) -> str:
    """Cria prompt do loop ReAct."""
    return f"""
Você é um agente ReAct para RAG clínico de treinamento.

Objetivo:
Responder à pergunta usando ferramentas de recuperação. Você deve pensar, escolher uma ação, observar o resultado e decidir se já tem evidência suficiente.

Pergunta original:
{question}

Ferramentas disponíveis:

1. retrieve_baseline
Busca vetorial no índice FAISS baseline.
Entrada:
{{"query": "...", "top_k": {default_top_k}}}

2. retrieve_semantic
Busca vetorial no índice FAISS semântico.
Entrada:
{{"query": "...", "top_k": {default_top_k}}}

3. query_rewrite
Reescreve a pergunta para melhorar recuperação.
Entrada:
{{"query": "..."}}

4. final_answer
Finaliza com a resposta.
Entrada:
{{"answer": "..."}}

Regras:
1. Use ferramentas antes de responder, exceto se já houver observação suficiente no trace.
2. Não invente dados clínicos.
3. A resposta final deve citar documento, página e chunk.
4. Se a evidência for insuficiente, responda que não encontrou informação suficiente.
5. Para exames laboratoriais, prefira evidência de hemograma_e_bioquimica.
6. Para perguntas com paciente explícito, confirme paciente e termo clínico no mesmo chunk.
7. Retorne apenas JSON válido, sem markdown.

Trace atual:
{json.dumps(trace, ensure_ascii=False, indent=2)}

Formato obrigatório:
{{
  "thought": "raciocínio curto sobre a próxima ação",
  "action": "retrieve_baseline",
  "action_input": {{"query": "...", "top_k": {default_top_k}}}
}}

Ações permitidas:
- retrieve_baseline
- retrieve_semantic
- query_rewrite
- final_answer
""".strip()


def run_tool(
    action: str,
    action_input: Dict[str, Any],
    artifact_store: ArtifactStore,
    bedrock_client,
    embedding_model_id: str,
    default_top_k: int,
) -> Tuple[str, List[Dict]]:
    """Executa uma ferramenta do agente."""
    action = normalize_text(action).lower()

    if action in {"retrieve_baseline", "retrieve_semantic"}:
        index_mode = "baseline" if action == "retrieve_baseline" else "semantic"
        query = normalize_text(str(action_input.get("query") or ""))
        top_k = int(action_input.get("top_k") or default_top_k)

        if not query:
            return "Consulta vazia. Use query_rewrite ou informe uma query válida.", []

        try:
            results = search_faiss(
                artifact_store=artifact_store,
                bedrock_client=bedrock_client,
                query=query,
                index_mode=index_mode,
                embedding_model_id=embedding_model_id,
                top_k=top_k,
            )
        except Exception as exc:
            return f"Falha na ferramenta {action}: {exc}", []

        observation = {
            "tool": action,
            "query": query,
            "top_k": top_k,
            "results": summarize_results_for_observation(results),
        }

        return json.dumps(observation, ensure_ascii=False, indent=2), results

    if action == "query_rewrite":
        query = normalize_text(str(action_input.get("query") or ""))
        rewritten_query = build_local_rewrite(query)

        observation = {
            "tool": "query_rewrite",
            "original_query": query,
            "rewritten_query": rewritten_query,
            "detected_entities": {
                "patient_id": extract_patient_id(query),
                "patient_name": extract_patient_name(query),
                "document_types": infer_document_types(query),
                "clinical_terms": extract_clinical_terms(query),
            },
        }

        return json.dumps(observation, ensure_ascii=False, indent=2), []

    return f"Ação desconhecida: {action}", []


def build_final_answer_prompt(
    question: str,
    trace: List[Dict[str, Any]],
    gathered_results: List[Dict],
) -> str:
    """Cria prompt de segurança para resposta final caso o agente não finalize."""
    context = summarize_results_for_observation(gathered_results, max_chars=1200)

    return f"""
Você é um assistente clínico de recuperação de conhecimento para treinamento de RAG.

Responda apenas com base no CONTEXTO RECUPERADO.

Pergunta:
{question}

Trace ReAct:
{json.dumps(trace, ensure_ascii=False, indent=2)}

Contexto recuperado:
{json.dumps(context, ensure_ascii=False, indent=2)}

Regras:
1. Não invente dados clínicos.
2. Cite documento, página e chunk.
3. Se não houver evidência suficiente, diga que não encontrou informação suficiente.
4. Seja direto.

Resposta:
""".strip()


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove chunks duplicados preservando a primeira ocorrência."""
    seen = set()
    deduplicated = []

    for item in results:
        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        deduplicated.append(item)

    return deduplicated


def build_sources(results: List[Dict]) -> List[Dict]:
    """Monta fontes estruturadas para auditoria."""
    sources = []

    for item in deduplicate_results(results):
        sources.append({
            "rank": item.get("rank"),
            "score": item.get("score"),
            "query_used": item.get("query_used"),
            "index_mode": item.get("index_mode"),
            "document_name": item.get("document_name"),
            "document_type": item.get("document_type"),
            "chunk_number": item.get("chunk_number"),
            "total_chunks": item.get("total_chunks"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "s3_uri": item.get("s3_uri"),
            "chunk_strategy": item.get("chunk_strategy"),
            "patient_id": item.get("patient_id"),
            "patient_name": item.get("patient_name"),
            "clinical_section": item.get("clinical_section"),
        })

    return sources


def run_react_agent(
    question: str,
    artifact_store: ArtifactStore,
    bedrock_client,
    embedding_model_id: str,
    llm_model_id: str,
    top_k: int,
    max_steps: int,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Executa loop ReAct."""
    trace: List[Dict[str, Any]] = []
    gathered_results: List[Dict] = []
    final_answer: Optional[str] = None

    for step in range(1, max_steps + 1):
        prompt = build_react_prompt(
            question=question,
            trace=trace,
            default_top_k=top_k,
        )

        try:
            decision = invoke_claude_json(
                bedrock_client=bedrock_client,
                model_id=llm_model_id,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            decision = {
                "thought": f"Falha ao obter ação JSON do LLM. Usando busca baseline como fallback. Erro: {exc}",
                "action": "retrieve_baseline",
                "action_input": {
                    "query": question,
                    "top_k": top_k,
                },
            }

        thought = normalize_text(str(decision.get("thought") or ""))
        action = normalize_text(str(decision.get("action") or ""))
        action_input = decision.get("action_input") or {}

        step_record: Dict[str, Any] = {
            "step": step,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

        if action == "final_answer":
            final_answer = normalize_text(str(action_input.get("answer") or ""))

            if not final_answer:
                final_answer = "Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança."

            step_record["observation"] = "Resposta final produzida pelo agente."
            trace.append(step_record)
            break

        observation, tool_results = run_tool(
            action=action,
            action_input=action_input,
            artifact_store=artifact_store,
            bedrock_client=bedrock_client,
            embedding_model_id=embedding_model_id,
            default_top_k=top_k,
        )

        step_record["observation"] = observation
        trace.append(step_record)

        if tool_results:
            gathered_results.extend(tool_results)

    gathered_results = deduplicate_results(gathered_results)

    if not final_answer:
        prompt = build_final_answer_prompt(
            question=question,
            trace=trace,
            gathered_results=gathered_results,
        )

        final_answer = invoke_claude_text(
            bedrock_client=bedrock_client,
            model_id=llm_model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return {
        "question": question,
        "answer": final_answer,
        "sources": build_sources(gathered_results),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa agente ReAct para RAG clínico com ferramentas de retrieval e query rewrite."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Pergunta clínica em linguagem natural.",
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 onde estão os artefatos FAISS.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--embedding-model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Modelo de embedding no Amazon Bedrock.",
    )

    parser.add_argument(
        "--llm-model-id",
        default=DEFAULT_LLM_MODEL_ID,
        help="Modelo LLM ou ARN do inference profile no Amazon Bedrock.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Quantidade de chunks recuperados por chamada de ferramenta.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Número máximo de passos ReAct.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens gerados pelo LLM.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperatura do LLM.",
    )

    parser.add_argument(
        "--use-local-cache",
        action="store_true",
        help=(
            "Usa arquivos FAISS locais se existirem. "
            "Por padrão, sempre baixa os artefatos do S3."
        ),
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    artifact_store = ArtifactStore(
        s3_client=s3_client,
        bucket=args.bucket,
        use_local_cache=args.use_local_cache,
    )

    print("Iniciando ReAct Agent")
    print(f"Bucket: {args.bucket}")
    print(f"Top-K por ferramenta: {args.top_k}")
    print(f"Max steps: {args.max_steps}")
    print(f"Usar cache local: {args.use_local_cache}")

    result = run_react_agent(
        question=args.question,
        artifact_store=artifact_store,
        bedrock_client=bedrock_client,
        embedding_model_id=args.embedding_model_id,
        llm_model_id=args.llm_model_id,
        top_k=args.top_k,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nResposta:")
    print(result["answer"])

    print("\nFontes estruturadas:")
    print(json.dumps(result["sources"], ensure_ascii=False, indent=2))

    print("\nTrace ReAct:")
    print(json.dumps(result["trace"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()