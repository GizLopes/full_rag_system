# OBJETIVO PRINCIPAL
# Orquestrar RAG com LangGraph e exportar grafo visual dinâmico da resposta, evidências, avaliação e fontes

import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    raise RuntimeError(
        "LangGraph não está instalado. Execute: pip install langgraph"
    ) from exc


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
DEFAULT_MIN_VECTOR_SCORE = 0.25
DEFAULT_MIN_RELEVANCE_SCORE = 0.55
DEFAULT_MAX_CORRECTIONS = 1
DEFAULT_MAX_TOKENS = 1200
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


class RAGState(TypedDict, total=False):
    question: str
    current_query: str
    rewritten_query: Optional[str]
    index_mode: str
    documents: List[Dict]
    evaluation: Dict[str, Any]
    answer: str
    sources: List[Dict]
    correction_count: int
    trace: List[Dict[str, Any]]


class RuntimeClients:
    """Agrupa clientes e artefatos usados pelos nós do LangGraph."""

    def __init__(
        self,
        s3_client,
        bedrock_client,
        index,
        metadata: List[Dict],
        args,
    ) -> None:
        self.s3_client = s3_client
        self.bedrock_client = bedrock_client
        self.index = index
        self.metadata = metadata
        self.args = args


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
    index_mode: str,
    use_local_cache: bool = False,
) -> Tuple[faiss.Index, List[Dict]]:
    """Carrega índice FAISS e metadados para baseline ou semantic."""
    index_prefix, index_file, metadata_file = resolve_index_config(index_mode)
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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza vetor para usar inner product como similaridade por cosseno."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1

    return vector / norm


def embed_question(
    bedrock_client,
    question: str,
    model_id: str,
) -> np.ndarray:
    """Gera embedding da pergunta usando Titan Embeddings."""
    payload = {
        "inputText": question,
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
    index: faiss.Index,
    metadata: List[Dict],
    question: str,
    bedrock_client,
    embedding_model_id: str,
    top_k: int,
    index_mode: str,
) -> List[Dict]:
    """Executa busca FAISS para uma pergunta."""
    query_vector = embed_question(
        bedrock_client=bedrock_client,
        question=question,
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
        item["query_used"] = question
        item["index_mode"] = index_mode
        results.append(item)

    return results


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


def local_evaluate_retrieval(
    question: str,
    results: List[Dict],
    min_vector_score: float,
) -> Dict[str, Any]:
    """Avalia retrieval por heurística local."""
    if not results:
        return {
            "decision": "rewrite",
            "relevance_score": 0.0,
            "relevant_chunk_ids": [],
            "reason": "Nenhum chunk recuperado.",
            "rewrite_suggestion": build_local_rewrite(question),
        }

    best_score = max(item.get("score", 0) for item in results)
    patient_id = extract_patient_id(question)
    patient_name = extract_patient_name(question)
    clinical_terms = extract_clinical_terms(question)
    relevant_chunk_ids = []

    for item in results:
        searchable = normalize_for_matching(
            " ".join(
                str(value)
                for value in [
                    item.get("text"),
                    item.get("document_name"),
                    item.get("document_type"),
                    item.get("patient_id"),
                    item.get("patient_name"),
                    item.get("clinical_section"),
                ]
                if value
            )
        )

        patient_match = True

        if patient_id:
            patient_match = patient_id.lower() in searchable

        if patient_name:
            patient_match = normalize_for_matching(patient_name) in searchable

        term_match = True

        if clinical_terms:
            term_match = any(
                normalize_for_matching(term) in searchable
                for term in clinical_terms
            )

        if patient_match and term_match:
            relevant_chunk_ids.append(item.get("chunk_id"))

    if relevant_chunk_ids and best_score >= min_vector_score:
        return {
            "decision": "pass",
            "relevance_score": float(best_score),
            "relevant_chunk_ids": relevant_chunk_ids,
            "reason": "Contexto contém correspondência de paciente/termo clínico.",
            "rewrite_suggestion": None,
        }

    return {
        "decision": "rewrite",
        "relevance_score": float(best_score),
        "relevant_chunk_ids": relevant_chunk_ids,
        "reason": "Contexto inicial não parece suficiente para responder com segurança.",
        "rewrite_suggestion": build_local_rewrite(question),
    }


def invoke_claude_json(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Invoca Claude e retorna JSON."""
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
        raise RuntimeError("Claude não retornou conteúdo.")

    raw_text = content[0].get("text", "").strip()

    if not raw_text:
        raise RuntimeError("Claude retornou texto vazio.")

    return extract_json_from_text(raw_text)


def build_evaluation_prompt(question: str, results: List[Dict]) -> str:
    """Cria prompt para avaliar contexto recuperado."""
    evidence = []

    for item in results:
        evidence.append({
            "rank": item.get("rank"),
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
            "document_name": item.get("document_name"),
            "document_type": item.get("document_type"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "text": item.get("text", "")[:1800],
        })

    return f"""
Você é um avaliador de retrieval para um grafo LangGraph RAG clínico de treinamento.

Avalie se os chunks recuperados são suficientes para responder à pergunta.

Pergunta:
{question}

Chunks:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Critérios:
1. O contexto deve conter evidência direta para responder.
2. Se a pergunta pedir resultado de exame, o chunk precisa conter paciente e valor do exame.
3. Se a evidência estiver incompleta, recomende rewrite.
4. Não responda à pergunta clínica.
5. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "decision": "pass",
  "relevance_score": 0.0,
  "relevant_chunk_ids": [],
  "reason": "explicação curta",
  "rewrite_suggestion": "consulta reescrita ou null"
}}

Valores permitidos para decision:
- "pass"
- "rewrite"
- "insufficient"
""".strip()


def evaluate_retrieval(
    bedrock_client,
    question: str,
    results: List[Dict],
    model_id: str,
    max_tokens: int,
    temperature: float,
    use_llm: bool,
    min_vector_score: float,
) -> Dict[str, Any]:
    """Avalia se o retrieval é suficiente."""
    if not use_llm:
        return local_evaluate_retrieval(
            question=question,
            results=results,
            min_vector_score=min_vector_score,
        )

    prompt = build_evaluation_prompt(
        question=question,
        results=results,
    )

    try:
        evaluation = invoke_claude_json(
            bedrock_client=bedrock_client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except (ClientError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Aviso: falha no evaluator LLM. Usando avaliação local. Erro: {exc}")
        evaluation = local_evaluate_retrieval(
            question=question,
            results=results,
            min_vector_score=min_vector_score,
        )

    evaluation.setdefault("decision", "rewrite")
    evaluation.setdefault("relevance_score", 0.0)
    evaluation.setdefault("relevant_chunk_ids", [])
    evaluation.setdefault("reason", "")
    evaluation.setdefault("rewrite_suggestion", build_local_rewrite(question))

    return evaluation


def build_rewrite_prompt(question: str, evaluation: Dict[str, Any]) -> str:
    """Cria prompt para query rewrite."""
    patient_id = extract_patient_id(question)
    patient_name = extract_patient_name(question)
    document_types = infer_document_types(question)
    clinical_terms = extract_clinical_terms(question)

    return f"""
Você é um nó de query rewrite em um grafo LangGraph RAG clínico.

A recuperação anterior foi insuficiente. Reescreva a pergunta para melhorar o próximo retrieval.

Pergunta original:
{question}

Avaliação:
{json.dumps(evaluation, ensure_ascii=False, indent=2)}

Entidades detectadas:
patient_id: {patient_id}
patient_name: {patient_name}
document_types: {json.dumps(document_types, ensure_ascii=False)}
clinical_terms: {json.dumps(clinical_terms, ensure_ascii=False)}

Regras:
1. Não responda à pergunta clínica.
2. Não invente valor clínico.
3. Preserve paciente, exame e tipo documental quando existirem.
4. Gere uma única query curta e objetiva.
5. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "rewritten_query": "...",
  "correction_reason": "explicação curta"
}}
""".strip()


def rewrite_query(
    bedrock_client,
    question: str,
    evaluation: Dict[str, Any],
    model_id: str,
    max_tokens: int,
    temperature: float,
    use_llm: bool,
) -> Dict[str, Any]:
    """Reescreve query para nova recuperação."""
    fallback_query = evaluation.get("rewrite_suggestion") or build_local_rewrite(question)

    if not use_llm:
        return {
            "rewritten_query": fallback_query,
            "correction_reason": "Query reescrita por regras locais.",
        }

    prompt = build_rewrite_prompt(
        question=question,
        evaluation=evaluation,
    )

    try:
        result = invoke_claude_json(
            bedrock_client=bedrock_client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except (ClientError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Aviso: falha no rewrite LLM. Usando rewrite local. Erro: {exc}")
        result = {
            "rewritten_query": fallback_query,
            "correction_reason": "Fallback local após falha do rewrite LLM.",
        }

    if not result.get("rewritten_query"):
        result["rewritten_query"] = fallback_query

    return result


def build_context(results: List[Dict]) -> str:
    """Monta contexto enviado ao LLM."""
    blocks = []

    for item in results:
        source = item.get("document_name", "documento_desconhecido")
        document_type = item.get("document_type", "tipo_desconhecido")
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        chunk_number = item.get("chunk_number")
        total_chunks = item.get("total_chunks")
        score = item.get("score", 0)
        text = item.get("text", "")

        citation = (
            f"Fonte: {source} | "
            f"Tipo: {document_type} | "
            f"Página(s): {page_start}-{page_end} | "
            f"Chunk: {chunk_number}/{total_chunks} | "
            f"Score FAISS: {score:.4f}"
        )

        blocks.append(f"[{citation}]\n{text}")

    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(
    question: str,
    context: str,
    trace: List[Dict[str, Any]],
) -> str:
    """Cria prompt final para resposta."""
    return f"""
Você é um assistente clínico de recuperação de conhecimento para treinamento de RAG.

Regras obrigatórias:
1. Responda apenas com base no CONTEXTO.
2. Não invente dados clínicos.
3. Se o CONTEXTO não trouxer evidência suficiente, responda:
   "Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança."
4. Cite documento, página e chunk usados.
5. Seja direto e objetivo.
6. Não faça diagnóstico novo.
7. Não recomende conduta clínica fora do que estiver nos documentos.

PERGUNTA ORIGINAL:
{question}

TRACE LANGGRAPH:
{json.dumps(trace, ensure_ascii=False, indent=2)}

CONTEXTO:
{context}

RESPOSTA:
""".strip()


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


def build_sources(results: List[Dict]) -> List[Dict]:
    """Monta fontes estruturadas para auditoria."""
    sources = []

    for item in results:
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


def create_retrieve_node(runtime: RuntimeClients):
    """Cria nó de retrieval."""

    def retrieve_node(state: RAGState) -> RAGState:
        current_query = state.get("current_query") or state["question"]
        correction_count = state.get("correction_count", 0)

        documents = search_faiss(
            index=runtime.index,
            metadata=runtime.metadata,
            question=current_query,
            bedrock_client=runtime.bedrock_client,
            embedding_model_id=runtime.args.embedding_model_id,
            top_k=runtime.args.top_k,
            index_mode=runtime.args.index_mode,
        )

        trace = state.get("trace", [])
        trace.append({
            "node": "retrieve",
            "query": current_query,
            "documents": [
                {
                    "rank": item.get("rank"),
                    "score": item.get("score"),
                    "document_name": item.get("document_name"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "chunk_id": item.get("chunk_id"),
                }
                for item in documents
            ],
        })

        return {
            **state,
            "documents": documents,
            "trace": trace,
            "correction_count": correction_count,
        }

    return retrieve_node


def create_evaluate_node(runtime: RuntimeClients):
    """Cria nó de avaliação."""

    def evaluate_node(state: RAGState) -> RAGState:
        evaluation = evaluate_retrieval(
            bedrock_client=runtime.bedrock_client,
            question=state["question"],
            results=state.get("documents", []),
            model_id=runtime.args.llm_model_id,
            max_tokens=runtime.args.max_tokens,
            temperature=runtime.args.temperature,
            use_llm=not runtime.args.no_evaluator_llm,
            min_vector_score=runtime.args.min_vector_score,
        )

        trace = state.get("trace", [])
        trace.append({
            "node": "evaluate",
            "evaluation": evaluation,
        })

        return {
            **state,
            "evaluation": evaluation,
            "trace": trace,
        }

    return evaluate_node


def create_rewrite_node(runtime: RuntimeClients):
    """Cria nó de query rewrite."""

    def rewrite_node(state: RAGState) -> RAGState:
        correction_count = state.get("correction_count", 0) + 1

        rewrite_result = rewrite_query(
            bedrock_client=runtime.bedrock_client,
            question=state["question"],
            evaluation=state.get("evaluation", {}),
            model_id=runtime.args.llm_model_id,
            max_tokens=runtime.args.max_tokens,
            temperature=runtime.args.temperature,
            use_llm=not runtime.args.no_rewrite_llm,
        )

        rewritten_query = rewrite_result.get("rewritten_query") or build_local_rewrite(
            state["question"]
        )

        trace = state.get("trace", [])
        trace.append({
            "node": "rewrite",
            "rewrite_result": rewrite_result,
            "correction_count": correction_count,
        })

        return {
            **state,
            "current_query": rewritten_query,
            "rewritten_query": rewritten_query,
            "correction_count": correction_count,
            "trace": trace,
        }

    return rewrite_node


def create_answer_node(runtime: RuntimeClients):
    """Cria nó de resposta final."""

    def answer_node(state: RAGState) -> RAGState:
        documents = state.get("documents", [])
        context = build_context(documents)
        trace = state.get("trace", [])

        if not documents:
            answer = "Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança."
        else:
            prompt = build_answer_prompt(
                question=state["question"],
                context=context,
                trace=trace,
            )

            answer = invoke_claude_text(
                bedrock_client=runtime.bedrock_client,
                model_id=runtime.args.llm_model_id,
                prompt=prompt,
                max_tokens=runtime.args.max_tokens,
                temperature=runtime.args.temperature,
            )

        sources = build_sources(documents)

        trace.append({
            "node": "answer",
            "sources_count": len(sources),
        })

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "trace": trace,
        }

    return answer_node


def create_router(args):
    """Cria roteador condicional do grafo."""

    def route_after_evaluation(state: RAGState) -> str:
        evaluation = state.get("evaluation", {})
        decision = evaluation.get("decision")
        relevance_score = float(evaluation.get("relevance_score", 0))
        correction_count = state.get("correction_count", 0)

        if decision == "pass" and relevance_score >= args.min_relevance_score:
            return "answer"

        if correction_count >= args.max_corrections:
            return "answer"

        return "rewrite"

    return route_after_evaluation



def export_langgraph_visual(
    output_png: str,
    output_dot: str,
    dpi: int = 180,
) -> None:
    """
    Exporta uma visualização real do grafo em PNG e DOT.

    O PNG usa networkx + matplotlib.
    O DOT permite abrir o grafo em ferramentas como Graphviz, VS Code Graphviz Preview
    ou qualquer visualizador compatível.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        print(
            "Aviso: não foi possível gerar imagem do grafo. "
            "Instale as dependências com: pip install matplotlib networkx"
        )
        print(f"Detalhe: {exc}")
        return

    graph = nx.DiGraph()

    nodes = {
        "START": {
            "label": "START",
            "kind": "start",
        },
        "retrieve": {
            "label": "retrieve\nFAISS retrieval",
            "kind": "retrieval",
        },
        "evaluate": {
            "label": "evaluate\ncontext grading",
            "kind": "evaluation",
        },
        "rewrite": {
            "label": "rewrite\nquery correction",
            "kind": "correction",
        },
        "answer": {
            "label": "answer\nClaude response",
            "kind": "generation",
        },
        "END": {
            "label": "END",
            "kind": "end",
        },
    }

    edges = [
        ("START", "retrieve", "start"),
        ("retrieve", "evaluate", "documents"),
        ("evaluate", "answer", "pass"),
        ("evaluate", "rewrite", "rewrite needed"),
        ("rewrite", "retrieve", "retry"),
        ("answer", "END", "final"),
    ]

    for node, attrs in nodes.items():
        graph.add_node(node, **attrs)

    for source_node, target_node, label in edges:
        graph.add_edge(source_node, target_node, label=label)

    positions = {
        "START": (0, 0),
        "retrieve": (2, 0),
        "evaluate": (4, 0),
        "rewrite": (4, -1.8),
        "answer": (6, 0),
        "END": (8, 0),
    }

    node_colors = {
        "start": "#D9EAF7",
        "retrieval": "#D8F3DC",
        "evaluation": "#FFF3B0",
        "correction": "#FFD6A5",
        "generation": "#E5D4EF",
        "end": "#EAEAEA",
    }

    colors = [
        node_colors.get(graph.nodes[node].get("kind"), "#FFFFFF")
        for node in graph.nodes
    ]

    labels = {
        node: attrs["label"]
        for node, attrs in graph.nodes(data=True)
    }

    edge_labels = {
        (source_node, target_node): attrs["label"]
        for source_node, target_node, attrs in graph.edges(data=True)
    }

    plt.figure(figsize=(12, 5))

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=colors,
        node_size=3600,
        edgecolors="#333333",
        linewidths=1.2,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=9,
        font_weight="bold",
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=1.8,
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.52,
    )

    plt.title("Clinical RAG LangGraph Flow", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close()

    write_dot_file(
        graph=graph,
        output_dot=output_dot,
    )

    print(f"Grafo PNG gerado: {output_png}")
    print(f"Grafo DOT gerado: {output_dot}")


def write_dot_file(graph, output_dot: str) -> None:
    """Salva uma versão DOT do grafo para visualização externa."""
    node_style = {
        "start": {
            "fillcolor": "#D9EAF7",
        },
        "retrieval": {
            "fillcolor": "#D8F3DC",
        },
        "evaluation": {
            "fillcolor": "#FFF3B0",
        },
        "correction": {
            "fillcolor": "#FFD6A5",
        },
        "generation": {
            "fillcolor": "#E5D4EF",
        },
        "end": {
            "fillcolor": "#EAEAEA",
        },
    }

    lines = [
        "digraph LangGraphRAG {",
        '  rankdir=LR;',
        '  graph [fontname="Arial", bgcolor="white"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", color="#333333"];',
        '  edge [fontname="Arial", color="#333333"];',
    ]

    for node, attrs in graph.nodes(data=True):
        label = attrs.get("label", node).replace("\n", "\\n")
        kind = attrs.get("kind", "")
        fillcolor = node_style.get(kind, {}).get("fillcolor", "#FFFFFF")
        lines.append(
            f'  "{node}" [label="{label}", fillcolor="{fillcolor}"];'
        )

    for source_node, target_node, attrs in graph.edges(data=True):
        label = attrs.get("label", "")
        lines.append(
            f'  "{source_node}" -> "{target_node}" [label="{label}"];'
        )

    lines.append("}")

    Path(output_dot).write_text("\n".join(lines), encoding="utf-8")

def build_graph(runtime: RuntimeClients):
    """Monta o grafo LangGraph de RAG."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", create_retrieve_node(runtime))
    graph.add_node("evaluate", create_evaluate_node(runtime))
    graph.add_node("rewrite", create_rewrite_node(runtime))
    graph.add_node("answer", create_answer_node(runtime))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        create_router(runtime.args),
        {
            "answer": "answer",
            "rewrite": "rewrite",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)

    return graph.compile()



def shorten_text(value: Any, max_len: int = 90) -> str:
    """Encurta texto para labels do grafo visual."""
    text = normalize_text(str(value or ""))

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def xml_escape(value: Any) -> str:
    """Escapa texto para SVG/HTML."""
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def wrap_text(value: Any, max_chars: int = 34, max_lines: int = 5) -> List[str]:
    """Quebra texto em linhas curtas para caber dentro dos cards SVG."""
    text = normalize_text(str(value or ""))

    if not text:
        return [""]

    words = text.split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(current)

    if len(lines) > max_lines:
        lines = lines[:max_lines]

    joined = " ".join(lines)
    if len(joined) < len(text) and lines:
        lines[-1] = lines[-1][: max(0, max_chars - 3)] + "..."

    return lines or [""]


def svg_text_block(
    lines: List[str],
    x: float,
    y: float,
    font_size: int = 13,
    color: str = "#1f2937",
    weight: str = "400",
    line_height: int = 18,
) -> str:
    """Renderiza linhas de texto em SVG."""
    tspans = []

    for idx, line in enumerate(lines):
        dy = 0 if idx == 0 else line_height
        tspans.append(
            f'<tspan x="{x}" dy="{dy}">{xml_escape(line)}</tspan>'
        )

    return (
        f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{font_size}" fill="{color}" font-weight="{weight}">'
        + "".join(tspans)
        + "</text>"
    )


def svg_card(
    node_id: str,
    title: str,
    subtitle: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "#334155",
    badge: Optional[str] = None,
    title_color: str = "#0f172a",
) -> str:
    """Cria um card SVG com título, badge e texto quebrado."""
    parts = [
        f'<g id="{xml_escape(node_id)}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>',
    ]

    if badge:
        badge_x = x + width - 84
        parts.append(
            f'<rect x="{badge_x}" y="{y + 12}" width="68" height="24" rx="12" '
            f'fill="#ffffff" stroke="{stroke}" stroke-width="0.8" opacity="0.92"/>'
        )
        parts.append(
            svg_text_block(
                [badge],
                badge_x + 12,
                y + 29,
                font_size=11,
                color="#334155",
                weight="700",
                line_height=14,
            )
        )

    title_lines = wrap_text(title, max_chars=28, max_lines=2)
    parts.append(
        svg_text_block(
            title_lines,
            x + 16,
            y + 28,
            font_size=14,
            color=title_color,
            weight="800",
            line_height=17,
        )
    )

    subtitle_lines = wrap_text(subtitle, max_chars=34, max_lines=5)
    subtitle_y = y + 62 if len(title_lines) == 1 else y + 78

    parts.append(
        svg_text_block(
            subtitle_lines,
            x + 16,
            subtitle_y,
            font_size=12,
            color="#334155",
            weight="500",
            line_height=16,
        )
    )

    parts.append("</g>")

    return "\n".join(parts)


def svg_arrow(
    source: Tuple[float, float],
    target: Tuple[float, float],
    label: str = "",
    color: str = "#64748b",
) -> str:
    """Cria uma seta SVG curva entre cards."""
    sx, sy = source
    tx, ty = target
    mid_x = (sx + tx) / 2
    path = (
        f'M {sx} {sy} '
        f'C {mid_x} {sy}, {mid_x} {ty}, {tx} {ty}'
    )

    label_x = mid_x
    label_y = (sy + ty) / 2 - 8

    label_svg = ""

    if label:
        label_svg = (
            f'<rect x="{label_x - 52}" y="{label_y - 15}" width="104" height="22" '
            f'rx="11" fill="#ffffff" stroke="{color}" stroke-width="0.6" opacity="0.95"/>'
            + svg_text_block(
                [label],
                label_x - 42,
                label_y,
                font_size=10,
                color=color,
                weight="700",
                line_height=12,
            )
        )

    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8" '
        f'marker-end="url(#arrowhead)"/>'
        + label_svg
    )


def compact_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrupa informações do trace para desenhar o grafo dinâmico."""
    retrieves = []
    evaluations = []
    rewrites = []

    for step in trace:
        node = step.get("node")

        if node == "retrieve":
            retrieves.append(step)
        elif node == "evaluate":
            evaluations.append(step)
        elif node == "rewrite":
            rewrites.append(step)

    return {
        "retrieves": retrieves,
        "evaluations": evaluations,
        "rewrites": rewrites,
    }


def export_answer_graph_visual(
    state: RAGState,
    output_png: str,
    output_dot: str,
    output_svg: str = "langgraph_answer_graph.svg",
    output_html: str = "langgraph_answer_graph.html",
    dpi: int = 180,
) -> None:
    """
    Exporta um grafo dinâmico da resposta em SVG/HTML legível.

    Esta versão não usa networkx para o desenho final. Ela cria um evidence map
    com layout controlado, cards maiores e textos quebrados, evitando sobreposição.
    """
    question = state.get("question", "")
    trace = state.get("trace", [])
    sources = state.get("sources", [])
    answer = state.get("answer", "")

    grouped = compact_trace(trace)
    retrieves = grouped["retrieves"]
    evaluations = grouped["evaluations"]
    rewrites = grouped["rewrites"]

    max_candidates = 5
    card_w = 300
    card_h = 118
    gap_y = 20

    retrieve_columns = max(1, len(retrieves))
    canvas_width = 1900 + max(0, retrieve_columns - 1) * 120
    candidate_count = max(
        [min(len(item.get("documents", [])), max_candidates) for item in retrieves] or [1]
    )
    canvas_height = max(860, 250 + candidate_count * (card_h + gap_y))

    x_question = 40
    x_retrieve = 380
    x_evaluate = 760
    x_final = 1120
    x_answer = 1500

    y_top = 60
    y_candidates = 230

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
        '<polygon points="0 0, 10 3.5, 0 7" fill="#64748b"/>',
        "</marker>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="4" flood-color="#000000" flood-opacity="0.14"/>',
        "</filter>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>',
        svg_text_block(
            ["Dynamic Answer Graph", "Clinical RAG evidence path"],
            40,
            34,
            font_size=18,
            color="#0f172a",
            weight="900",
            line_height=22,
        ),
    ]

    # Question card
    svg_parts.append(
        f'<g filter="url(#shadow)">'
        + svg_card(
            node_id="question",
            title="Question",
            subtitle=question,
            x=x_question,
            y=y_top,
            width=card_w,
            height=card_h + 24,
            fill="#dbeafe",
            stroke="#2563eb",
            badge="input",
        )
        + "</g>"
    )

    last_flow_anchor = (x_question + card_w, y_top + 70)

    # Retrieve attempts with candidates
    for idx, retrieve in enumerate(retrieves, start=1):
        x = x_retrieve + (idx - 1) * 90
        y = y_top + (idx - 1) * 34
        query = retrieve.get("query", "")

        retrieve_node = f"retrieve_{idx}"
        svg_parts.append(
            f'<g filter="url(#shadow)">'
            + svg_card(
                node_id=retrieve_node,
                title=f"Retrieve attempt {idx}",
                subtitle=query,
                x=x,
                y=y,
                width=card_w,
                height=card_h + 10,
                fill="#dcfce7",
                stroke="#16a34a",
                badge="FAISS",
            )
            + "</g>"
        )

        svg_parts.append(
            svg_arrow(
                last_flow_anchor,
                (x, y + 64),
                label="query" if idx == 1 else "retry",
            )
        )

        documents = retrieve.get("documents", [])[:max_candidates]

        for doc_idx, doc in enumerate(documents, start=1):
            cy = y_candidates + (doc_idx - 1) * (card_h + gap_y)
            score = doc.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            title = f"Candidate {doc.get('rank')}"
            subtitle = (
                f"{doc.get('document_name')} | p. {doc.get('page_start')}-{doc.get('page_end')} | "
                f"score {score_text} | {doc.get('chunk_id')}"
            )

            svg_parts.append(
                svg_card(
                    node_id=f"candidate_{idx}_{doc_idx}",
                    title=title,
                    subtitle=subtitle,
                    x=x,
                    y=cy,
                    width=card_w,
                    height=card_h,
                    fill="#f0fdf4",
                    stroke="#86efac",
                    badge="chunk",
                )
            )

            svg_parts.append(
                svg_arrow(
                    (x + card_w / 2, y + card_h + 10),
                    (x + card_w / 2, cy),
                    label="candidate" if doc_idx == 1 else "",
                    color="#94a3b8",
                )
            )

        last_flow_anchor = (x + card_w, y + 64)

    # Evaluation card, using latest evaluation
    evaluation = evaluations[-1].get("evaluation", {}) if evaluations else {}
    decision = evaluation.get("decision", "n/a")
    relevance = evaluation.get("relevance_score", "n/a")
    reason = evaluation.get("reason", "")

    eval_fill = "#dcfce7" if decision == "pass" else "#ffedd5"
    eval_stroke = "#16a34a" if decision == "pass" else "#f97316"

    svg_parts.append(
        f'<g filter="url(#shadow)">'
        + svg_card(
            node_id="evaluation",
            title="Evaluation",
            subtitle=f"decision: {decision} | relevance: {relevance} | {reason}",
            x=x_evaluate,
            y=y_top,
            width=card_w,
            height=card_h + 26,
            fill=eval_fill,
            stroke=eval_stroke,
            badge="grade",
        )
        + "</g>"
    )

    svg_parts.append(
        svg_arrow(
            last_flow_anchor,
            (x_evaluate, y_top + 74),
            label="grade",
        )
    )

    # Rewrite card if happened
    if rewrites:
        rewrite = rewrites[-1].get("rewrite_result", {})
        rewritten_query = rewrite.get("rewritten_query", "")
        correction_reason = rewrite.get("correction_reason", "")

        svg_parts.append(
            f'<g filter="url(#shadow)">'
            + svg_card(
                node_id="rewrite",
                title="Query rewrite",
                subtitle=f"{rewritten_query} | {correction_reason}",
                x=x_evaluate,
                y=y_top + 210,
                width=card_w,
                height=card_h + 24,
                fill="#fed7aa",
                stroke="#ea580c",
                badge="rewrite",
            )
            + "</g>"
        )

        svg_parts.append(
            svg_arrow(
                (x_evaluate + card_w / 2, y_top + card_h + 26),
                (x_evaluate + card_w / 2, y_top + 210),
                label="correct",
                color="#ea580c",
            )
        )

    # Final sources summary
    svg_parts.append(
        f'<g filter="url(#shadow)">'
        + svg_card(
            node_id="final_sources",
            title="Final selected sources",
            subtitle=f"{len(sources)} chunks selected for answer grounding",
            x=x_final,
            y=y_top,
            width=card_w,
            height=card_h,
            fill="#ede9fe",
            stroke="#7c3aed",
            badge="context",
        )
        + "</g>"
    )

    svg_parts.append(
        svg_arrow(
            (x_evaluate + card_w, y_top + 74),
            (x_final, y_top + 60),
            label="selected",
            color="#7c3aed",
        )
    )

    for source_idx, source in enumerate(sources[:max_candidates], start=1):
        sy = y_candidates + (source_idx - 1) * (card_h + gap_y)
        title = f"Source {source_idx}"
        subtitle = (
            f"{source.get('document_name')} | p. {source.get('page_start')}-{source.get('page_end')} | "
            f"chunk {source.get('chunk_number')}/{source.get('total_chunks')} | "
            f"{source.get('document_type')}"
        )

        svg_parts.append(
            svg_card(
                node_id=f"source_{source_idx}",
                title=title,
                subtitle=subtitle,
                x=x_final,
                y=sy,
                width=card_w,
                height=card_h,
                fill="#f5f3ff",
                stroke="#c4b5fd",
                badge="cited",
            )
        )

        svg_parts.append(
            svg_arrow(
                (x_final + card_w / 2, y_top + card_h),
                (x_final + card_w / 2, sy),
                label="cited" if source_idx == 1 else "",
                color="#a78bfa",
            )
        )

    # Answer card
    svg_parts.append(
        f'<g filter="url(#shadow)">'
        + svg_card(
            node_id="answer",
            title="Answer",
            subtitle=answer,
            x=x_answer,
            y=y_top,
            width=340,
            height=card_h + 78,
            fill="#cffafe",
            stroke="#0891b2",
            badge="final",
        )
        + "</g>"
    )

    svg_parts.append(
        svg_arrow(
            (x_final + card_w, y_top + 60),
            (x_answer, y_top + 78),
            label="grounded",
            color="#0891b2",
        )
    )

    # Legend
    legend_y = canvas_height - 20
    svg_parts.append(
        svg_text_block(
            ["Legend: blue=input | green=retrieval | orange=correction | purple=sources | cyan=final answer"],
            40,
            legend_y,
            font_size=13,
            color="#475569",
            weight="700",
            line_height=16,
        )
    )

    svg_parts.append("</svg>")

    svg_content = "\n".join(svg_parts)

    Path(output_svg).write_text(svg_content, encoding="utf-8")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Dynamic Answer Graph - Clinical RAG</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      font-family: Inter, Arial, sans-serif;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 12px 18px;
      background: #0f172a;
      color: white;
      font-weight: 700;
      box-shadow: 0 2px 12px rgba(0,0,0,.18);
    }}
    .canvas {{
      padding: 18px;
      overflow: auto;
    }}
    svg {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 14px;
    }}
  </style>
</head>
<body>
  <div class="toolbar">Dynamic Answer Graph - Clinical RAG Evidence Path</div>
  <div class="canvas">
    {svg_content}
  </div>
</body>
</html>
"""
    Path(output_html).write_text(html_content, encoding="utf-8")

    write_answer_dot_file_from_svg_state(
        state=state,
        output_dot=output_dot,
    )

    png_generated = False

    try:
        import cairosvg

        cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            write_to=output_png,
            dpi=dpi,
        )
        png_generated = True
    except Exception as exc:
        print(
            "Aviso: PNG não foi gerado porque cairosvg não está disponível ou falhou. "
            "Use o SVG/HTML, que são os formatos principais desta versão."
        )
        print(f"Detalhe: {exc}")

    print(f"Grafo dinâmico SVG gerado: {output_svg}")
    print(f"Grafo dinâmico HTML gerado: {output_html}")
    print(f"Grafo dinâmico DOT gerado: {output_dot}")

    if png_generated:
        print(f"Grafo dinâmico PNG gerado: {output_png}")


def write_answer_dot_file_from_svg_state(
    state: RAGState,
    output_dot: str,
) -> None:
    """Salva DOT simplificado do grafo dinâmico."""
    trace = state.get("trace", [])
    sources = state.get("sources", [])

    lines = [
        "digraph DynamicAnswerGraph {",
        "  rankdir=LR;",
        '  graph [fontname="Arial", bgcolor="white"];',
        '  node [shape=box, style="rounded,filled", fontname="Arial", color="#334155"];',
        '  edge [fontname="Arial", color="#64748b"];',
        '  "Question" [fillcolor="#dbeafe"];',
    ]

    previous = "Question"

    retrieve_count = 0
    evaluation_count = 0
    rewrite_count = 0

    for step in trace:
        node = step.get("node")

        if node == "retrieve":
            retrieve_count += 1
            name = f"Retrieve {retrieve_count}"
            label = shorten_text(step.get("query"), 60).replace('"', "'")
            lines.append(f'  "{name}" [label="{name}\\n{label}", fillcolor="#dcfce7"];')
            lines.append(f'  "{previous}" -> "{name}" [label="query"];')
            previous = name

        elif node == "evaluate":
            evaluation_count += 1
            evaluation = step.get("evaluation", {})
            name = f"Evaluate {evaluation_count}"
            decision = evaluation.get("decision")
            relevance = evaluation.get("relevance_score")
            lines.append(
                f'  "{name}" [label="{name}\\ndecision: {decision}\\nrelevance: {relevance}", fillcolor="#ffedd5"];'
            )
            lines.append(f'  "{previous}" -> "{name}" [label="grade"];')
            previous = name

        elif node == "rewrite":
            rewrite_count += 1
            name = f"Rewrite {rewrite_count}"
            lines.append(f'  "{name}" [fillcolor="#fed7aa"];')
            lines.append(f'  "{previous}" -> "{name}" [label="correct"];')
            previous = name

    lines.append('  "Final Sources" [fillcolor="#ede9fe"];')
    lines.append(f'  "{previous}" -> "Final Sources" [label="selected"];')

    for idx, source in enumerate(sources[:8], start=1):
        label = (
            f"Source {idx}\\n"
            f"{shorten_text(source.get('document_name'), 28)}\\n"
            f"p. {source.get('page_start')}-{source.get('page_end')}"
        ).replace('"', "'")
        source_node = f"Source {idx}"
        lines.append(f'  "{source_node}" [label="{label}", fillcolor="#f5f3ff"];')
        lines.append(f'  "Final Sources" -> "{source_node}" [label="cited"];')

    lines.append('  "Answer" [fillcolor="#cffafe"];')
    lines.append('  "Final Sources" -> "Answer" [label="grounded"];')
    lines.append("}")

    Path(output_dot).write_text("\n".join(lines), encoding="utf-8")

def print_results(state: RAGState) -> None:
    """Mostra saída final do grafo."""
    print("\nResposta:")
    print(state.get("answer", ""))

    print("\nFontes estruturadas:")
    print(json.dumps(state.get("sources", []), ensure_ascii=False, indent=2))

    print("\nTrace LangGraph:")
    print(json.dumps(state.get("trace", []), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa RAG agentic com LangGraph, FAISS, Bedrock Titan e Claude."
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
        "--index-mode",
        choices=["baseline", "semantic"],
        default="baseline",
        help="Índice usado pelo grafo. Default: baseline.",
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
        help="Quantidade de chunks recuperados por tentativa.",
    )

    parser.add_argument(
        "--min-vector-score",
        type=float,
        default=DEFAULT_MIN_VECTOR_SCORE,
        help="Score vetorial mínimo usado pela avaliação local.",
    )

    parser.add_argument(
        "--min-relevance-score",
        type=float,
        default=DEFAULT_MIN_RELEVANCE_SCORE,
        help="Score mínimo do avaliador para responder sem correção.",
    )

    parser.add_argument(
        "--max-corrections",
        type=int,
        default=DEFAULT_MAX_CORRECTIONS,
        help="Número máximo de ciclos rewrite/retrieve.",
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
        "--no-evaluator-llm",
        action="store_true",
        help="Usa avaliação local em vez do avaliador LLM.",
    )

    parser.add_argument(
        "--no-rewrite-llm",
        action="store_true",
        help="Usa rewrite local em vez do rewrite LLM.",
    )

    parser.add_argument(
        "--use-local-cache",
        action="store_true",
        help=(
            "Usa arquivos FAISS locais se existirem. "
            "Por padrão, sempre baixa os artefatos do S3."
        ),
    )

    parser.add_argument(
        "--graph-png",
        default="langgraph_rag_graph.png",
        help="Arquivo PNG de saída para visualização do grafo.",
    )

    parser.add_argument(
        "--graph-dot",
        default="langgraph_rag_graph.dot",
        help="Arquivo DOT de saída para visualização do grafo em Graphviz.",
    )

    parser.add_argument(
        "--no-graph-export",
        action="store_true",
        help="Não exporta a visualização estática do workflow.",
    )

    parser.add_argument(
        "--answer-graph-png",
        default="langgraph_answer_graph.png",
        help="Arquivo PNG do grafo dinâmico da resposta.",
    )

    parser.add_argument(
        "--answer-graph-dot",
        default="langgraph_answer_graph.dot",
        help="Arquivo DOT do grafo dinâmico da resposta.",
    )

    parser.add_argument(
        "--answer-graph-svg",
        default="langgraph_answer_graph.svg",
        help="Arquivo SVG do grafo dinâmico da resposta.",
    )

    parser.add_argument(
        "--answer-graph-html",
        default="langgraph_answer_graph.html",
        help="Arquivo HTML do grafo dinâmico da resposta.",
    )

    parser.add_argument(
        "--no-answer-graph",
        action="store_true",
        help="Não exporta o grafo dinâmico da resposta.",
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    print("Iniciando LangGraph RAG")
    print(f"Bucket: {args.bucket}")
    print(f"Index mode: {args.index_mode}")
    print(f"Top-K: {args.top_k}")
    print(f"Max corrections: {args.max_corrections}")
    print(f"Usar avaliador LLM: {not args.no_evaluator_llm}")
    print(f"Usar rewrite LLM: {not args.no_rewrite_llm}")
    print(f"Usar cache local: {args.use_local_cache}")

    if not args.no_graph_export:
        export_langgraph_visual(
            output_png=args.graph_png,
            output_dot=args.graph_dot,
        )

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=args.bucket,
        index_mode=args.index_mode,
        use_local_cache=args.use_local_cache,
    )

    runtime = RuntimeClients(
        s3_client=s3_client,
        bedrock_client=bedrock_client,
        index=index,
        metadata=metadata,
        args=args,
    )

    app = build_graph(runtime)

    initial_state: RAGState = {
        "question": args.question,
        "current_query": args.question,
        "index_mode": args.index_mode,
        "documents": [],
        "evaluation": {},
        "answer": "",
        "sources": [],
        "correction_count": 0,
        "trace": [],
    }

    final_state = app.invoke(initial_state)

    if not args.no_answer_graph:
        export_answer_graph_visual(
            state=final_state,
            output_png=args.answer_graph_png,
            output_dot=args.answer_graph_dot,
            output_svg=args.answer_graph_svg,
            output_html=args.answer_graph_html,
        )

    print_results(final_state)


if __name__ == "__main__":
    main()

# OBSERVAÇÃO IMPORTANTE
# SVG = imagem vetorial baseado em XML. Não usa pixels (como o PNG ou JPG), mas sim coordenadas matemáticas para desenhar linhas e formas (zoom NÃO muda a qualidade)
# MATPLOTLIB = representação visual de matrizes numéricas 2D ou 3D usando a função imshow() que plota valores como pixels coloridos (zoom muda SIM a qualidade)