# OBJETIVO PRINCIPAL
# Executar Corrective RAG com avaliação de contexto, query rewrite, nova recuperação e resposta final com fontes.

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
DEFAULT_RETRY_TOP_K = 8
DEFAULT_MIN_VECTOR_SCORE = 0.25
DEFAULT_MIN_RELEVANCE_SCORE = 0.55
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
    retrieval_stage: str,
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
        item["retrieval_stage"] = retrieval_stage
        item["index_mode"] = index_mode
        item["query_used"] = question
        results.append(item)

    return results


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
        stage = item.get("retrieval_stage")
        index_mode = item.get("index_mode")
        text = item.get("text", "")

        citation = (
            f"Fonte: {source} | "
            f"Tipo: {document_type} | "
            f"Página(s): {page_start}-{page_end} | "
            f"Chunk: {chunk_number}/{total_chunks} | "
            f"Score FAISS: {score:.4f} | "
            f"Stage: {stage} | "
            f"Index: {index_mode}"
        )

        blocks.append(
            f"[{citation}]\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_evaluation_prompt(
    question: str,
    results: List[Dict],
) -> str:
    """Cria prompt para avaliar se o contexto recuperado é suficiente."""
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
Você é um avaliador de retrieval para um sistema Corrective RAG clínico de treinamento.

Sua tarefa é avaliar se os chunks recuperados são suficientes para responder à pergunta original.

Pergunta:
{question}

Chunks recuperados:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Critérios:
1. O contexto deve conter evidência direta para responder.
2. Se a pergunta pedir resultado de exame, o chunk precisa conter paciente e valor do exame.
3. Se a evidência estiver incompleta, recomende rewrite.
4. Se os chunks forem irrelevantes, recomende rewrite.
5. Não responda à pergunta clínica.
6. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "decision": "pass",
  "relevance_score": 0.0,
  "relevant_chunk_ids": [],
  "reason": "explicação curta",
  "rewrite_suggestion": "consulta reescrita ou null"
}}

Valores permitidos para decision:
- "pass": contexto suficiente
- "rewrite": precisa reescrever e recuperar novamente
- "insufficient": contexto claramente insuficiente
""".strip()


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


def local_evaluate_retrieval(
    question: str,
    results: List[Dict],
    min_vector_score: float,
) -> Dict[str, Any]:
    """Fallback local de avaliação do retrieval."""
    if not results:
        return {
            "decision": "rewrite",
            "relevance_score": 0.0,
            "relevant_chunk_ids": [],
            "reason": "Nenhum chunk recuperado.",
            "rewrite_suggestion": build_local_rewrite(question),
        }

    best_score = max(item.get("score", 0) for item in results)
    question_norm = normalize_for_matching(question)
    patient_id = extract_patient_id(question)
    patient_name = extract_patient_name(question)
    clinical_terms = extract_clinical_terms(question)

    relevant_chunk_ids = []

    for item in results:
        text_norm = normalize_for_matching(item.get("text", ""))
        metadata_norm = normalize_for_matching(
            " ".join(
                str(value)
                for value in [
                    item.get("document_name"),
                    item.get("document_type"),
                    item.get("patient_id"),
                    item.get("patient_name"),
                    item.get("clinical_section"),
                ]
                if value
            )
        )

        searchable = f"{text_norm} {metadata_norm}"

        patient_match = True
        if patient_id:
            patient_match = patient_id.lower() in searchable

        if patient_name:
            patient_match = normalize_for_matching(patient_name) in searchable

        term_match = True
        if clinical_terms:
            term_match = any(normalize_for_matching(term) in searchable for term in clinical_terms)

        if patient_match and term_match:
            relevant_chunk_ids.append(item.get("chunk_id"))

    if relevant_chunk_ids and best_score >= min_vector_score:
        return {
            "decision": "pass",
            "relevance_score": float(best_score),
            "relevant_chunk_ids": relevant_chunk_ids,
            "reason": "Heurística local encontrou paciente/termo clínico nos chunks recuperados.",
            "rewrite_suggestion": None,
        }

    return {
        "decision": "rewrite",
        "relevance_score": float(best_score),
        "relevant_chunk_ids": relevant_chunk_ids,
        "reason": "Score ou correspondência textual insuficiente para responder com segurança.",
        "rewrite_suggestion": build_local_rewrite(question),
    }


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
        print(f"Aviso: falha no avaliador LLM. Usando avaliação local. Erro: {exc}")
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


def build_rewrite_prompt(
    question: str,
    evaluation: Dict[str, Any],
) -> str:
    """Cria prompt para reescrever consulta após falha no retrieval."""
    patient_id = extract_patient_id(question)
    patient_name = extract_patient_name(question)
    document_types = infer_document_types(question)
    clinical_terms = extract_clinical_terms(question)

    return f"""
Você é um componente de correção de query para Corrective RAG clínico.

A primeira recuperação foi avaliada como insuficiente ou parcial. Reescreva a pergunta para melhorar o próximo retrieval.

Pergunta original:
{question}

Avaliação do retrieval:
{json.dumps(evaluation, ensure_ascii=False, indent=2)}

Entidades detectadas por regra:
patient_id: {patient_id}
patient_name: {patient_name}
document_types: {json.dumps(document_types, ensure_ascii=False)}
clinical_terms: {json.dumps(clinical_terms, ensure_ascii=False)}

Regras:
1. Não responda à pergunta clínica.
2. Não invente valor clínico.
3. Preserve paciente, exame, documento e termo clínico quando existirem.
4. Se houver pergunta sobre exame, priorize termos laboratoriais e document_type hemograma_e_bioquimica.
5. Gere uma única query curta e objetiva.
6. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "rewritten_query": "...",
  "target_document_types": [],
  "clinical_terms": [],
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
    if evaluation.get("rewrite_suggestion"):
        suggested = evaluation["rewrite_suggestion"]
    else:
        suggested = build_local_rewrite(question)

    if not use_llm:
        return {
            "rewritten_query": suggested,
            "target_document_types": infer_document_types(question),
            "clinical_terms": extract_clinical_terms(question),
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
            "rewritten_query": suggested,
            "target_document_types": infer_document_types(question),
            "clinical_terms": extract_clinical_terms(question),
            "correction_reason": "Fallback local aplicado após falha do rewrite LLM.",
        }

    if not result.get("rewritten_query"):
        result["rewritten_query"] = suggested

    return result


def select_final_results(
    initial_results: List[Dict],
    retry_results: List[Dict],
    fallback_results: List[Dict],
    final_evaluation: Dict[str, Any],
    top_k: int,
) -> List[Dict]:
    """Seleciona chunks finais priorizando etapa aprovada pela avaliação."""
    relevant_chunk_ids = set(final_evaluation.get("relevant_chunk_ids") or [])

    candidate_groups = [
        fallback_results,
        retry_results,
        initial_results,
    ]

    for group in candidate_groups:
        if not group:
            continue

        if relevant_chunk_ids:
            filtered = [
                item
                for item in group
                if item.get("chunk_id") in relevant_chunk_ids
            ]

            if filtered:
                return filtered[:top_k]

        return group[:top_k]

    return []


def build_answer_prompt(
    question: str,
    context: str,
    correction_trace: Dict[str, Any],
) -> str:
    """Cria prompt final para resposta Corrective RAG."""
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
8. Ignore chunks irrelevantes mesmo que tenham score alto.

PERGUNTA ORIGINAL:
{question}

TRACE CORRECTIVE RAG:
{json.dumps(correction_trace, ensure_ascii=False, indent=2)}

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
            "retrieval_stage": item.get("retrieval_stage"),
            "index_mode": item.get("index_mode"),
            "query_used": item.get("query_used"),
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


def print_results(title: str, results: List[Dict]) -> None:
    """Mostra resultados recuperados."""
    print("\n" + "=" * 80)
    print(title)

    if not results:
        print("Nenhum resultado.")
        return

    for item in results:
        print("\n" + "-" * 80)
        print(f"Rank: {item.get('rank')}")
        print(f"Score FAISS: {item.get('score', 0):.4f}")
        print(f"Stage: {item.get('retrieval_stage')}")
        print(f"Index: {item.get('index_mode')}")
        print(f"Query: {item.get('query_used')}")
        print(f"Documento: {item.get('document_name')}")
        print(f"Tipo: {item.get('document_type')}")
        print(f"Chunk: {item.get('chunk_number')} de {item.get('total_chunks')}")
        print(f"Páginas: {item.get('page_start')} - {item.get('page_end')}")
        print(f"Estratégia: {item.get('chunk_strategy')}")
        print(f"Fonte: {item.get('s3_uri')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa Corrective RAG com avaliação, query rewrite e nova recuperação."
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
        help="Índice inicial usado no Corrective RAG. Default: baseline.",
    )

    parser.add_argument(
        "--semantic-fallback",
        action="store_true",
        help="Após rewrite insuficiente, tenta também o índice semântico.",
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
        help="Quantidade de chunks recuperados inicialmente.",
    )

    parser.add_argument(
        "--retry-top-k",
        type=int,
        default=DEFAULT_RETRY_TOP_K,
        help="Quantidade de chunks recuperados após query rewrite.",
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
        help="Score mínimo do avaliador para aceitar contexto.",
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
        help="Usa query rewrite local em vez do rewrite LLM.",
    )

    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Executa apenas recuperação corretiva, sem resposta final do Claude.",
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Mostra o contexto enviado ao LLM.",
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

    print("Iniciando Corrective RAG")
    print(f"Bucket: {args.bucket}")
    print(f"Index inicial: {args.index_mode}")
    print(f"Semantic fallback: {args.semantic_fallback}")
    print(f"Top-K inicial: {args.top_k}")
    print(f"Top-K retry: {args.retry_top_k}")
    print(f"Usar avaliador LLM: {not args.no_evaluator_llm}")
    print(f"Usar rewrite LLM: {not args.no_rewrite_llm}")
    print(f"Usar cache local: {args.use_local_cache}")

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=args.bucket,
        index_mode=args.index_mode,
        use_local_cache=args.use_local_cache,
    )

    initial_results = search_faiss(
        index=index,
        metadata=metadata,
        question=args.question,
        bedrock_client=bedrock_client,
        embedding_model_id=args.embedding_model_id,
        top_k=args.top_k,
        retrieval_stage="initial",
        index_mode=args.index_mode,
    )

    initial_evaluation = evaluate_retrieval(
        bedrock_client=bedrock_client,
        question=args.question,
        results=initial_results,
        model_id=args.llm_model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_llm=not args.no_evaluator_llm,
        min_vector_score=args.min_vector_score,
    )

    retry_results: List[Dict] = []
    retry_evaluation: Dict[str, Any] = {}
    fallback_results: List[Dict] = []
    fallback_evaluation: Dict[str, Any] = {}
    rewrite_result: Dict[str, Any] = {}

    final_evaluation = initial_evaluation

    should_rewrite = (
        initial_evaluation.get("decision") != "pass"
        or float(initial_evaluation.get("relevance_score", 0)) < args.min_relevance_score
    )

    if should_rewrite:
        rewrite_result = rewrite_query(
            bedrock_client=bedrock_client,
            question=args.question,
            evaluation=initial_evaluation,
            model_id=args.llm_model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_llm=not args.no_rewrite_llm,
        )

        rewritten_query = rewrite_result.get("rewritten_query") or build_local_rewrite(args.question)

        retry_results = search_faiss(
            index=index,
            metadata=metadata,
            question=rewritten_query,
            bedrock_client=bedrock_client,
            embedding_model_id=args.embedding_model_id,
            top_k=args.retry_top_k,
            retrieval_stage="rewrite_retry",
            index_mode=args.index_mode,
        )

        retry_evaluation = evaluate_retrieval(
            bedrock_client=bedrock_client,
            question=args.question,
            results=retry_results,
            model_id=args.llm_model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_llm=not args.no_evaluator_llm,
            min_vector_score=args.min_vector_score,
        )

        final_evaluation = retry_evaluation

        retry_still_bad = (
            retry_evaluation.get("decision") != "pass"
            or float(retry_evaluation.get("relevance_score", 0)) < args.min_relevance_score
        )

        if retry_still_bad and args.semantic_fallback and args.index_mode != "semantic":
            semantic_index, semantic_metadata = load_faiss_artifacts(
                s3_client=s3_client,
                bucket=args.bucket,
                index_mode="semantic",
                use_local_cache=args.use_local_cache,
            )

            fallback_results = search_faiss(
                index=semantic_index,
                metadata=semantic_metadata,
                question=rewritten_query,
                bedrock_client=bedrock_client,
                embedding_model_id=args.embedding_model_id,
                top_k=args.retry_top_k,
                retrieval_stage="semantic_fallback",
                index_mode="semantic",
            )

            fallback_evaluation = evaluate_retrieval(
                bedrock_client=bedrock_client,
                question=args.question,
                results=fallback_results,
                model_id=args.llm_model_id,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                use_llm=not args.no_evaluator_llm,
                min_vector_score=args.min_vector_score,
            )

            final_evaluation = fallback_evaluation

    final_results = select_final_results(
        initial_results=initial_results,
        retry_results=retry_results,
        fallback_results=fallback_results,
        final_evaluation=final_evaluation,
        top_k=args.top_k,
    )

    correction_trace = {
        "original_question": args.question,
        "initial_index_mode": args.index_mode,
        "initial_evaluation": initial_evaluation,
        "rewrite_applied": bool(rewrite_result),
        "rewrite_result": rewrite_result,
        "retry_evaluation": retry_evaluation,
        "semantic_fallback_applied": bool(fallback_results),
        "fallback_evaluation": fallback_evaluation,
        "final_evaluation": final_evaluation,
        "final_stage": final_results[0].get("retrieval_stage") if final_results else None,
    }

    print("\nPergunta:")
    print(args.question)

    print_results("Resultados iniciais", initial_results)

    print("\nAvaliação inicial:")
    print(json.dumps(initial_evaluation, ensure_ascii=False, indent=2))

    if rewrite_result:
        print("\nQuery rewrite:")
        print(json.dumps(rewrite_result, ensure_ascii=False, indent=2))
        print_results("Resultados após rewrite", retry_results)
        print("\nAvaliação após rewrite:")
        print(json.dumps(retry_evaluation, ensure_ascii=False, indent=2))

    if fallback_results:
        print_results("Resultados semantic fallback", fallback_results)
        print("\nAvaliação semantic fallback:")
        print(json.dumps(fallback_evaluation, ensure_ascii=False, indent=2))

    print_results("Resultados finais selecionados", final_results)

    print("\nTrace Corrective RAG:")
    print(json.dumps(correction_trace, ensure_ascii=False, indent=2))

    if not final_results:
        print("\nResposta:")
        print("Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança.")
        return

    context = build_context(final_results)

    if args.show_context:
        print("\nContexto enviado ao LLM:")
        print(context)

    if args.no_answer:
        print("\nFontes estruturadas:")
        print(json.dumps(build_sources(final_results), ensure_ascii=False, indent=2))
        return

    prompt = build_answer_prompt(
        question=args.question,
        context=context,
        correction_trace=correction_trace,
    )

    answer = invoke_claude_text(
        bedrock_client=bedrock_client,
        model_id=args.llm_model_id,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nResposta:")
    print(answer)

    print("\nFontes estruturadas:")
    print(json.dumps(build_sources(final_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()