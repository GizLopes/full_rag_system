# OBJETIVO PRINCIPAL
# Gerar múltiplas queries, executar FAISS para cada uma e
# consolidar resultados via RRF para ampliar cobertura do retrieval

import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

DEFAULT_QUERY_COUNT = 5
DEFAULT_RETRIEVAL_K_PER_QUERY = 10
DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_MAX_TOKENS = 1024
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
    """Normaliza texto para extração simples de entidades."""
    text = text or ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def normalize_for_matching(text: str) -> str:
    """Normaliza texto para matching sem acento."""
    text = normalize_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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


def extract_patient_id(question: str) -> str | None:
    """Extrai patient_id quando explícito."""
    match = re.search(r"\bP\d{3}\b", question, flags=re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return None


def extract_patient_name(question: str) -> str | None:
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
    """Extrai JSON retornado pelo LLM."""
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
        raise ValueError("Resposta do LLM não contém JSON válido.")

    return json.loads(match.group(0))


def build_query_generation_prompt(
    question: str,
    query_count: int,
    patient_id: str | None,
    patient_name: str | None,
    document_types: List[str],
    clinical_terms: List[str],
) -> str:
    """Cria prompt para gerar variações de query para RAG-Fusion."""
    return f"""
Você é um componente de Multi-Query para RAG-Fusion em um sistema clínico de treinamento.

Sua tarefa é criar variações úteis da pergunta para aumentar a cobertura do retrieval.

Contexto:
- O corpus tem documentos sintéticos clínicos.
- Tipos documentais possíveis: consultas_ambulatoriais, hemograma_e_bioquimica, alta_hospitalar, parecer_cardiologista, ressonancia_coluna.
- A busca será feita em FAISS vetorial e os rankings serão combinados por Reciprocal Rank Fusion.
- Não responda à pergunta clínica.

Pergunta original:
{question}

Entidades detectadas por regra:
patient_id: {patient_id}
patient_name: {patient_name}
document_types: {json.dumps(document_types, ensure_ascii=False)}
clinical_terms: {json.dumps(clinical_terms, ensure_ascii=False)}

Regras:
1. Gere exatamente {query_count} queries.
2. A primeira query deve preservar a pergunta original normalizada.
3. As demais devem variar foco, termos clínicos, paciente, tipo documental e sinônimos.
4. Não invente resultado clínico.
5. Não invente patient_id.
6. Não gere queries longas demais.
7. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ],
  "detected_entities": {{
    "patient_id": null,
    "patient_name": null,
    "document_types": [],
    "clinical_terms": []
  }},
  "generation_summary": "Resumo curto do objetivo das variações."
}}
""".strip()


def invoke_claude_for_queries(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Invoca Claude para gerar queries alternativas."""
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
        raise RuntimeError("Claude não retornou conteúdo para geração de queries.")

    raw_text = content[0].get("text", "").strip()

    if not raw_text:
        raise RuntimeError("Claude retornou texto vazio para geração de queries.")

    return extract_json_from_text(raw_text)


def build_rule_based_queries(
    question: str,
    query_count: int,
) -> Dict[str, Any]:
    """Gera queries alternativas por regras locais."""
    normalized_question = normalize_text(question)
    patient_id = extract_patient_id(normalized_question)
    patient_name = extract_patient_name(normalized_question)
    document_types = infer_document_types(normalized_question)
    clinical_terms = extract_clinical_terms(normalized_question)

    queries = [normalized_question]

    if patient_name and clinical_terms:
        queries.append(f"{patient_name} {' '.join(clinical_terms)}")

    if patient_id and clinical_terms:
        queries.append(f"{patient_id} {' '.join(clinical_terms)}")

    if document_types and clinical_terms:
        queries.append(f"{' '.join(document_types)} {' '.join(clinical_terms)}")

    if patient_name and document_types:
        queries.append(f"{patient_name} {' '.join(document_types)}")

    if clinical_terms:
        queries.append(" ".join(clinical_terms))

    if patient_name:
        queries.append(patient_name)

    if patient_id:
        queries.append(patient_id)

    deduplicated = []

    for query in queries:
        query = normalize_text(query)

        if query and query.lower() not in [item.lower() for item in deduplicated]:
            deduplicated.append(query)

    return {
        "queries": deduplicated[:query_count],
        "detected_entities": {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "document_types": document_types,
            "clinical_terms": clinical_terms,
        },
        "generation_summary": "Queries geradas por regras locais para ampliar cobertura do retrieval.",
    }


def generate_queries(
    bedrock_client,
    question: str,
    query_count: int,
    model_id: str,
    max_tokens: int,
    temperature: float,
    use_llm: bool,
) -> Dict[str, Any]:
    """Gera múltiplas queries para RAG-Fusion."""
    normalized_question = normalize_text(question)
    patient_id = extract_patient_id(normalized_question)
    patient_name = extract_patient_name(normalized_question)
    document_types = infer_document_types(normalized_question)
    clinical_terms = extract_clinical_terms(normalized_question)

    if not use_llm:
        return build_rule_based_queries(
            question=normalized_question,
            query_count=query_count,
        )

    prompt = build_query_generation_prompt(
        question=normalized_question,
        query_count=query_count,
        patient_id=patient_id,
        patient_name=patient_name,
        document_types=document_types,
        clinical_terms=clinical_terms,
    )

    try:
        result = invoke_claude_for_queries(
            bedrock_client=bedrock_client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except (ClientError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Aviso: falha ao gerar queries com LLM. Usando fallback local. Erro: {exc}")
        result = build_rule_based_queries(
            question=normalized_question,
            query_count=query_count,
        )

    queries = result.get("queries") or []

    if normalized_question.lower() not in [query.lower() for query in queries]:
        queries.insert(0, normalized_question)

    clean_queries = []

    for query in queries:
        query = normalize_text(str(query))

        if query and query.lower() not in [item.lower() for item in clean_queries]:
            clean_queries.append(query)

    result["queries"] = clean_queries[:query_count]

    detected_entities = result.get("detected_entities") or {}
    detected_entities.setdefault("patient_id", patient_id)
    detected_entities.setdefault("patient_name", patient_name)
    detected_entities.setdefault("document_types", document_types)
    detected_entities.setdefault("clinical_terms", clinical_terms)
    result["detected_entities"] = detected_entities

    return result


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


def search_faiss_for_query(
    index: faiss.Index,
    metadata: List[Dict],
    query: str,
    query_number: int,
    bedrock_client,
    embedding_model_id: str,
    retrieval_k: int,
) -> List[Dict]:
    """Executa FAISS para uma query individual."""
    query_vector = embed_query(
        bedrock_client=bedrock_client,
        query=query,
        model_id=embedding_model_id,
    )

    search_k = min(retrieval_k, len(metadata))
    scores, positions = index.search(query_vector, search_k)

    results = []

    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        if position < 0:
            continue

        item = metadata[position].copy()
        item["query_number"] = query_number
        item["query_text"] = query
        item["query_rank"] = rank
        item["query_score"] = float(score)
        item["position"] = int(position)

        results.append(item)

    return results


def reciprocal_rank_fusion(
    per_query_results: List[List[Dict]],
    top_k: int,
    rrf_k: int,
) -> List[Dict]:
    """Combina múltiplos rankings usando Reciprocal Rank Fusion."""
    fused: Dict[str, Dict] = {}

    for query_results in per_query_results:
        for item in query_results:
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = item.copy()
                fused[chunk_id]["rrf_score"] = 0.0
                fused[chunk_id]["matched_queries"] = []
                fused[chunk_id]["per_query_evidence"] = []
                fused[chunk_id]["best_query_score"] = item.get("query_score", 0)

            rank = item.get("query_rank")
            query_number = item.get("query_number")
            query_text = item.get("query_text")
            query_score = item.get("query_score", 0)

            if rank:
                fused[chunk_id]["rrf_score"] += 1 / (rrf_k + rank)

            fused[chunk_id]["matched_queries"].append(query_number)
            fused[chunk_id]["per_query_evidence"].append({
                "query_number": query_number,
                "query_text": query_text,
                "query_rank": rank,
                "query_score": query_score,
            })

            if query_score > fused[chunk_id].get("best_query_score", 0):
                fused[chunk_id]["best_query_score"] = query_score

    fused_results = list(fused.values())
    fused_results.sort(
        key=lambda item: (
            item.get("rrf_score", 0),
            item.get("best_query_score", 0),
        ),
        reverse=True,
    )

    for rank, item in enumerate(fused_results[:top_k], start=1):
        item["fusion_rank"] = rank
        item["matched_queries"] = sorted(set(item.get("matched_queries", [])))

    return fused_results[:top_k]


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
        rrf_score = item.get("rrf_score", 0)
        best_query_score = item.get("best_query_score", 0)
        matched_queries = item.get("matched_queries", [])
        text = item.get("text", "")

        citation = (
            f"Fonte: {source} | "
            f"Tipo: {document_type} | "
            f"Página(s): {page_start}-{page_end} | "
            f"Chunk: {chunk_number}/{total_chunks} | "
            f"RRF score: {rrf_score:.4f} | "
            f"Best vector score: {best_query_score:.4f} | "
            f"Matched queries: {matched_queries}"
        )

        blocks.append(
            f"[{citation}]\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_answer_prompt(question: str, queries: List[str], context: str) -> str:
    """Cria prompt final para resposta com RAG-Fusion."""
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
8. Use a pergunta original como prioridade; as queries alternativas servem apenas para recuperação.

PERGUNTA ORIGINAL:
{question}

QUERIES USADAS NO RAG-FUSION:
{json.dumps(queries, ensure_ascii=False, indent=2)}

CONTEXTO:
{context}

RESPOSTA:
""".strip()


def invoke_claude_for_answer(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoca Claude para resposta final."""
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
            "fusion_rank": item.get("fusion_rank"),
            "rrf_score": item.get("rrf_score"),
            "best_query_score": item.get("best_query_score"),
            "matched_queries": item.get("matched_queries"),
            "per_query_evidence": item.get("per_query_evidence"),
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


def print_query_results(queries: List[str], per_query_results: List[List[Dict]]) -> None:
    """Mostra top resultados por query."""
    print("\nQueries geradas:")

    for idx, query in enumerate(queries, start=1):
        print(f"{idx}. {query}")

    for idx, results in enumerate(per_query_results, start=1):
        print("\n" + "=" * 80)
        print(f"Resultados da query {idx}: {queries[idx - 1]}")

        for item in results[:5]:
            print("\n" + "-" * 80)
            print(f"Rank na query: {item.get('query_rank')}")
            print(f"Score vetorial: {item.get('query_score', 0):.4f}")
            print(f"Documento: {item.get('document_name')}")
            print(f"Tipo: {item.get('document_type')}")
            print(f"Chunk: {item.get('chunk_number')} de {item.get('total_chunks')}")
            print(f"Páginas: {item.get('page_start')} - {item.get('page_end')}")
            print(f"Estratégia: {item.get('chunk_strategy')}")


def print_fusion_results(results: List[Dict]) -> None:
    """Mostra resultados finais do RAG-Fusion."""
    print("\n" + "=" * 80)
    print("Top resultados RAG-Fusion")

    if not results:
        print("Nenhum resultado.")
        return

    for item in results:
        print("\n" + "-" * 80)
        print(f"Fusion rank: {item.get('fusion_rank')}")
        print(f"RRF score: {item.get('rrf_score', 0):.4f}")
        print(f"Best vector score: {item.get('best_query_score', 0):.4f}")
        print(f"Matched queries: {item.get('matched_queries')}")
        print(f"Documento: {item.get('document_name')}")
        print(f"Tipo: {item.get('document_type')}")
        print(f"Chunk: {item.get('chunk_number')} de {item.get('total_chunks')}")
        print(f"Páginas: {item.get('page_start')} - {item.get('page_end')}")
        print(f"Estratégia: {item.get('chunk_strategy')}")
        print(f"Fonte: {item.get('s3_uri')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa RAG-Fusion com multi-query retrieval, FAISS e RRF."
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
        help="Índice usado no RAG-Fusion. Default: baseline.",
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
        "--query-count",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help="Quantidade de queries alternativas para RAG-Fusion.",
    )

    parser.add_argument(
        "--retrieval-k-per-query",
        type=int,
        default=DEFAULT_RETRIEVAL_K_PER_QUERY,
        help="Quantidade de chunks recuperados por query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Quantidade final de chunks após RRF.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="Constante do Reciprocal Rank Fusion.",
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
        "--no-query-llm",
        action="store_true",
        help="Gera queries alternativas apenas por regras locais.",
    )

    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Executa apenas retrieval fusion, sem chamar Claude para resposta final.",
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

    index_prefix, index_file, metadata_file = resolve_index_config(args.index_mode)

    print("Iniciando RAG-Fusion")
    print(f"Bucket: {args.bucket}")
    print(f"Index mode: {args.index_mode}")
    print(f"Index prefix: {index_prefix}")
    print(f"Query count: {args.query_count}")
    print(f"Retrieval K por query: {args.retrieval_k_per_query}")
    print(f"Top-K final: {args.top_k}")
    print(f"RRF K: {args.rrf_k}")
    print(f"Usar cache local: {args.use_local_cache}")

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=args.bucket,
        index_prefix=index_prefix,
        index_file=index_file,
        metadata_file=metadata_file,
        use_local_cache=args.use_local_cache,
    )

    query_generation = generate_queries(
        bedrock_client=bedrock_client,
        question=args.question,
        query_count=args.query_count,
        model_id=args.llm_model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_llm=not args.no_query_llm,
    )

    queries = query_generation.get("queries", [])

    if not queries:
        print("Nenhuma query gerada. Encerrando.")
        return

    per_query_results = []

    for query_number, query in enumerate(queries, start=1):
        results = search_faiss_for_query(
            index=index,
            metadata=metadata,
            query=query,
            query_number=query_number,
            bedrock_client=bedrock_client,
            embedding_model_id=args.embedding_model_id,
            retrieval_k=args.retrieval_k_per_query,
        )
        per_query_results.append(results)

    fused_results = reciprocal_rank_fusion(
        per_query_results=per_query_results,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )

    print("\nPergunta original:")
    print(args.question)

    print("\nQuery generation:")
    print(json.dumps(query_generation, ensure_ascii=False, indent=2))

    print_query_results(
        queries=queries,
        per_query_results=per_query_results,
    )

    print_fusion_results(fused_results)

    if not fused_results:
        print("\nResposta:")
        print("Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança.")
        return

    context = build_context(fused_results)

    if args.show_context:
        print("\nContexto enviado ao LLM:")
        print(context)

    if args.no_answer:
        print("\nFontes estruturadas:")
        print(json.dumps(build_sources(fused_results), ensure_ascii=False, indent=2))
        return

    prompt = build_answer_prompt(
        question=args.question,
        queries=queries,
        context=context,
    )

    answer = invoke_claude_for_answer(
        bedrock_client=bedrock_client,
        model_id=args.llm_model_id,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nResposta:")
    print(answer)

    print("\nFontes estruturadas:")
    print(json.dumps(build_sources(fused_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()