# OBJETIVO PRINCIPAL
# Executar busca híbrida combinando FAISS vetorial, BM25 lexical e RRF,
# melhorando recuperação antes da resposta final com Claude

import argparse
import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
DEFAULT_VECTOR_CANDIDATES = 20
DEFAULT_BM25_CANDIDATES = 20
DEFAULT_RRF_K = 60
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0

STOPWORDS = {
    "a", "ao", "aos", "as", "da", "das", "de", "do", "dos", "e", "em",
    "foi", "o", "os", "ou", "para", "por", "qual", "quais", "que",
    "resultado", "sobre", "um", "uma", "no", "na", "nos", "nas", "com",
    "paciente", "pacientes", "me", "traga", "informe", "diga", "mostre",
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
    """Resolve os arquivos do índice baseline ou semantic."""
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
    """Normaliza texto para busca lexical."""
    text = text or ""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokeniza removendo stopwords curtas."""
    normalized = normalize_text(text)

    tokens = [
        token
        for token in normalized.split()
        if len(token) > 1 and token not in STOPWORDS
    ]

    return tokens


def build_searchable_text(item: Dict) -> str:
    """Monta texto de busca lexical a partir do chunk e metadados."""
    metadata = item.get("metadata") or {}

    fields = [
        item.get("text"),
        item.get("document_name"),
        item.get("document_type"),
        item.get("patient_id"),
        item.get("patient_name"),
        item.get("clinical_section"),
        metadata.get("document_name"),
        metadata.get("document_type"),
        metadata.get("patient_id"),
        metadata.get("patient_name"),
        metadata.get("clinical_section"),
    ]

    return " ".join(str(field) for field in fields if field)


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


def vector_search(
    index: faiss.Index,
    metadata: List[Dict],
    query_vector: np.ndarray,
    candidates: int,
) -> List[Dict]:
    """Executa busca vetorial FAISS."""
    search_k = min(candidates, len(metadata))
    scores, positions = index.search(query_vector, search_k)

    results = []

    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        if position < 0:
            continue

        item = metadata[position].copy()
        item["vector_rank"] = rank
        item["vector_score"] = float(score)
        item["position"] = int(position)
        results.append(item)

    return results


def build_bm25_statistics(metadata: List[Dict]) -> Tuple[List[List[str]], Dict[str, int], float]:
    """Prepara corpus tokenizado e document frequency para BM25."""
    corpus_tokens: List[List[str]] = []
    document_frequency: Dict[str, int] = defaultdict(int)

    for item in metadata:
        tokens = tokenize(build_searchable_text(item))
        corpus_tokens.append(tokens)

        for token in set(tokens):
            document_frequency[token] += 1

    avg_doc_length = sum(len(tokens) for tokens in corpus_tokens) / max(len(corpus_tokens), 1)

    return corpus_tokens, dict(document_frequency), avg_doc_length


def bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    document_frequency: Dict[str, int],
    total_docs: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Calcula BM25 para um documento."""
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_length = len(doc_tokens)
    term_frequency = Counter(doc_tokens)
    score = 0.0

    for token in query_tokens:
        tf = term_frequency.get(token, 0)

        if tf == 0:
            continue

        df = document_frequency.get(token, 0)
        idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / max(avg_doc_length, 1)))

        score += idf * (numerator / denominator)

    return score


def lexical_search_bm25(
    metadata: List[Dict],
    query: str,
    candidates: int,
) -> List[Dict]:
    """Executa busca lexical BM25 local."""
    query_tokens = tokenize(query)

    corpus_tokens, document_frequency, avg_doc_length = build_bm25_statistics(metadata)
    total_docs = len(metadata)

    scored_items = []

    for position, item in enumerate(metadata):
        score = bm25_score(
            query_tokens=query_tokens,
            doc_tokens=corpus_tokens[position],
            document_frequency=document_frequency,
            total_docs=total_docs,
            avg_doc_length=avg_doc_length,
        )

        if score <= 0:
            continue

        scored_items.append((score, position, item))

    scored_items.sort(key=lambda x: x[0], reverse=True)

    results = []

    for rank, (score, position, item) in enumerate(scored_items[:candidates], start=1):
        result = item.copy()
        result["bm25_rank"] = rank
        result["bm25_score"] = float(score)
        result["position"] = int(position)
        results.append(result)

    return results


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    top_k: int,
    rrf_k: int,
    min_score: float,
) -> List[Dict]:
    """Combina rankings vetorial e lexical usando Reciprocal Rank Fusion."""
    fused: Dict[str, Dict] = {}

    for item in vector_results:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue

        if chunk_id not in fused:
            fused[chunk_id] = item.copy()
            fused[chunk_id]["rrf_score"] = 0.0

        vector_rank = item.get("vector_rank")
        if vector_rank:
            fused[chunk_id]["rrf_score"] += 1 / (rrf_k + vector_rank)
            fused[chunk_id]["vector_rank"] = vector_rank
            fused[chunk_id]["vector_score"] = item.get("vector_score")

    for item in bm25_results:
        chunk_id = item.get("chunk_id")
        if not chunk_id:
            continue

        if chunk_id not in fused:
            fused[chunk_id] = item.copy()
            fused[chunk_id]["rrf_score"] = 0.0

        bm25_rank = item.get("bm25_rank")
        if bm25_rank:
            fused[chunk_id]["rrf_score"] += 1 / (rrf_k + bm25_rank)
            fused[chunk_id]["bm25_rank"] = bm25_rank
            fused[chunk_id]["bm25_score"] = item.get("bm25_score")

    fused_results = list(fused.values())
    fused_results.sort(key=lambda item: item.get("rrf_score", 0), reverse=True)

    filtered_results = [
        item
        for item in fused_results
        if item.get("rrf_score", 0) >= min_score
    ]

    for rank, item in enumerate(filtered_results[:top_k], start=1):
        item["hybrid_rank"] = rank

    return filtered_results[:top_k]


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
        vector_score = item.get("vector_score")
        bm25_score_value = item.get("bm25_score")
        text = item.get("text", "")

        citation = (
            f"Fonte: {source} | "
            f"Tipo: {document_type} | "
            f"Página(s): {page_start}-{page_end} | "
            f"Chunk: {chunk_number}/{total_chunks} | "
            f"Hybrid RRF: {rrf_score:.4f} | "
            f"Vector score: {vector_score} | "
            f"BM25 score: {bm25_score_value}"
        )

        blocks.append(
            f"[{citation}]\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    """Cria prompt para RAG com hybrid search."""
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

PERGUNTA:
{question}

CONTEXTO:
{context}

RESPOSTA:
""".strip()


def invoke_claude(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoca Claude via Amazon Bedrock."""
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
            "hybrid_rank": item.get("hybrid_rank"),
            "rrf_score": item.get("rrf_score"),
            "vector_rank": item.get("vector_rank"),
            "vector_score": item.get("vector_score"),
            "bm25_rank": item.get("bm25_rank"),
            "bm25_score": item.get("bm25_score"),
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


def print_results(
    title: str,
    results: List[Dict],
    rank_key: str,
    score_key: str,
) -> None:
    """Mostra resultados no terminal."""
    print(f"\n{title}")

    if not results:
        print("Nenhum resultado.")
        return

    for item in results:
        print("\n" + "-" * 80)
        print(f"Rank: {item.get(rank_key)}")
        print(f"Score: {item.get(score_key)}")
        print(f"Documento: {item.get('document_name')}")
        print(f"Tipo: {item.get('document_type')}")
        print(f"Chunk: {item.get('chunk_number')} de {item.get('total_chunks')}")
        print(f"Páginas: {item.get('page_start')} - {item.get('page_end')}")
        print(f"Estratégia: {item.get('chunk_strategy')}")
        print(f"Fonte: {item.get('s3_uri')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa hybrid search com FAISS vetorial, BM25 lexical e RRF."
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
        help="Índice usado no hybrid search. Default: baseline.",
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
        help="Quantidade final de chunks após fusão híbrida.",
    )

    parser.add_argument(
        "--vector-candidates",
        type=int,
        default=DEFAULT_VECTOR_CANDIDATES,
        help="Quantidade de candidatos na busca vetorial.",
    )

    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=DEFAULT_BM25_CANDIDATES,
        help="Quantidade de candidatos na busca lexical BM25.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help="Constante do Reciprocal Rank Fusion.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Score mínimo de RRF para manter resultado.",
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
        "--show-context",
        action="store_true",
        help="Mostra o contexto enviado ao LLM.",
    )

    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Executa apenas hybrid search, sem chamar Claude para resposta final.",
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

    print("Iniciando hybrid search")
    print(f"Bucket: {args.bucket}")
    print(f"Index mode: {args.index_mode}")
    print(f"Index prefix: {index_prefix}")
    print(f"Top-K final: {args.top_k}")
    print(f"Vector candidates: {args.vector_candidates}")
    print(f"BM25 candidates: {args.bm25_candidates}")
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

    query_vector = embed_question(
        bedrock_client=bedrock_client,
        question=args.question,
        model_id=args.embedding_model_id,
    )

    vector_results = vector_search(
        index=index,
        metadata=metadata,
        query_vector=query_vector,
        candidates=args.vector_candidates,
    )

    bm25_results = lexical_search_bm25(
        metadata=metadata,
        query=args.question,
        candidates=args.bm25_candidates,
    )

    hybrid_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        min_score=args.min_score,
    )

    print("\nPergunta:")
    print(args.question)

    print_results(
        title="Top resultados vetoriais FAISS",
        results=vector_results[:args.top_k],
        rank_key="vector_rank",
        score_key="vector_score",
    )

    print_results(
        title="Top resultados lexicais BM25",
        results=bm25_results[:args.top_k],
        rank_key="bm25_rank",
        score_key="bm25_score",
    )

    print_results(
        title="Top resultados híbridos FAISS + BM25 + RRF",
        results=hybrid_results,
        rank_key="hybrid_rank",
        score_key="rrf_score",
    )

    if not hybrid_results:
        print("\nResposta:")
        print("Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança.")
        return

    context = build_context(hybrid_results)

    if args.show_context:
        print("\nContexto enviado ao LLM:")
        print(context)

    if args.no_answer:
        print("\nFontes estruturadas:")
        print(json.dumps(build_sources(hybrid_results), ensure_ascii=False, indent=2))
        return

    prompt = build_prompt(
        question=args.question,
        context=context,
    )

    answer = invoke_claude(
        bedrock_client=bedrock_client,
        model_id=args.llm_model_id,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nResposta:")
    print(answer)

    print("\nFontes estruturadas:")
    print(json.dumps(build_sources(hybrid_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()