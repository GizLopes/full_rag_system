# OBJETIVO PRINCIPAL
# Executar RAG baseline com Titan Embeddings, FAISS e Claude, recuperando chunks do índice baseline e gerando resposta com fontes

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_REGION = "us-east-1"

DEFAULT_INDEX_PREFIX = "index/"
DEFAULT_INDEX_FILE = "clinical_faiss.index"
DEFAULT_METADATA_FILE = "clinical_faiss_metadata.pkl"

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.20
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0


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
    ou metadados antigos. Use --use-local-cache apenas se quiser reaproveitar
    os artefatos locais deliberadamente.
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


def load_faiss_artifacts(
    s3_client,
    bucket: str,
    index_prefix: str,
    index_file: str,
    metadata_file: str,
    use_local_cache: bool = False,
) -> Tuple[faiss.Index, List[Dict]]:
    """Carrega índice FAISS baseline e metadados sempre baixando do S3 por padrão."""
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
    query_vector: np.ndarray,
    top_k: int,
) -> List[Dict]:
    """Executa busca vetorial baseline no FAISS."""
    search_k = min(top_k, len(metadata))
    scores, positions = index.search(query_vector, search_k)

    results = []

    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        if position < 0:
            continue

        item = metadata[position].copy()
        item["rank"] = rank
        item["score"] = float(score)
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
        text = item.get("text", "")

        citation = (
            f"Fonte: {source} | "
            f"Tipo: {document_type} | "
            f"Página(s): {page_start}-{page_end} | "
            f"Chunk: {chunk_number}/{total_chunks} | "
            f"Score FAISS: {score:.4f}"
        )

        blocks.append(
            f"[{citation}]\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    """Cria prompt baseline de RAG."""
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
            "rank": item.get("rank"),
            "score": item.get("score"),
            "document_name": item.get("document_name"),
            "document_type": item.get("document_type"),
            "chunk_number": item.get("chunk_number"),
            "total_chunks": item.get("total_chunks"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "s3_uri": item.get("s3_uri"),
            "chunk_strategy": item.get("chunk_strategy"),
        })

    return sources


def print_retrieval_summary(
    question: str,
    results: List[Dict],
) -> None:
    """Mostra resumo dos chunks recuperados."""
    print("\nPergunta:")
    print(question)

    print("\nChunks recuperados no baseline FAISS:")

    for item in results:
        print("\n" + "-" * 80)
        print(f"Rank: {item.get('rank')}")
        print(f"Score FAISS: {item.get('score', 0):.4f}")
        print(f"Documento: {item.get('document_name')}")
        print(f"Tipo: {item.get('document_type')}")
        print(f"Chunk: {item.get('chunk_number')} de {item.get('total_chunks')}")
        print(f"Páginas: {item.get('page_start')} - {item.get('page_end')}")
        print(f"Estratégia: {item.get('chunk_strategy')}")
        print(f"Fonte: {item.get('s3_uri')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa RAG baseline com FAISS, Titan Embeddings e Claude no Bedrock."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Pergunta clínica em linguagem natural.",
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 onde estão os artefatos FAISS baseline.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--index-prefix",
        default=DEFAULT_INDEX_PREFIX,
        help="Prefixo S3 do índice FAISS baseline.",
    )

    parser.add_argument(
        "--index-file",
        default=DEFAULT_INDEX_FILE,
        help="Arquivo local/S3 do índice FAISS baseline.",
    )

    parser.add_argument(
        "--metadata-file",
        default=DEFAULT_METADATA_FILE,
        help="Arquivo local/S3 com metadados do índice baseline.",
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
        help="Quantidade de chunks recuperados no FAISS.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Score mínimo para aceitar o melhor chunk recuperado.",
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
        "--use-local-cache",
        action="store_true",
        help=(
            "Usa arquivos FAISS locais se existirem. "
            "Por padrão, o script sempre baixa os artefatos do S3."
        ),
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    print("Iniciando baseline RAG")
    print(f"Bucket: {args.bucket}")
    print(f"Prefixo do índice: {args.index_prefix}")
    print(f"Embedding model: {args.embedding_model_id}")
    print(f"LLM model: {args.llm_model_id}")
    print(f"Top-K: {args.top_k}")
    print(f"Usar cache local: {args.use_local_cache}")

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=args.bucket,
        index_prefix=args.index_prefix,
        index_file=args.index_file,
        metadata_file=args.metadata_file,
        use_local_cache=args.use_local_cache,
    )

    query_vector = embed_question(
        bedrock_client=bedrock_client,
        question=args.question,
        model_id=args.embedding_model_id,
    )

    results = search_faiss(
        index=index,
        metadata=metadata,
        query_vector=query_vector,
        top_k=args.top_k,
    )

    print_retrieval_summary(
        question=args.question,
        results=results,
    )

    if not results:
        print("\nResposta:")
        print("Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança.")
        return

    best_score = results[0].get("score", 0)

    if best_score < args.min_score:
        print("\nResposta:")
        print("Não encontrei informação suficiente nos documentos clínicos disponíveis para responder com segurança.")
        print(f"Melhor score encontrado: {best_score:.4f}")
        return

    context = build_context(results)

    if args.show_context:
        print("\nContexto enviado ao LLM:")
        print(context)

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
    print(json.dumps(build_sources(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()