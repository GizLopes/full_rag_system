# OBJETIVO PRINCIPAL
# Cria índices FAISS para embeddings baseline e semânticos, salva artefatos locais e no S3,
# permitindo comparar retrieval por chunking tradicional versus otimizado

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_REGION = "us-east-1"

BASELINE_EMBEDDINGS_KEY = "embeddings/clinical_embeddings.jsonl"
BASELINE_INDEX_FILE = "clinical_faiss.index"
BASELINE_METADATA_FILE = "clinical_faiss_metadata.pkl"
BASELINE_JSONL_FILE = "clinical_faiss_documents.jsonl"
BASELINE_S3_INDEX_PREFIX = "index/"

SEMANTIC_EMBEDDINGS_KEY = "embeddings_semantic/clinical_embeddings_semantic.jsonl"
SEMANTIC_INDEX_FILE = "clinical_faiss_semantic.index"
SEMANTIC_METADATA_FILE = "clinical_faiss_semantic_metadata.pkl"
SEMANTIC_JSONL_FILE = "clinical_faiss_semantic_documents.jsonl"
SEMANTIC_S3_INDEX_PREFIX = "index_semantic/"


INDEX_CONFIGS = [
    {
        "name": "baseline",
        "embeddings_key": BASELINE_EMBEDDINGS_KEY,
        "index_file": BASELINE_INDEX_FILE,
        "metadata_file": BASELINE_METADATA_FILE,
        "jsonl_file": BASELINE_JSONL_FILE,
        "s3_index_prefix": BASELINE_S3_INDEX_PREFIX,
    },
    {
        "name": "semantic",
        "embeddings_key": SEMANTIC_EMBEDDINGS_KEY,
        "index_file": SEMANTIC_INDEX_FILE,
        "metadata_file": SEMANTIC_METADATA_FILE,
        "jsonl_file": SEMANTIC_JSONL_FILE,
        "s3_index_prefix": SEMANTIC_S3_INDEX_PREFIX,
    },
]


def read_jsonl_from_s3(
    s3_client,
    bucket: str,
    key: str,
) -> List[Dict]:
    """Lê um arquivo JSONL do S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")

    records: List[Dict] = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Linha inválida no JSONL {key}: {line_number}") from exc

    return records


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normaliza vetores para usar inner product como similaridade por cosseno."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vectors / norms


def get_safe_metadata(record: Dict) -> Dict:
    """
    Mantém metadados úteis para comparação baseline vs semantic.

    O índice baseline não terá necessariamente patient_id, patient_name ou
    clinical_section. O índice semantic deve carregar esses campos quando
    disponíveis no arquivo de chunks semânticos.
    """
    metadata = dict(record.get("metadata") or {})

    useful_fields = [
        "chunk_id",
        "document_id",
        "document_name",
        "document_type",
        "chunk_strategy",
        "embedding_strategy",
        "chunk_number",
        "total_chunks",
        "semantic_block_number",
        "semantic_split_number",
        "chunk_size",
        "chunk_overlap",
        "patient_id",
        "patient_name",
        "clinical_section",
        "page_start",
        "page_end",
        "s3_bucket",
        "s3_key",
        "s3_uri",
        "etag",
        "content_hash",
        "embedding_model_id",
        "embedding_dimensions",
        "embedding_normalized",
    ]

    for field in useful_fields:
        if field in record and field not in metadata:
            metadata[field] = record.get(field)

    return metadata


def prepare_vectors_and_metadata(
    records: List[Dict],
    index_name: str,
) -> Tuple[np.ndarray, List[Dict]]:
    """Prepara matriz de vetores e metadados alinhados por posição."""
    vectors = []
    metadata = []

    skipped = 0

    for position, record in enumerate(records):
        embedding = record.get("embedding")
        chunk_id = record.get("chunk_id")
        text = record.get("text")

        if not embedding or not chunk_id or not text:
            skipped += 1
            continue

        vectors.append(embedding)

        item_metadata = get_safe_metadata(record)

        metadata.append({
            "index_name": index_name,
            "position": len(metadata),
            "chunk_id": chunk_id,
            "document_id": record.get("document_id"),
            "document_name": record.get("document_name"),
            "document_type": record.get("document_type"),
            "chunk_strategy": record.get("chunk_strategy"),
            "embedding_strategy": record.get("embedding_strategy"),
            "chunk_number": record.get("chunk_number"),
            "total_chunks": record.get("total_chunks"),
            "semantic_block_number": record.get("semantic_block_number"),
            "semantic_split_number": record.get("semantic_split_number"),
            "patient_id": record.get("patient_id"),
            "patient_name": record.get("patient_name"),
            "clinical_section": record.get("clinical_section"),
            "page_start": record.get("page_start"),
            "page_end": record.get("page_end"),
            "s3_bucket": record.get("s3_bucket"),
            "s3_key": record.get("s3_key"),
            "s3_uri": record.get("s3_uri"),
            "content_hash": record.get("content_hash"),
            "text": text,
            "metadata": item_metadata,
        })

    if skipped:
        print(f"Registros ignorados por ausência de embedding/text/chunk_id: {skipped}")

    if not vectors:
        raise ValueError(f"Nenhum embedding válido encontrado para o índice {index_name}.")

    vector_array = np.array(vectors, dtype="float32")

    if vector_array.ndim != 2:
        raise ValueError(f"Matriz de vetores inválida para {index_name}: shape={vector_array.shape}")

    vector_array = normalize_vectors(vector_array)

    return vector_array, metadata


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Cria índice FAISS local usando similaridade por cosseno."""
    dimensions = vectors.shape[1]

    index = faiss.IndexFlatIP(dimensions)
    index.add(vectors)

    return index


def save_jsonl(
    records: Iterable[Dict],
    output_file: Path,
) -> None:
    """Salva registros em JSONL na pasta atual."""
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_faiss_artifacts(
    index: faiss.Index,
    metadata: List[Dict],
    index_file: str,
    metadata_file: str,
    jsonl_file: str,
) -> List[str]:
    """Salva artefatos FAISS diretamente na pasta atual, sem criar subpasta."""
    index_path = Path(index_file)
    metadata_path = Path(metadata_file)
    jsonl_path = Path(jsonl_file)

    faiss.write_index(index, str(index_path))

    with metadata_path.open("wb") as f:
        pickle.dump(metadata, f)

    save_jsonl(metadata, jsonl_path)

    return [
        str(index_path),
        str(metadata_path),
        str(jsonl_path),
    ]


def upload_files_to_s3(
    s3_client,
    bucket: str,
    prefix: str,
    file_paths: List[str],
) -> None:
    """Sobe os artefatos locais para o S3."""
    clean_prefix = prefix.strip("/")

    for file_path in file_paths:
        local_path = Path(file_path)
        s3_key = f"{clean_prefix}/{local_path.name}"

        s3_client.upload_file(
            str(local_path),
            bucket,
            s3_key,
        )

        print(f"Arquivo enviado para S3: s3://{bucket}/{s3_key}")


def process_index_config(
    s3_client,
    bucket: str,
    config: Dict,
) -> Dict:
    """Cria um índice FAISS para uma configuração de embeddings."""
    index_name = config["name"]
    embeddings_key = config["embeddings_key"]
    index_file = config["index_file"]
    metadata_file = config["metadata_file"]
    jsonl_file = config["jsonl_file"]
    s3_index_prefix = config["s3_index_prefix"]

    print("\n" + "=" * 80)
    print(f"Iniciando indexação FAISS: {index_name}")
    print(f"Embeddings: s3://{bucket}/{embeddings_key}")
    print(f"Arquivos locais:")
    print(f"  FAISS: {index_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"  JSONL: {jsonl_file}")
    print(f"Saída S3: s3://{bucket}/{s3_index_prefix.strip('/')}/")
    print("=" * 80)

    try:
        records = read_jsonl_from_s3(
            s3_client=s3_client,
            bucket=bucket,
            key=embeddings_key,
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")

        if error_code in {"NoSuchKey", "404", "NotFound"}:
            raise RuntimeError(
                f"Arquivo de embeddings não encontrado: s3://{bucket}/{embeddings_key}. "
                f"Rode o gerador de embeddings correspondente antes."
            ) from exc

        raise RuntimeError(f"Erro ao ler embeddings no S3: {exc}") from exc

    print(f"Registros lidos: {len(records)}")

    vectors, metadata = prepare_vectors_and_metadata(
        records=records,
        index_name=index_name,
    )

    print(f"Vetores válidos: {vectors.shape[0]}")
    print(f"Dimensões: {vectors.shape[1]}")

    index = build_faiss_index(vectors)

    local_files = save_faiss_artifacts(
        index=index,
        metadata=metadata,
        index_file=index_file,
        metadata_file=metadata_file,
        jsonl_file=jsonl_file,
    )

    print("\nFAISS indexing complete")
    print(f"Índice: {index_name}")
    print(f"Arquivo FAISS: {index_file}")
    print(f"Metadados PKL: {metadata_file}")
    print(f"Documentos JSONL: {jsonl_file}")

    upload_files_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        prefix=s3_index_prefix,
        file_paths=local_files,
    )

    print(f"Upload para S3 concluído: s3://{bucket}/{s3_index_prefix.strip('/')}/")

    return {
        "index_name": index_name,
        "records_read": len(records),
        "vectors_indexed": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "local_files": local_files,
        "s3_prefix": f"s3://{bucket}/{s3_index_prefix.strip('/')}/",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cria índices FAISS locais para embeddings baseline e semantic. "
            "Por padrão, gera os dois índices."
        )
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 onde estão os JSONL de embeddings.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--only",
        choices=["both", "baseline", "semantic"],
        default="both",
        help="Escolhe qual índice gerar. Default: both.",
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)

    print("Iniciando indexação local com FAISS")
    print(f"Bucket: s3://{args.bucket}")
    print(f"Modo: {args.only}")

    if args.only == "both":
        configs = INDEX_CONFIGS
    else:
        configs = [
            config for config in INDEX_CONFIGS
            if config["name"] == args.only
        ]

    summaries = []

    for config in configs:
        summary = process_index_config(
            s3_client=s3_client,
            bucket=args.bucket,
            config=config,
        )
        summaries.append(summary)

    print("\nResumo final")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()