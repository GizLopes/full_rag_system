# OBJETIVO PRINCIPAL
# Gerar embeddings Titan para chunks semânticos por paciente/seção, salvando JSONL local e no S3 para retrieval otimizado

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import boto3
from botocore.exceptions import ClientError


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_REGION = "us-east-1"

DEFAULT_CHUNKS_KEY = "chunks_semantic/clinical_chunks_semantic.jsonl"
DEFAULT_OUTPUT_KEY = "embeddings_semantic/clinical_embeddings_semantic.jsonl"
DEFAULT_OUTPUT_FILE = "clinical_embeddings_semantic.jsonl"

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_NORMALIZE = True
DEFAULT_MAX_RETRIES = 5

STRATEGY_NAME = "semantic"
SCRIPT_DESCRIPTION = "Gera embeddings Titan para os chunks semânticos do RAG clínico."

def read_jsonl_from_s3(
    s3_client,
    bucket: str,
    key: str,
) -> List[Dict]:
    """Lê um arquivo JSONL do S3 e retorna uma lista de registros."""
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
            raise ValueError(f"Linha inválida no JSONL: {line_number}") from exc

    return records


def write_jsonl_to_s3(
    s3_client,
    bucket: str,
    key: str,
    records: Iterable[Dict],
) -> None:
    """Escreve registros JSONL no S3."""
    body = "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records
    )

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="application/json",
    )


def save_jsonl(
    records: Iterable[Dict],
    output_file: Path,
) -> None:
    """Salva embeddings localmente em formato JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_titan_embedding(
    bedrock_runtime_client,
    text: str,
    model_id: str,
    dimensions: int,
    normalize: bool,
    max_retries: int,
) -> List[float]:
    """Gera embedding usando Amazon Bedrock Titan Embeddings."""
    payload = {
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = bedrock_runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                accept="application/json",
                contentType="application/json",
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding")

            if not embedding:
                raise RuntimeError("Resposta do Bedrock não retornou embedding.")

            return embedding

        except ClientError as exc:
            last_error = exc
            error_code = exc.response.get("Error", {}).get("Code", "")

            retryable = error_code in {
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "InternalServerException",
            }

            if not retryable or attempt == max_retries:
                raise

            sleep_seconds = min(2 ** attempt, 30)
            print(
                f"  Retry Bedrock {attempt}/{max_retries} após erro "
                f"{error_code}. Aguardando {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

        except Exception as exc:
            last_error = exc

            if attempt == max_retries:
                raise

            sleep_seconds = min(2 ** attempt, 30)
            print(
                f"  Retry Bedrock {attempt}/{max_retries}. "
                f"Aguardando {sleep_seconds}s."
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Falha ao gerar embedding: {last_error}")


def validate_chunk_record(record: Dict, index: int) -> Optional[str]:
    """Valida se o registro tem texto."""
    text = record.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return f"chunk_{index}: texto vazio"

    return None


def enrich_chunks_with_embeddings(
    records: List[Dict],
    bedrock_runtime_client,
    model_id: str,
    dimensions: int,
    normalize: bool,
    max_retries: int,
    strategy_name: str,
) -> List[Dict]:
    """Adiciona embedding a cada chunk."""
    enriched_records: List[Dict] = []
    total = len(records)

    for index, record in enumerate(records, start=1):
        validation_error = validate_chunk_record(record, index)

        if validation_error:
            print(f"Ignorando registro inválido: {validation_error}")
            continue

        text = record["text"].strip()
        chunk_id = record.get("chunk_id", f"chunk_{index}")

        print(f"Gerando embedding {index}/{total}: {chunk_id}")

        embedding = generate_titan_embedding(
            bedrock_runtime_client=bedrock_runtime_client,
            text=text,
            model_id=model_id,
            dimensions=dimensions,
            normalize=normalize,
            max_retries=max_retries,
        )

        enriched_record = dict(record)
        enriched_record["embedding"] = embedding
        enriched_record["embedding_model_id"] = model_id
        enriched_record["embedding_dimensions"] = dimensions
        enriched_record["embedding_normalized"] = normalize
        enriched_record["embedding_strategy"] = strategy_name

        metadata = dict(enriched_record.get("metadata") or {})
        metadata["embedding_model_id"] = model_id
        metadata["embedding_dimensions"] = dimensions
        metadata["embedding_normalized"] = normalize
        metadata["embedding_strategy"] = strategy_name
        enriched_record["metadata"] = metadata

        enriched_records.append(enriched_record)

    return enriched_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=SCRIPT_DESCRIPTION
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 onde estão os chunks.",
    )

    parser.add_argument(
        "--chunks-key",
        default=DEFAULT_CHUNKS_KEY,
        help="Key S3 do JSONL de chunks.",
    )

    parser.add_argument(
        "--output-key",
        default=DEFAULT_OUTPUT_KEY,
        help="Key S3 do JSONL com embeddings.",
    )

    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Arquivo JSONL local de saída.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Modelo de embedding no Bedrock.",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        help="Dimensão do vetor.",
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=DEFAULT_NORMALIZE,
        help="Normaliza os vetores gerados pelo Titan.",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize",
        help="Desabilita normalização no Titan.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Número máximo de tentativas no Bedrock.",
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    bedrock_runtime_client = boto3.client(
        "bedrock-runtime",
        region_name=args.region,
    )

    print("Iniciando geração de embeddings")
    print(f"Estratégia: {STRATEGY_NAME}")
    print(f"Bucket: s3://{args.bucket}")
    print(f"Chunks: s3://{args.bucket}/{args.chunks_key}")
    print(f"Saída S3: s3://{args.bucket}/{args.output_key}")
    print(f"Saída local: {args.output_file}")
    print(f"Modelo: {args.model_id}")
    print(f"Dimensões: {args.dimensions}")
    print(f"Normalizar no Titan: {args.normalize}")

    try:
        chunks = read_jsonl_from_s3(
            s3_client=s3_client,
            bucket=args.bucket,
            key=args.chunks_key,
        )
    except ClientError as exc:
        raise RuntimeError(
            f"Erro ao ler chunks no S3: s3://{args.bucket}/{args.chunks_key}. "
            f"Confirme se o chunking correspondente foi executado."
        ) from exc

    if not chunks:
        print("Nenhum chunk encontrado.")
        return

    enriched_records = enrich_chunks_with_embeddings(
        records=chunks,
        bedrock_runtime_client=bedrock_runtime_client,
        model_id=args.model_id,
        dimensions=args.dimensions,
        normalize=args.normalize,
        max_retries=args.max_retries,
        strategy_name=STRATEGY_NAME,
    )

    write_jsonl_to_s3(
        s3_client=s3_client,
        bucket=args.bucket,
        key=args.output_key,
        records=enriched_records,
    )

    save_jsonl(
        records=enriched_records,
        output_file=Path(args.output_file),
    )

    print("\nEmbeddings complete")
    print(f"Estratégia: {STRATEGY_NAME}")
    print(f"Chunks lidos: {len(chunks)}")
    print(f"Embeddings gerados: {len(enriched_records)}")
    print(f"Arquivo S3: s3://{args.bucket}/{args.output_key}")
    print(f"Arquivo local: {args.output_file}")


if __name__ == "__main__":
    main()