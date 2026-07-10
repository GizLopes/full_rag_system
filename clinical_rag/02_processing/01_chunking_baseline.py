# OBJETIVO PRINCIPAL
# Gerar chunks por tokens com RecursiveCharacterTextSplitter, preservando páginas, documentos,
# S3 URI, tipo documental e metadados para retrieval baseline no RAG

import argparse
import hashlib
import json
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import boto3
import tiktoken
from botocore.exceptions import ClientError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_PREFIX = "rag-database/"
DEFAULT_OUTPUT_FILE = "clinical_chunks.jsonl"
DEFAULT_OUTPUT_PREFIX = "chunks/"
DEFAULT_REGION = "us-east-1"

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60

ENCODING = tiktoken.get_encoding("cl100k_base")


DOCUMENT_TYPE_KEYWORDS: Dict[str, str] = {
    "consulta": "consultas_ambulatoriais",
    "consultas": "consultas_ambulatoriais",
    "ambulatorial": "consultas_ambulatoriais",
    "ambulatoriais": "consultas_ambulatoriais",
    "hemograma": "hemograma_e_bioquimica",
    "bioquimica": "hemograma_e_bioquimica",
    "bioquímica": "hemograma_e_bioquimica",
    "ressonancia": "ressonancia_coluna",
    "ressonância": "ressonancia_coluna",
    "coluna": "ressonancia_coluna",
    "cardiologista": "parecer_cardiologista",
    "cardiologia": "parecer_cardiologista",
    "parecer": "parecer_medico",
    "alta": "alta_hospitalar",
    "hospitalar": "alta_hospitalar",
}


def normalize_text(text: str) -> str:
    """Remove excesso de espaços e quebras sem destruir o conteúdo clínico."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Conta tokens usando a mesma codificação utilizada no chunking."""
    return len(ENCODING.encode(text))


def infer_document_type(s3_key: str) -> str:
    """Infere o tipo documental pelo nome/caminho do arquivo."""
    key_lower = s3_key.lower()

    for keyword, document_type in DOCUMENT_TYPE_KEYWORDS.items():
        if keyword in key_lower:
            return document_type

    return "documento_clinico"


def make_document_id(bucket: str, s3_key: str, etag: Optional[str] = None) -> str:
    """Cria um ID estável para o documento com base na origem S3."""
    base = f"s3://{bucket}/{s3_key}:{etag or ''}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def make_content_hash(text: str) -> str:
    """Cria hash do conteúdo do chunk para deduplicação e auditoria."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_pdf_objects(s3_client, bucket: str, prefix: str) -> Iterable[Dict]:
    """Lista todos os PDFs dentro de um prefixo S3 com paginação."""
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            if key.lower().endswith(".pdf"):
                yield obj


def read_pdf_from_s3(s3_client, bucket: str, s3_key: str) -> BytesIO:
    """Baixa um PDF do S3 em memória."""
    response = s3_client.get_object(Bucket=bucket, Key=s3_key)
    return BytesIO(response["Body"].read())


def extract_pdf_pages(pdf_bytes: BytesIO) -> List[Dict]:
    """Extrai texto por página do PDF."""
    reader = PdfReader(pdf_bytes)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = normalize_text(raw_text)

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def build_page_aware_text(pages: List[Dict]) -> str:
    """Inclui marcador de página para preservar rastreabilidade no chunk."""
    blocks = []

    for item in pages:
        blocks.append(f"[PAGE {item['page']}]\n{item['text']}")

    return "\n\n".join(blocks)


def infer_page_range(chunk_text: str) -> Tuple[Optional[int], Optional[int]]:
    """Infere páginas presentes no chunk a partir dos marcadores [PAGE X]."""
    pages = [int(x) for x in re.findall(r"\[PAGE (\d+)\]", chunk_text)]

    if not pages:
        return None, None

    return min(pages), max(pages)


def build_baseline_splitter(
    chunk_size: int,
    chunk_overlap: int,
) -> RecursiveCharacterTextSplitter:
    """Cria o splitter baseline por tokens."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "; ", ", ", " "],
    )


def chunk_page_text(
    splitter: RecursiveCharacterTextSplitter,
    page_text: str,
) -> List[str]:
    """Divide o texto de uma única página em chunks."""
    clean_text = normalize_text(page_text)

    if not clean_text:
        return []

    raw_chunks = splitter.split_text(clean_text)

    return [
        normalize_text(chunk)
        for chunk in raw_chunks
        if normalize_text(chunk)
    ]


def chunk_s3_pdf_baseline(
    s3_client,
    bucket: str,
    obj: Dict,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    """
    Lê um PDF do S3 e gera chunks baseline.

    Estratégia baseline corrigida:
    - extrai texto por página;
    - aplica RecursiveCharacterTextSplitter dentro de cada página;
    - cada chunk recebe page_start e page_end iguais à página de origem;
    - evita chunks com página None;
    - mantém metadados completos para RAG, auditoria e citações.
    """
    s3_key = obj["Key"]
    etag = obj.get("ETag", "").replace('"', "")
    document_id = make_document_id(bucket=bucket, s3_key=s3_key, etag=etag)
    document_name = Path(s3_key).name
    document_type = infer_document_type(s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"

    pdf_bytes = read_pdf_from_s3(
        s3_client=s3_client,
        bucket=bucket,
        s3_key=s3_key,
    )

    pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        return []

    splitter = build_baseline_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    temp_chunks: List[Dict] = []

    for page in pages:
        page_number = page["page"]
        page_text = page["text"]

        page_chunks = chunk_page_text(
            splitter=splitter,
            page_text=page_text,
        )

        for chunk_text in page_chunks:
            temp_chunks.append({
                "text": chunk_text,
                "page_start": page_number,
                "page_end": page_number,
            })

    if not temp_chunks:
        return []

    total_chunks = len(temp_chunks)
    records: List[Dict] = []

    for idx, item in enumerate(temp_chunks, start=1):
        clean_chunk = item["text"]
        page_start = item["page_start"]
        page_end = item["page_end"]
        chunk_id = f"{document_id}::baseline_chunk_{idx:06d}"

        records.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "document_name": document_name,
            "document_type": document_type,
            "chunk_strategy": "baseline_page_recursive_token_chunking",
            "chunk_number": idx,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "page_start": page_start,
            "page_end": page_end,
            "s3_bucket": bucket,
            "s3_key": s3_key,
            "s3_uri": s3_uri,
            "etag": etag,
            "content_hash": make_content_hash(clean_chunk),
            "text": clean_chunk,
            "metadata": {
                "source": s3_uri,
                "document_id": document_id,
                "document_name": document_name,
                "document_type": document_type,
                "chunk_strategy": "baseline_page_recursive_token_chunking",
                "chunk_number": idx,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "page_start": page_start,
                "page_end": page_end,
                "s3_bucket": bucket,
                "s3_key": s3_key,
                "etag": etag,
            },
        })

    return records

def save_jsonl(records: List[Dict], output_file: Path) -> None:
    """Salva registros em JSONL local."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def upload_jsonl_to_s3(
    s3_client,
    bucket: str,
    records: List[Dict],
    key: str,
) -> None:
    """Salva o JSONL diretamente no S3."""
    jsonl_content = "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records
    )

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=jsonl_content.encode("utf-8"),
        ContentType="application/json",
    )


def process_all_pdfs(
    s3_client,
    bucket: str,
    prefix: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Dict], int, int]:
    """Processa todos os PDFs encontrados no S3."""
    try:
        pdf_objects = list(list_pdf_objects(s3_client, bucket, prefix))
    except ClientError as exc:
        raise RuntimeError(f"Erro ao listar objetos no S3: {exc}") from exc

    if not pdf_objects:
        return [], 0, 0

    all_chunks: List[Dict] = []
    processed_count = 0

    for obj in pdf_objects:
        s3_key = obj["Key"]
        print(f"\nProcessando: s3://{bucket}/{s3_key}")

        try:
            chunks = chunk_s3_pdf_baseline(
                s3_client=s3_client,
                bucket=bucket,
                obj=obj,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:
            print(f"  Erro ao processar {s3_key}: {exc}")
            continue

        processed_count += 1
        print(f"  Chunks gerados: {len(chunks)}")
        all_chunks.extend(chunks)

    return all_chunks, len(pdf_objects), processed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera chunks baseline por página para RAG a partir de PDFs clínicos no S3."
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 dos documentos clínicos.",
    )

    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Prefixo S3 onde estão os PDFs.",
    )

    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Arquivo JSONL local de saída.",
    )

    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefixo S3 onde o JSONL será salvo.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Tamanho máximo do chunk em tokens.",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Overlap entre chunks em tokens.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    output_file = Path(args.output_file)

    print("Iniciando chunking baseline por página a partir do S3")
    print(f"Bucket: {args.bucket}")
    print(f"Prefixo: {args.prefix}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Saída local: {output_file}")

    all_chunks, found_count, processed_count = process_all_pdfs(
        s3_client=s3_client,
        bucket=args.bucket,
        prefix=args.prefix,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if not all_chunks:
        print("\nNenhum chunk gerado.")
        print(f"PDFs encontrados: {found_count}")
        print(f"PDFs processados: {processed_count}")
        return

    save_jsonl(all_chunks, output_file)

    s3_output_key = f"{args.output_prefix.rstrip('/')}/{output_file.name}"

    upload_jsonl_to_s3(
        s3_client=s3_client,
        bucket=args.bucket,
        records=all_chunks,
        key=s3_output_key,
    )

    print("\nChunking baseline complete")
    print(f"PDFs encontrados: {found_count}")
    print(f"PDFs processados: {processed_count}")
    print(f"Total de chunks: {len(all_chunks)}")
    print(f"Arquivo local: {output_file}")
    print(f"Arquivo S3: s3://{args.bucket}/{s3_output_key}")


if __name__ == "__main__":
    main()