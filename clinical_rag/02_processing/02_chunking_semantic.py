# OBJETIVO PRINCIPAL
# Gerar chunks por paciente e seção clínica, evitando misturar pacientes no mesmo contexto e
# melhorando precisão, rastreabilidade e qualidade do retrieval

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
DEFAULT_OUTPUT_FILE = "clinical_chunks_semantic.jsonl"
DEFAULT_OUTPUT_PREFIX = "chunks_semantic/"
DEFAULT_REGION = "us-east-1"

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80

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


SECTION_KEYWORDS: Dict[str, str] = {
    "queixa_principal": "queixa_principal",
    "queixa": "queixa_principal",
    "historia": "historia_clinica",
    "história": "historia_clinica",
    "hipotese": "hipotese_diagnostica",
    "hipótese": "hipotese_diagnostica",
    "diagnostico": "diagnostico",
    "diagnóstico": "diagnostico",
    "medicacoes_atuais": "medicacoes",
    "medicações_atuais": "medicacoes",
    "medicamentos": "medicacoes",
    "medicação": "medicacoes",
    "medicacao": "medicacoes",
    "prescricao": "prescricao",
    "prescrição": "prescricao",
    "conduta": "conduta",
    "exames": "exames",
    "resultado": "resultado_exame",
    "resultados": "resultado_exame",
    "creatinina": "resultado_exame",
    "hemoglobina": "resultado_exame",
    "plaquetas": "resultado_exame",
    "glicemia": "resultado_exame",
    "impressao": "impressao_diagnostica",
    "impressão": "impressao_diagnostica",
    "laudo": "laudo",
    "alta": "alta_hospitalar",
    "internacao": "internacao",
    "internação": "internacao",
}


NAME_STOP_WORDS = [
    "data_",
    "data ",
    "idade",
    "sexo",
    "queixa",
    "historia",
    "história",
    "diagnostico",
    "diagnóstico",
    "medicacoes",
    "medicações",
    "medicamentos",
    "prescricao",
    "prescrição",
    "conduta",
    "exames",
    "resultado",
    "creatinina",
    "hemoglobina",
    "plaquetas",
    "glicemia",
    "impressao",
    "impressão",
    "laudo",
    "alta",
    "internacao",
    "internação",
    "patient_id",
]


def normalize_text(text: str) -> str:
    """Remove excesso de espaços e quebras sem destruir o conteúdo clínico."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(text: str) -> str:
    """Normalização simples para matching textual."""
    text = text.lower()
    replacements = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ç": "c",
    }

    for source, target in replacements.items():
        text = text.replace(source, target)

    return text


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


def infer_clinical_section(text: str, document_type: str) -> str:
    """Infere a seção clínica mais provável para o bloco."""
    text_norm = normalize_for_match(text)

    for keyword, section in SECTION_KEYWORDS.items():
        if normalize_for_match(keyword) in text_norm:
            return section

    return document_type


def extract_patient_id(text: str) -> Optional[str]:
    """Extrai patient_id no formato P000."""
    match = re.search(r"\bP\d{3}\b", text, flags=re.IGNORECASE)

    if match:
        return match.group(0).upper()

    return None


def extract_patient_name(text: str) -> Optional[str]:
    """Extrai nome do paciente a partir do campo nome."""
    match = re.search(
        r"\bnome\s*[:\-]?\s+([A-Za-zÀ-ÿ ]{3,120})",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    raw_name = re.sub(r"\s+", " ", match.group(1)).strip(" :-")
    raw_name_norm = normalize_for_match(raw_name)

    cut_position = len(raw_name)

    for stop_word in NAME_STOP_WORDS:
        stop_norm = normalize_for_match(stop_word)
        position = raw_name_norm.find(stop_norm)

        if position >= 0:
            cut_position = min(cut_position, position)

    name = raw_name[:cut_position].strip(" :-")

    if not name:
        return None

    words = name.split()

    if len(words) > 5:
        name = " ".join(words[:5])

    return name


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


def split_page_into_patient_blocks(page: Dict) -> List[Dict]:
    """
    Divide uma página em blocos por paciente.

    Se a página tiver múltiplos patient_id, cada paciente vira um bloco separado.
    Se não houver patient_id, a página inteira vira um bloco.
    """
    text = page["text"]
    matches = list(re.finditer(r"\bP\d{3}\b", text, flags=re.IGNORECASE))

    if len(matches) <= 1:
        return [{
            "page_start": page["page"],
            "page_end": page["page"],
            "text": text,
        }]

    blocks = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block_text = normalize_text(text[start:end])

        if block_text:
            blocks.append({
                "page_start": page["page"],
                "page_end": page["page"],
                "text": block_text,
            })

    return blocks


def split_large_semantic_block(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """Divide blocos semânticos grandes sem perder limites principais."""
    if count_tokens(text) <= chunk_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
        separators=[
            "\n\n",
            "\n",
            "conduta",
            "prescricao",
            "prescrição",
            "medicacoes_atuais",
            "medicações_atuais",
            "resultado",
            ". ",
            "; ",
            ", ",
            " ",
        ],
    )

    return [
        normalize_text(chunk)
        for chunk in splitter.split_text(text)
        if normalize_text(chunk)
    ]


def build_semantic_header(
    patient_id: Optional[str],
    patient_name: Optional[str],
    document_type: str,
    clinical_section: str,
) -> str:
    """Cria cabeçalho semântico para manter entidade e seção dentro do chunk."""
    header_parts = []

    if patient_id:
        header_parts.append(f"patient_id {patient_id}")

    if patient_name:
        header_parts.append(f"nome {patient_name}")

    header_parts.append(f"document_type {document_type}")
    header_parts.append(f"clinical_section {clinical_section}")

    return "\n".join(header_parts)


def chunk_s3_pdf_semantic(
    s3_client,
    bucket: str,
    obj: Dict,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict]:
    """
    Lê um PDF do S3 e gera chunks semânticos.

    Estratégia semântica:
    - separa blocos por paciente quando há patient_id;
    - preserva documento, página, paciente e seção clínica;
    - evita misturar múltiplos pacientes no mesmo chunk;
    - usa splitter por tokens apenas quando o bloco semântico é grande demais.
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

    semantic_blocks: List[Dict] = []

    for page in pages:
        semantic_blocks.extend(split_page_into_patient_blocks(page))

    records: List[Dict] = []
    chunk_counter = 0

    for block in semantic_blocks:
        block_text = normalize_text(block["text"])

        if not block_text:
            continue

        patient_id = extract_patient_id(block_text)
        patient_name = extract_patient_name(block_text)
        clinical_section = infer_clinical_section(
            text=block_text,
            document_type=document_type,
        )

        header = build_semantic_header(
            patient_id=patient_id,
            patient_name=patient_name,
            document_type=document_type,
            clinical_section=clinical_section,
        )

        enriched_block = normalize_text(f"{header}\n\n{block_text}")
        split_chunks = split_large_semantic_block(
            text=enriched_block,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for split_idx, chunk_text in enumerate(split_chunks, start=1):
            clean_chunk = normalize_text(chunk_text)

            if not clean_chunk:
                continue

            chunk_counter += 1
            chunk_id = f"{document_id}::semantic_chunk_{chunk_counter:06d}"

            records.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": document_name,
                "document_type": document_type,
                "chunk_strategy": "semantic_patient_section_chunking",
                "semantic_block_number": len(records) + 1,
                "semantic_split_number": split_idx,
                "chunk_number": chunk_counter,
                "total_chunks": None,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "patient_id": patient_id,
                "patient_name": patient_name,
                "clinical_section": clinical_section,
                "page_start": block["page_start"],
                "page_end": block["page_end"],
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
                    "chunk_strategy": "semantic_patient_section_chunking",
                    "semantic_split_number": split_idx,
                    "chunk_number": chunk_counter,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "clinical_section": clinical_section,
                    "page_start": block["page_start"],
                    "page_end": block["page_end"],
                    "s3_bucket": bucket,
                    "s3_key": s3_key,
                    "etag": etag,
                },
            })

    total_chunks = len(records)

    for record in records:
        record["total_chunks"] = total_chunks
        record["metadata"]["total_chunks"] = total_chunks

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
            chunks = chunk_s3_pdf_semantic(
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
        print(f"  Chunks semânticos gerados: {len(chunks)}")
        all_chunks.extend(chunks)

    return all_chunks, len(pdf_objects), processed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera chunks semânticos para RAG a partir de PDFs clínicos no S3."
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
        help="Overlap entre chunks em tokens quando o bloco semântico for grande.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    args = parser.parse_args()

    s3_client = boto3.client("s3", region_name=args.region)
    output_file = Path(args.output_file)

    print("Iniciando chunking semântico a partir do S3")
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
        print("\nNenhum chunk semântico gerado.")
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

    print("\nChunking semântico complete")
    print(f"PDFs encontrados: {found_count}")
    print(f"PDFs processados: {processed_count}")
    print(f"Total de chunks semânticos: {len(all_chunks)}")
    print(f"Arquivo local: {output_file}")
    print(f"Arquivo S3: s3://{args.bucket}/{s3_output_key}")


if __name__ == "__main__":
    main()