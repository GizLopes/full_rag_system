# OBJETIVO PRINCIPAL
# Avaliar respostas finais do RAG clínico contra ground truth, verificando termos esperados, fontes, paciente e evidência

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

DEFAULT_EVAL_FILE = "ground_truth_evaluation_dataset.jsonl"
DEFAULT_OUTPUT_JSONL = "answer_eval_results.jsonl"

DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.0


def download_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    local_file: Path,
    use_local_cache: bool = False,
) -> None:
    """
    Baixa artefato do S3.

    Por padrão, sempre substitui o arquivo local para evitar índice ou metadados antigos.
    """
    if use_local_cache and local_file.exists():
        print(f"Usando cache local: {local_file}")
        return

    if local_file.exists():
        print(f"Substituindo arquivo local antigo: {local_file}")

    print(f"Baixando do S3: s3://{bucket}/{s3_key}")

    try:
        s3_client.download_file(bucket, s3_key, str(local_file))
    except ClientError as exc:
        raise RuntimeError(f"Erro ao baixar {s3_key} do S3: {exc}") from exc


def resolve_index_config(index_mode: str) -> Tuple[str, str, str]:
    """Resolve prefixo e arquivos do índice."""
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
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Carrega índice FAISS e metadados."""
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

    if not isinstance(metadata, list):
        raise ValueError("Metadados FAISS deveriam ser uma lista.")

    return index, metadata


def normalize_text(value: Any) -> str:
    """Normaliza texto para comparação robusta."""
    text = str(value or "")
    text = text.replace("\x00", " ")
    text = text.replace("_", " ")
    text = text.replace(",", ".")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza vetor para inner product como cosine similarity."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1

    return vector / norm


def embed_question(
    bedrock_client,
    question: str,
    model_id: str,
) -> np.ndarray:
    """Gera embedding Titan para a pergunta."""
    payload = {"inputText": question}

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


def get_nested_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna metadados internos quando existirem."""
    metadata = record.get("metadata")

    if isinstance(metadata, dict):
        return metadata

    return {}


def get_first_value(record: Dict[str, Any], keys: List[str]) -> Any:
    """Busca valor em diferentes nomes possíveis de campos."""
    nested = get_nested_dict(record)

    for key in keys:
        if key in record and record[key] not in [None, ""]:
            return record[key]

        if key in nested and nested[key] not in [None, ""]:
            return nested[key]

    return None


def get_document_name(record: Dict[str, Any]) -> Any:
    """Obtém nome do documento."""
    return get_first_value(
        record,
        [
            "document_name",
            "source_file",
            "file_name",
            "filename",
            "pdf_name",
            "source_document",
        ],
    )


def get_document_type(record: Dict[str, Any]) -> Any:
    """Obtém tipo documental."""
    return get_first_value(
        record,
        [
            "document_type",
            "doc_type",
            "source_type",
            "clinical_document_type",
        ],
    )


def get_patient_id(record: Dict[str, Any]) -> Any:
    """Obtém patient_id."""
    return get_first_value(
        record,
        [
            "patient_id",
            "paciente_id",
            "id_paciente",
        ],
    )


def get_patient_name(record: Dict[str, Any]) -> Any:
    """Obtém nome do paciente."""
    return get_first_value(
        record,
        [
            "patient_name",
            "paciente",
            "nome_paciente",
        ],
    )


def get_page_start(record: Dict[str, Any]) -> Any:
    """Obtém página inicial."""
    return get_first_value(
        record,
        [
            "page_start",
            "page",
            "page_number",
            "pagina",
        ],
    )


def get_page_end(record: Dict[str, Any]) -> Any:
    """Obtém página final."""
    return get_first_value(
        record,
        [
            "page_end",
            "page",
            "page_number",
            "pagina",
        ],
    )


def get_text(record: Dict[str, Any]) -> str:
    """Obtém texto do chunk."""
    value = get_first_value(
        record,
        [
            "text",
            "chunk_text",
            "content",
            "page_content",
        ],
    )

    return str(value or "")


def compact_source(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduz fonte recuperada para relatório."""
    return {
        "rank": result.get("rank"),
        "score": result.get("score"),
        "chunk_id": result.get("chunk_id"),
        "document_name": get_document_name(result),
        "document_type": get_document_type(result),
        "patient_id": get_patient_id(result),
        "patient_name": get_patient_name(result),
        "clinical_section": result.get("clinical_section"),
        "chunk_number": result.get("chunk_number"),
        "total_chunks": result.get("total_chunks"),
        "page_start": get_page_start(result),
        "page_end": get_page_end(result),
        "s3_uri": result.get("s3_uri"),
        "chunk_strategy": result.get("chunk_strategy"),
        "embedding_strategy": result.get("embedding_strategy"),
        "text_preview": get_text(result)[:700],
    }


def build_context(retrieved_results: List[Dict[str, Any]]) -> str:
    """Monta contexto textual com fontes numeradas."""
    blocks = []

    for result in retrieved_results:
        source = compact_source(result)

        header = (
            f"[Fonte {source['rank']}]\n"
            f"document_name: {source['document_name']}\n"
            f"document_type: {source['document_type']}\n"
            f"patient_id: {source['patient_id']}\n"
            f"patient_name: {source['patient_name']}\n"
            f"page_start: {source['page_start']}\n"
            f"page_end: {source['page_end']}\n"
            f"chunk_number: {source['chunk_number']}/{source['total_chunks']}\n"
            f"chunk_id: {source['chunk_id']}\n"
        )

        blocks.append(
            header
            + "text:\n"
            + get_text(result)
        )

    return "\n\n---\n\n".join(blocks)


def search_faiss(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    question: str,
    bedrock_client,
    embedding_model_id: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Executa busca vetorial."""
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
        item["faiss_position"] = int(position)
        results.append(item)

    return results


def invoke_claude_answer(
    bedrock_client,
    model_id: str,
    question: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Gera resposta final com Claude usando somente o contexto recuperado."""
    prompt = f"""
Você é um assistente clínico para um case de treinamento de RAG.

Responda à pergunta usando exclusivamente o contexto recuperado.

Pergunta:
{question}

Contexto recuperado:
{context}

Regras obrigatórias:
1. Não invente dados.
2. Se a informação não estiver no contexto, responda que não encontrou evidência suficiente.
3. Cite documento, página e chunk usados.
4. Seja direto.
5. Não use conhecimento externo.

Resposta:
""".strip()

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


def parse_expected_terms(value: Any) -> List[str]:
    """Converte expected_terms para lista."""
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass

    return [
        item.strip()
        for item in re.split(r"[|;,]", text)
        if item.strip()
    ]


def empty_to_none(value: Any) -> Optional[str]:
    """Converte string vazia em None."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def normalize_ground_truth_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Padroniza um registro do ground truth."""
    question = str(record.get("question", "")).strip()

    if not question:
        raise ValueError("Registro de ground truth sem question.")

    return {
        "question": question,
        "expected_chunk_id": empty_to_none(record.get("expected_chunk_id")),
        "expected_document_type": empty_to_none(record.get("expected_document_type")),
        "expected_document_name": empty_to_none(record.get("expected_document_name")),
        "expected_patient_id": empty_to_none(record.get("expected_patient_id")),
        "expected_patient_name": empty_to_none(record.get("expected_patient_name")),
        "expected_terms": parse_expected_terms(record.get("expected_terms")),
        "evaluation_scope": record.get("evaluation_scope", "retrieval"),
        "dataset_type": record.get("dataset_type", "ground_truth_seed"),
    }


def load_ground_truth_dataset(eval_file: str) -> List[Dict[str, Any]]:
    """Carrega ground truth JSONL."""
    path = Path(eval_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {eval_file}. "
            "Execute primeiro: python 00_ground_truth_evaluation_dataset.py"
        )

    if path.suffix.lower() != ".jsonl":
        raise ValueError("Este avaliador foi simplificado para usar apenas JSONL.")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(normalize_ground_truth_record(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido na linha {line_number}.") from exc

    return records


def text_contains_expected(value: str, expected: Optional[str]) -> bool:
    """Verifica se um valor esperado aparece no texto."""
    if not expected:
        return True

    return normalize_text(expected) in normalize_text(value)


def expected_term_found(term: str, answer: str) -> bool:
    """Verifica termo esperado com tolerância simples para separadores e números."""
    term_norm = normalize_text(term)
    answer_norm = normalize_text(answer)

    if term_norm in answer_norm:
        return True

    # Variante útil para termos técnicos em snake_case.
    term_words = [
        part
        for part in term_norm.split()
        if part not in {"mg", "dl", "mmhg"}
    ]

    if term_words and all(part in answer_norm for part in term_words):
        return True

    return False


def source_matches_expected(source: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verifica se fonte recuperada bate com documento, paciente e chunk esperado."""
    reasons = []

    checks = [
        (
            "expected_chunk_id",
            expected.get("expected_chunk_id"),
            source.get("chunk_id"),
        ),
        (
            "expected_document_type",
            expected.get("expected_document_type"),
            source.get("document_type"),
        ),
        (
            "expected_document_name",
            expected.get("expected_document_name"),
            source.get("document_name"),
        ),
        (
            "expected_patient_id",
            expected.get("expected_patient_id"),
            source.get("patient_id"),
        ),
        (
            "expected_patient_name",
            expected.get("expected_patient_name"),
            source.get("patient_name"),
        ),
    ]

    searchable_source = json.dumps(source, ensure_ascii=False)

    for field_name, expected_value, actual_value in checks:
        if not expected_value:
            continue

        expected_norm = normalize_text(expected_value)
        actual_norm = normalize_text(actual_value)
        searchable_norm = normalize_text(searchable_source)

        if expected_norm == actual_norm:
            continue

        if expected_norm in searchable_norm:
            continue

        reasons.append(f"source_mismatch:{field_name}")

    return len(reasons) == 0, reasons


def evaluate_answer(
    expected: Dict[str, Any],
    answer: str,
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Avalia resposta final sem LLM Judge."""
    answer_norm = normalize_text(answer)

    expected_terms = expected.get("expected_terms", [])
    missing_terms = [
        term
        for term in expected_terms
        if not expected_term_found(term, answer)
    ]

    answer_contains_expected_terms = len(missing_terms) == 0

    answer_mentions_patient = text_contains_expected(
        answer,
        expected.get("expected_patient_name"),
    ) or text_contains_expected(
        answer,
        expected.get("expected_patient_id"),
    )

    answer_mentions_document = text_contains_expected(
        answer,
        expected.get("expected_document_name"),
    ) or text_contains_expected(
        answer,
        expected.get("expected_document_type"),
    )

    answer_has_citation_signal = any(
        token in answer_norm
        for token in [
            "fonte",
            "documento",
            "pagina",
            "página",
            "chunk",
            "source",
        ]
    )

    source_checks = [
        source_matches_expected(source, expected)
        for source in sources
    ]

    source_match = any(passed for passed, _ in source_checks)

    source_mismatch_reasons = []

    for passed, reasons in source_checks:
        if not passed:
            source_mismatch_reasons.extend(reasons)

    passed = all(
        [
            answer_contains_expected_terms,
            answer_mentions_patient,
            answer_mentions_document,
            answer_has_citation_signal,
            source_match,
        ]
    )

    score_components = {
        "answer_contains_expected_terms": answer_contains_expected_terms,
        "answer_mentions_patient": answer_mentions_patient,
        "answer_mentions_document": answer_mentions_document,
        "answer_has_citation_signal": answer_has_citation_signal,
        "source_match": source_match,
    }

    score = round(
        sum(1 for value in score_components.values() if value)
        / len(score_components),
        4,
    )

    failure_reasons = []

    if missing_terms:
        failure_reasons.append(f"missing_expected_terms:{missing_terms}")

    if not answer_mentions_patient:
        failure_reasons.append("answer_missing_patient")

    if not answer_mentions_document:
        failure_reasons.append("answer_missing_document_or_type")

    if not answer_has_citation_signal:
        failure_reasons.append("answer_missing_citation_signal")

    if not source_match:
        failure_reasons.append(f"source_not_matching_expected:{source_mismatch_reasons[:8]}")

    return {
        "passed": passed,
        "score": score,
        "score_components": score_components,
        "missing_terms": missing_terms,
        "failure_reasons": failure_reasons,
    }


def select_index_modes(index_mode: str) -> List[str]:
    """Seleciona índices para avaliar."""
    if index_mode == "both":
        return ["baseline", "semantic"]

    return [index_mode]


def evaluate_index_mode(
    index_mode: str,
    s3_client,
    bedrock_client,
    bucket: str,
    examples: List[Dict[str, Any]],
    embedding_model_id: str,
    llm_model_id: str,
    top_k: int,
    max_tokens: int,
    temperature: float,
    use_local_cache: bool,
) -> List[Dict[str, Any]]:
    """Avalia respostas geradas por um índice."""
    print("\n" + "#" * 90)
    print(f"Avaliando respostas com índice: {index_mode}")
    print("#" * 90)

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=bucket,
        index_mode=index_mode,
        use_local_cache=use_local_cache,
    )

    evaluated_records = []

    for idx, expected in enumerate(examples, start=1):
        question = expected["question"]

        print("\n" + "=" * 90)
        print(f"[{index_mode}] Pergunta {idx}/{len(examples)}")
        print(question)

        retrieved_results = search_faiss(
            index=index,
            metadata=metadata,
            question=question,
            bedrock_client=bedrock_client,
            embedding_model_id=embedding_model_id,
            top_k=top_k,
        )

        context = build_context(retrieved_results)

        answer = invoke_claude_answer(
            bedrock_client=bedrock_client,
            model_id=llm_model_id,
            question=question,
            context=context,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        sources = [
            compact_source(result)
            for result in retrieved_results
        ]

        evaluation = evaluate_answer(
            expected=expected,
            answer=answer,
            sources=sources,
        )

        record = {
            "evaluation_type": "answer",
            "index_mode": index_mode,
            "question": question,
            "expected": expected,
            "answer": answer,
            "evaluation": evaluation,
            "sources": sources,
        }

        print(f"Passed: {evaluation['passed']}")
        print(f"Score: {evaluation['score']}")

        if evaluation["failure_reasons"]:
            print("Falhas:")
            print(json.dumps(evaluation["failure_reasons"], ensure_ascii=False, indent=2))

        evaluated_records.append(record)

    return evaluated_records


def save_jsonl(records: List[Dict[str, Any]], output_file: Path) -> None:
    """Salva avaliação em JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Imprime resumo agregado."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        grouped.setdefault(record["index_mode"], []).append(record)

    summaries = []

    for index_mode, items in grouped.items():
        total = len(items)
        passed = sum(1 for item in items if item["evaluation"]["passed"])
        avg_score = (
            sum(item["evaluation"]["score"] for item in items) / total
            if total
            else 0.0
        )

        summaries.append({
            "index_mode": index_mode,
            "total_questions": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_score": round(avg_score, 4),
        })

    print("\nResumo final")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia respostas finais do RAG clínico contra ground truth em JSONL."
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket S3 com artefatos FAISS.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--index-mode",
        choices=["baseline", "semantic", "both"],
        default="both",
        help="Índice usado para gerar respostas.",
    )

    parser.add_argument(
        "--eval-file",
        default=DEFAULT_EVAL_FILE,
        help="Arquivo ground truth JSONL.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL de saída.",
    )

    parser.add_argument(
        "--embedding-model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Modelo de embedding no Bedrock.",
    )

    parser.add_argument(
        "--llm-model-id",
        default=DEFAULT_LLM_MODEL_ID,
        help="Modelo LLM ou ARN do inference profile no Bedrock.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Quantidade de chunks usados como contexto.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens da resposta.",
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
            "Por padrão, sempre baixa do S3 para evitar metadados antigos."
        ),
    )

    args = parser.parse_args()

    examples = load_ground_truth_dataset(args.eval_file)
    index_modes = select_index_modes(args.index_mode)

    print("Iniciando avaliação de respostas")
    print(f"Dataset: {args.eval_file}")
    print(f"Perguntas: {len(examples)}")
    print(f"Index mode: {args.index_mode}")
    print(f"Top-K: {args.top_k}")
    print(f"Output: {args.output_jsonl}")
    print(f"Cache local: {args.use_local_cache}")

    s3_client = boto3.client("s3", region_name=args.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    all_records = []

    for index_mode in index_modes:
        records = evaluate_index_mode(
            index_mode=index_mode,
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            bucket=args.bucket,
            examples=examples,
            embedding_model_id=args.embedding_model_id,
            llm_model_id=args.llm_model_id,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_local_cache=args.use_local_cache,
        )
        all_records.extend(records)

    save_jsonl(all_records, Path(args.output_jsonl))
    print_summary(all_records)

    print("\nArquivo gerado:")
    print(args.output_jsonl)


if __name__ == "__main__":
    main()