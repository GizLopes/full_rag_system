# OBJETIVO PRINCIPAL
# Avaliar retrieval do RAG clínico, medindo Recall@K, MRR e acerto por documento, paciente e termos esperados.

import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

DEFAULT_TOP_K = 5
DEFAULT_OUTPUT_JSONL = "retrieval_eval_results.jsonl"

DEFAULT_EVAL_DATASET = [
    {
        "question": "Qual foi o resultado da creatinina da paciente Gabriela Lima?",
        "expected_document_type": "hemograma_e_bioquimica",
        "expected_document_name": "hemograma_e_bioquimica_032026.pdf",
        "expected_patient_id": "P007",
        "expected_patient_name": "Gabriela Lima",
        "expected_terms": ["creatinina_mg_dl", "1.46"],
    },
    {
        "question": "Qual medicação atual aparece para Carla Mendes na consulta ambulatorial?",
        "expected_document_type": "consultas_ambulatoriais",
        "expected_document_name": "consultas_ambulatoriais_032026.pdf",
        "expected_patient_id": "P003",
        "expected_patient_name": "Carla Mendes",
        "expected_terms": ["medicacoes_atuais", "Sulfato ferroso"],
    },
    {
        "question": "Qual foi a prescrição de alta de Bruno Almeida?",
        "expected_document_type": "alta_hospitalar",
        "expected_document_name": "alta_hospitalar.pdf",
        "expected_patient_id": "P002",
        "expected_patient_name": "Bruno Almeida",
        "expected_terms": ["prescricao_alta", "Metformina 850 mg"],
    },
    {
        "question": "Qual foi o risco cardiovascular de Henrique Rocha no parecer cardiologista?",
        "expected_document_type": "parecer_cardiologista",
        "expected_document_name": "parecer_cardiologista.pdf",
        "expected_patient_id": "P008",
        "expected_patient_name": "Henrique Rocha",
        "expected_terms": ["risco_cardiovascular", "alto"],
    },
    {
        "question": "Qual laudo de ressonância aparece para Diego Santos?",
        "expected_document_type": "ressonancia_coluna",
        "expected_document_name": "ressonancia_coluna.pdf",
        "expected_patient_id": "P004",
        "expected_patient_name": "Diego Santos",
        "expected_terms": [],
    },
]


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


def normalize_text(text: Any) -> str:
    """Normaliza texto para comparação."""
    text = str(text or "")
    text = text.replace("\x00", " ")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
        item["position"] = int(position)
        results.append(item)

    return results


def parse_expected_terms(value: Any) -> List[str]:
    """Converte termos esperados de JSON em lista."""
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


def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Padroniza campos esperados do dataset."""
    return {
        "question": str(example.get("question", "")).strip(),
        "expected_chunk_id": empty_to_none(example.get("expected_chunk_id")),
        "expected_document_type": empty_to_none(example.get("expected_document_type")),
        "expected_document_name": empty_to_none(example.get("expected_document_name")),
        "expected_patient_id": empty_to_none(example.get("expected_patient_id")),
        "expected_patient_name": empty_to_none(example.get("expected_patient_name")),
        "expected_terms": parse_expected_terms(example.get("expected_terms")),
    }


def empty_to_none(value: Any) -> Optional[str]:
    """Converte string vazia em None."""
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


def load_eval_dataset(eval_file: Optional[str]) -> List[Dict[str, Any]]:
    """Carrega dataset de avaliação ou usa dataset default."""
    if not eval_file:
        return [
            normalize_example(example)
            for example in DEFAULT_EVAL_DATASET
        ]

    path = Path(eval_file)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de avaliação não encontrado: {eval_file}")

    if path.suffix.lower() == ".jsonl":
        records = []

        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(normalize_example(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL inválido na linha {line_number}") from exc

        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("examples", [])

        if not isinstance(data, list):
            raise ValueError("Arquivo JSON deve conter uma lista ou {'examples': [...]}.")

        return [
            normalize_example(example)
            for example in data
        ]

    raise ValueError("Formato suportado: .jsonl ou .json")


def build_searchable_result_text(result: Dict[str, Any]) -> str:
    """Monta texto consolidado para avaliar match."""
    metadata = result.get("metadata") or {}

    fields = [
        result.get("chunk_id"),
        result.get("document_name"),
        result.get("document_type"),
        result.get("patient_id"),
        result.get("patient_name"),
        result.get("clinical_section"),
        result.get("text"),
        metadata.get("document_name"),
        metadata.get("document_type"),
        metadata.get("patient_id"),
        metadata.get("patient_name"),
        metadata.get("clinical_section"),
    ]

    return " ".join(str(field) for field in fields if field)


def field_matches(expected: Optional[str], actual: Any, searchable_text: str) -> bool:
    """Valida um campo esperado contra campo real e texto consolidado."""
    if not expected:
        return True

    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)

    if actual_norm and expected_norm == actual_norm:
        return True

    return expected_norm in normalize_text(searchable_text)


def result_is_hit(example: Dict[str, Any], result: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verifica se um resultado atende aos critérios esperados.

    A regra é conservadora: todo campo esperado informado precisa bater.
    """
    reasons = []
    searchable_text = build_searchable_result_text(result)

    checks = [
        (
            "chunk_id",
            field_matches(
                example.get("expected_chunk_id"),
                result.get("chunk_id"),
                searchable_text,
            ),
        ),
        (
            "document_type",
            field_matches(
                example.get("expected_document_type"),
                result.get("document_type"),
                searchable_text,
            ),
        ),
        (
            "document_name",
            field_matches(
                example.get("expected_document_name"),
                result.get("document_name"),
                searchable_text,
            ),
        ),
        (
            "patient_id",
            field_matches(
                example.get("expected_patient_id"),
                result.get("patient_id"),
                searchable_text,
            ),
        ),
        (
            "patient_name",
            field_matches(
                example.get("expected_patient_name"),
                result.get("patient_name"),
                searchable_text,
            ),
        ),
    ]

    for field, passed in checks:
        if not passed:
            reasons.append(f"missing_or_mismatch_{field}")

    expected_terms = example.get("expected_terms") or []
    searchable_norm = normalize_text(searchable_text)

    missing_terms = [
        term
        for term in expected_terms
        if normalize_text(term) not in searchable_norm
    ]

    if missing_terms:
        reasons.append(f"missing_terms:{missing_terms}")

    return len(reasons) == 0, reasons


def evaluate_single_question(
    example: Dict[str, Any],
    results: List[Dict],
    top_k: int,
) -> Dict[str, Any]:
    """Avalia uma pergunta individual."""
    hit_rank = None
    hit_result = None
    hit_reasons_by_rank = []
    hit_count = 0

    for result in results[:top_k]:
        is_hit, reasons = result_is_hit(example, result)

        hit_reasons_by_rank.append({
            "rank": result.get("rank"),
            "chunk_id": result.get("chunk_id"),
            "is_hit": is_hit,
            "reasons": reasons,
        })

        if is_hit:
            hit_count += 1

            if hit_rank is None:
                hit_rank = result.get("rank")
                hit_result = result

    recall_at_k = 1.0 if hit_rank is not None else 0.0
    mrr = 1.0 / hit_rank if hit_rank else 0.0
    precision_at_k = hit_count / max(len(results[:top_k]), 1)

    return {
        "question": example["question"],
        "expected": {
            "expected_chunk_id": example.get("expected_chunk_id"),
            "expected_document_type": example.get("expected_document_type"),
            "expected_document_name": example.get("expected_document_name"),
            "expected_patient_id": example.get("expected_patient_id"),
            "expected_patient_name": example.get("expected_patient_name"),
            "expected_terms": example.get("expected_terms"),
        },
        "hit": hit_rank is not None,
        "hit_rank": hit_rank,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "precision_at_k": precision_at_k,
        "best_score": results[0].get("score") if results else None,
        "hit_result": compact_result(hit_result) if hit_result else None,
        "rank_checks": hit_reasons_by_rank,
        "retrieved": [
            compact_result(result)
            for result in results[:top_k]
        ],
    }


def compact_result(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Reduz resultado para logs de avaliação."""
    if not result:
        return None

    return {
        "rank": result.get("rank"),
        "score": result.get("score"),
        "chunk_id": result.get("chunk_id"),
        "document_name": result.get("document_name"),
        "document_type": result.get("document_type"),
        "chunk_number": result.get("chunk_number"),
        "total_chunks": result.get("total_chunks"),
        "page_start": result.get("page_start"),
        "page_end": result.get("page_end"),
        "s3_uri": result.get("s3_uri"),
        "chunk_strategy": result.get("chunk_strategy"),
        "patient_id": result.get("patient_id"),
        "patient_name": result.get("patient_name"),
        "clinical_section": result.get("clinical_section"),
        "text_preview": str(result.get("text", ""))[:400],
    }


def summarize_results(
    index_mode: str,
    question_results: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """Calcula métricas agregadas."""
    total = len(question_results)

    if total == 0:
        return {
            "index_mode": index_mode,
            "top_k": top_k,
            "total_questions": 0,
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "precision_at_k": 0.0,
            "hit_count": 0,
        }

    hit_count = sum(1 for item in question_results if item["hit"])

    return {
        "index_mode": index_mode,
        "top_k": top_k,
        "total_questions": total,
        "hit_count": hit_count,
        "miss_count": total - hit_count,
        "recall_at_k": round(
            sum(item["recall_at_k"] for item in question_results) / total,
            4,
        ),
        "mrr": round(
            sum(item["mrr"] for item in question_results) / total,
            4,
        ),
        "precision_at_k": round(
            sum(item["precision_at_k"] for item in question_results) / total,
            4,
        ),
        "avg_best_score": round(
            sum((item["best_score"] or 0.0) for item in question_results) / total,
            4,
        ),
        "questions": question_results,
    }


def evaluate_index_mode(
    s3_client,
    bedrock_client,
    bucket: str,
    index_mode: str,
    examples: List[Dict[str, Any]],
    embedding_model_id: str,
    top_k: int,
    use_local_cache: bool,
) -> Dict[str, Any]:
    """Avalia um índice específico."""
    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=bucket,
        index_mode=index_mode,
        use_local_cache=use_local_cache,
    )

    question_results = []

    for idx, example in enumerate(examples, start=1):
        question = example["question"]

        print("\n" + "=" * 80)
        print(f"[{index_mode}] Avaliando {idx}/{len(examples)}")
        print(f"Pergunta: {question}")

        results = search_faiss(
            index=index,
            metadata=metadata,
            question=question,
            bedrock_client=bedrock_client,
            embedding_model_id=embedding_model_id,
            top_k=top_k,
        )

        evaluation = evaluate_single_question(
            example=example,
            results=results,
            top_k=top_k,
        )

        print(f"Hit: {evaluation['hit']}")
        print(f"Hit rank: {evaluation['hit_rank']}")
        print(f"Best score: {evaluation['best_score']}")

        question_results.append(evaluation)

    return summarize_results(
        index_mode=index_mode,
        question_results=question_results,
        top_k=top_k,
    )


def save_jsonl(index_summaries: List[Dict[str, Any]], output_file: Path) -> None:
    """Salva resultados por pergunta em JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for summary in index_summaries:
            index_mode = summary["index_mode"]

            for question_result in summary.get("questions", []):
                row = {
                    "index_mode": index_mode,
                    **question_result,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_index_modes(index_mode: str) -> List[str]:
    """Resolve quais índices avaliar."""
    if index_mode == "both":
        return ["baseline", "semantic"]

    return [index_mode]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia qualidade do retrieval FAISS com Recall@K, MRR e Precision@K."
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
        choices=["baseline", "semantic", "both"],
        default="both",
        help="Índice a avaliar. Default: both.",
    )

    parser.add_argument(
        "--eval-file",
        default=None,
        help="Arquivo .jsonl ou .json com perguntas e expectativas.",
    )

    parser.add_argument(
        "--embedding-model-id",
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Modelo de embedding no Amazon Bedrock.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="K usado em Recall@K, MRR e Precision@K.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL por pergunta.",
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

    examples = load_eval_dataset(args.eval_file)

    if not examples:
        print("Nenhum exemplo de avaliação encontrado.")
        return

    index_modes = select_index_modes(args.index_mode)

    print("Iniciando avaliação de retrieval")
    print(f"Bucket: {args.bucket}")
    print(f"Index mode: {args.index_mode}")
    print(f"Total de perguntas: {len(examples)}")
    print(f"Top-K: {args.top_k}")
    print(f"Eval file: {args.eval_file or 'dataset default interno'}")
    print(f"Usar cache local: {args.use_local_cache}")

    summaries = []

    for index_mode in index_modes:
        summary = evaluate_index_mode(
            s3_client=s3_client,
            bedrock_client=bedrock_client,
            bucket=args.bucket,
            index_mode=index_mode,
            examples=examples,
            embedding_model_id=args.embedding_model_id,
            top_k=args.top_k,
            use_local_cache=args.use_local_cache,
        )
        summaries.append(summary)

    final_report = {
        "evaluation_type": "retrieval",
        "top_k": args.top_k,
        "total_questions": len(examples),
        "index_modes": index_modes,
        "summaries": [
            {
                key: value
                for key, value in summary.items()
                if key != "questions"
            }
            for summary in summaries
        ],
        "details": summaries,
    }

    save_jsonl(summaries, Path(args.output_jsonl))

    print("\nResumo final")
    print(json.dumps(final_report["summaries"], ensure_ascii=False, indent=2))

    print("\nArquivos gerados:")
    print(args.output_jsonl)

if __name__ == "__main__":
    main()