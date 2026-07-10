# OBJETIVO PRINCIPAL
# Unir ground truth dataset e ground truth curado do Streamlit em um dataset final JSONL para avaliação offline.

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SEED_JSONL = "ground_truth_evaluation_dataset.jsonl"
DEFAULT_CURATED_JSONL = "curated_ground_truth_dataset.jsonl"
DEFAULT_OUTPUT_JSONL = "full_ground_truth_evaluation_dataset.jsonl"

VALID_DATASET_TYPES = {
    "ground_truth_seed",
    "curated_from_streamlit_interaction",
}


def normalize_text(value: Any) -> str:
    """Normaliza texto para comparação e deduplicação."""
    text = str(value or "")
    text = text.replace("\x00", " ")
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_display_text(value: Any) -> str:
    """Normaliza texto preservando forma legível."""
    return str(value or "").strip()


def parse_expected_terms(value: Any) -> List[str]:
    """Converte expected_terms para lista."""
    if value is None:
        return []

    if isinstance(value, list):
        return [
            normalize_display_text(item)
            for item in value
            if normalize_display_text(item)
        ]

    text = normalize_display_text(value)

    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    normalize_display_text(item)
                    for item in parsed
                    if normalize_display_text(item)
                ]
        except json.JSONDecodeError:
            pass

    return [
        item.strip()
        for item in re.split(r"[|;,]", text)
        if item.strip()
    ]


def load_jsonl(input_file: str, required: bool) -> List[Dict[str, Any]]:
    """Carrega arquivo JSONL."""
    path = Path(input_file)

    if not path.exists():
        if required:
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {input_file}")

        print(f"Aviso: arquivo opcional não encontrado, ignorando: {input_file}")
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL inválido em {input_file}, linha {line_number}."
                ) from exc

    return records


def save_jsonl(records: List[Dict[str, Any]], output_file: str) -> None:
    """Salva registros em JSONL."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_record(record: Dict[str, Any], default_dataset_type: str) -> Dict[str, Any]:
    """Padroniza campos do dataset final."""
    dataset_type = normalize_display_text(record.get("dataset_type")) or default_dataset_type

    if dataset_type not in VALID_DATASET_TYPES:
        dataset_type = default_dataset_type

    evaluation_scope = normalize_display_text(record.get("evaluation_scope"))

    if not evaluation_scope:
        evaluation_scope = "retrieval_answer" if dataset_type == "curated_from_streamlit_interaction" else "retrieval"

    return {
        "question": normalize_display_text(record.get("question")),
        "expected_chunk_id": normalize_display_text(record.get("expected_chunk_id")),
        "expected_document_type": normalize_display_text(record.get("expected_document_type")),
        "expected_document_name": normalize_display_text(record.get("expected_document_name")),
        "expected_patient_id": normalize_display_text(record.get("expected_patient_id")),
        "expected_patient_name": normalize_display_text(record.get("expected_patient_name")),
        "expected_terms": parse_expected_terms(record.get("expected_terms")),
        "expected_answer": normalize_display_text(record.get("expected_answer")),
        "evaluation_scope": evaluation_scope,
        "dataset_type": dataset_type,
        "source_interaction_id": normalize_display_text(record.get("source_interaction_id")),
        "source_timestamp_utc": normalize_display_text(record.get("source_timestamp_utc")),
        "curation_status": normalize_display_text(record.get("curation_status")),
        "curation_notes": normalize_display_text(record.get("curation_notes")),
    }


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida campos mínimos para avaliação."""
    errors = []

    required_fields = [
        "question",
        "expected_document_type",
        "expected_document_name",
        "expected_patient_id",
        "expected_patient_name",
    ]

    for field in required_fields:
        if not normalize_display_text(record.get(field)):
            errors.append(f"missing:{field}")

    if not isinstance(record.get("expected_terms"), list):
        errors.append("invalid:expected_terms")

    if record.get("dataset_type") == "curated_from_streamlit_interaction":
        if not normalize_display_text(record.get("source_interaction_id")):
            errors.append("warning:missing_source_interaction_id")

        if not normalize_display_text(record.get("expected_answer")):
            errors.append("warning:missing_expected_answer")

    hard_errors = [
        error
        for error in errors
        if not error.startswith("warning:")
    ]

    return len(hard_errors) == 0, errors


def build_dedup_key(record: Dict[str, Any], strategy: str) -> str:
    """Cria chave de deduplicação."""
    if strategy == "question":
        return normalize_text(record.get("question"))

    if strategy == "question_patient_document":
        return "|".join([
            normalize_text(record.get("question")),
            normalize_text(record.get("expected_patient_id")),
            normalize_text(record.get("expected_document_name")),
            normalize_text(record.get("expected_document_type")),
        ])

    if strategy == "source_interaction_id":
        source_interaction_id = normalize_text(record.get("source_interaction_id"))

        if source_interaction_id:
            return f"source_interaction_id:{source_interaction_id}"

        return "|".join([
            normalize_text(record.get("question")),
            normalize_text(record.get("expected_patient_id")),
            normalize_text(record.get("expected_document_name")),
        ])

    raise ValueError(f"Estratégia de deduplicação inválida: {strategy}")


def merge_records(
    seed_records: List[Dict[str, Any]],
    curated_records: List[Dict[str, Any]],
    dedup_strategy: str,
    prefer_curated: bool,
    strict: bool,
) -> Dict[str, Any]:
    """Une seed e curado com validação e deduplicação."""
    normalized_seed = [
        normalize_record(record, default_dataset_type="ground_truth_seed")
        for record in seed_records
    ]

    normalized_curated = [
        normalize_record(record, default_dataset_type="curated_from_streamlit_interaction")
        for record in curated_records
    ]

    if prefer_curated:
        ordered_records = normalized_curated + normalized_seed
    else:
        ordered_records = normalized_seed + normalized_curated

    merged_by_key: Dict[str, Dict[str, Any]] = {}
    invalid_records = []
    duplicate_records = []

    for record in ordered_records:
        is_valid, validation_messages = validate_record(record)

        record["merge_validation_messages"] = validation_messages

        if not is_valid:
            invalid_records.append({
                "question": record.get("question"),
                "dataset_type": record.get("dataset_type"),
                "source_interaction_id": record.get("source_interaction_id"),
                "validation_messages": validation_messages,
            })

            if strict:
                continue

            continue

        dedup_key = build_dedup_key(record, strategy=dedup_strategy)

        if dedup_key in merged_by_key:
            duplicate_records.append({
                "dedup_key": dedup_key,
                "kept_dataset_type": merged_by_key[dedup_key].get("dataset_type"),
                "skipped_dataset_type": record.get("dataset_type"),
                "question": record.get("question"),
            })
            continue

        record["merge_dedup_key"] = dedup_key
        merged_by_key[dedup_key] = record

    merged_records = list(merged_by_key.values())

    return {
        "merged_records": merged_records,
        "invalid_records": invalid_records,
        "duplicate_records": duplicate_records,
        "summary": {
            "seed_input_records": len(seed_records),
            "curated_input_records": len(curated_records),
            "merged_records": len(merged_records),
            "invalid_records": len(invalid_records),
            "duplicate_records": len(duplicate_records),
            "dedup_strategy": dedup_strategy,
            "prefer_curated": prefer_curated,
            "strict": strict,
        },
    }


def print_summary(result: Dict[str, Any], output_jsonl: str) -> None:
    """Imprime resumo do merge."""
    dataset_counts: Dict[str, int] = {}

    for record in result["merged_records"]:
        dataset_type = record.get("dataset_type") or "unknown"
        dataset_counts[dataset_type] = dataset_counts.get(dataset_type, 0) + 1

    summary = dict(result["summary"])
    summary["dataset_counts"] = dataset_counts
    summary["output_jsonl"] = output_jsonl

    print("Merge concluído")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if result["invalid_records"]:
        print("\nRegistros inválidos ignorados:")
        print(json.dumps(result["invalid_records"][:20], ensure_ascii=False, indent=2))

    if result["duplicate_records"]:
        print("\nDuplicados ignorados:")
        print(json.dumps(result["duplicate_records"][:20], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Une ground truth seed e ground truth curado derivado do Streamlit. "
            "Depende de curated_ground_truth_dataset.jsonl gerado por 05_UI_curate_ground_truth_dataset.py."
        )
    )

    parser.add_argument(
        "--seed-jsonl",
        default=DEFAULT_SEED_JSONL,
        help="Dataset seed ground truth em JSONL.",
    )

    parser.add_argument(
        "--curated-jsonl",
        default=DEFAULT_CURATED_JSONL,
        help="Dataset curado a partir das interações do Streamlit.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Dataset final unificado em JSONL.",
    )

    parser.add_argument(
        "--dedup-strategy",
        choices=[
            "question",
            "question_patient_document",
            "source_interaction_id",
        ],
        default="question_patient_document",
        help="Estratégia para evitar duplicidade.",
    )

    parser.add_argument(
        "--prefer-curated",
        action="store_true",
        help="Em caso de duplicidade, prioriza registro curado em vez do seed.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Mantém validação estrita e ignora registros inválidos.",
    )

    parser.add_argument(
        "--allow-missing-curated",
        action="store_true",
        help="Permite executar mesmo sem curated_ground_truth_dataset.jsonl.",
    )

    args = parser.parse_args()

    seed_records = load_jsonl(
        input_file=args.seed_jsonl,
        required=True,
    )

    curated_records = load_jsonl(
        input_file=args.curated_jsonl,
        required=not args.allow_missing_curated,
    )

    result = merge_records(
        seed_records=seed_records,
        curated_records=curated_records,
        dedup_strategy=args.dedup_strategy,
        prefer_curated=args.prefer_curated,
        strict=args.strict,
    )

    save_jsonl(
        records=result["merged_records"],
        output_file=args.output_jsonl,
    )

    print_summary(
        result=result,
        output_jsonl=args.output_jsonl,
    )


if __name__ == "__main__":
    main()