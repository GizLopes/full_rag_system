# OBJETIVO PRINCIPAL
# Curar candidatos derivados do Streamlit e gerar ground truth JSONL validado para avaliação offline.

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT_JSONL = "curation_candidates.jsonl"
DEFAULT_OUTPUT_JSONL = "curated_ground_truth_dataset.jsonl"
DEFAULT_REVIEW_TEMPLATE_JSONL = "curation_review_template.jsonl"

APPROVED_STATUSES = {"approved", "curated", "validated"}
REJECTED_STATUSES = {"rejected", "discarded", "invalid"}


def load_jsonl(input_file: str) -> List[Dict[str, Any]]:
    """Carrega arquivo JSONL."""
    path = Path(input_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}. "
            "Execute antes: python 04_UI_convert_interaction_logs.py"
        )

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido na linha {line_number}.") from exc

    return records


def save_jsonl(records: List[Dict[str, Any]], output_file: str) -> None:
    """Salva registros em JSONL."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    """Normaliza texto simples."""
    return str(value or "").strip()


def empty_to_none(value: Any) -> Optional[str]:
    """Converte vazio em None."""
    text = normalize_text(value)

    if not text:
        return None

    return text


def parse_expected_terms(value: Any) -> List[str]:
    """Converte expected_terms para lista."""
    if value is None:
        return []

    if isinstance(value, list):
        return [
            normalize_text(item)
            for item in value
            if normalize_text(item)
        ]

    text = normalize_text(value)

    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                return [
                    normalize_text(item)
                    for item in parsed
                    if normalize_text(item)
                ]
        except json.JSONDecodeError:
            pass

    return [
        item.strip()
        for item in re.split(r"[|;,]", text)
        if item.strip()
    ]


def get_curation_status(candidate: Dict[str, Any]) -> str:
    """Obtém status de curadoria."""
    return normalize_text(candidate.get("curation_status")).lower() or "pending"


def should_skip_candidate(candidate: Dict[str, Any]) -> bool:
    """Descarta candidatos rejeitados."""
    return get_curation_status(candidate) in REJECTED_STATUSES


def should_include_candidate(
    candidate: Dict[str, Any],
    include_pending: bool,
    auto_approve_positive: bool,
) -> bool:
    """
    Decide se candidato entra no dataset curado.

    Por padrão, apenas status approved/curated/validated entra.
    """
    status = get_curation_status(candidate)

    if status in REJECTED_STATUSES:
        return False

    if status in APPROVED_STATUSES:
        return True

    if include_pending:
        return True

    if auto_approve_positive and candidate.get("user_feedback") == "positive":
        return True

    return False


def apply_suggestions_if_requested(
    candidate: Dict[str, Any],
    use_suggestions: bool,
) -> Dict[str, Any]:
    """Preenche expected_* com suggested_* quando solicitado."""
    updated = dict(candidate)

    if not use_suggestions:
        return updated

    mapping = {
        "expected_chunk_id": "suggested_expected_chunk_id",
        "expected_document_type": "suggested_expected_document_type",
        "expected_document_name": "suggested_expected_document_name",
        "expected_patient_id": "suggested_expected_patient_id",
        "expected_patient_name": "suggested_expected_patient_name",
    }

    for expected_field, suggested_field in mapping.items():
        if not normalize_text(updated.get(expected_field)):
            updated[expected_field] = updated.get(suggested_field, "")

    if not normalize_text(updated.get("expected_answer")):
        updated["expected_answer"] = updated.get("generated_answer", "")

    return updated


def build_review_template_record(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cria registro para revisão humana.

    O arquivo de template pode ser editado manualmente e depois usado como input
    deste mesmo script.
    """
    return {
        "candidate_id": candidate.get("candidate_id"),
        "source_interaction_id": candidate.get("source_interaction_id"),
        "source_timestamp_utc": candidate.get("source_timestamp_utc"),
        "curation_status": candidate.get("curation_status", "pending"),
        "curation_notes": candidate.get("curation_notes", ""),
        "question": candidate.get("question"),
        "generated_answer": candidate.get("generated_answer"),
        "user_feedback": candidate.get("user_feedback"),
        "feedback_comment": candidate.get("feedback_comment"),
        "needs_curation": candidate.get("needs_curation"),
        "suggested_expected_chunk_id": candidate.get("suggested_expected_chunk_id"),
        "suggested_expected_document_type": candidate.get("suggested_expected_document_type"),
        "suggested_expected_document_name": candidate.get("suggested_expected_document_name"),
        "suggested_expected_patient_id": candidate.get("suggested_expected_patient_id"),
        "suggested_expected_patient_name": candidate.get("suggested_expected_patient_name"),
        "suggested_expected_page_start": candidate.get("suggested_expected_page_start"),
        "suggested_expected_page_end": candidate.get("suggested_expected_page_end"),
        "expected_chunk_id": candidate.get("expected_chunk_id", ""),
        "expected_document_type": candidate.get("expected_document_type", ""),
        "expected_document_name": candidate.get("expected_document_name", ""),
        "expected_patient_id": candidate.get("expected_patient_id", ""),
        "expected_patient_name": candidate.get("expected_patient_name", ""),
        "expected_terms": parse_expected_terms(candidate.get("expected_terms")),
        "expected_answer": candidate.get("expected_answer", ""),
        "sources": candidate.get("sources", []),
    }


def validate_curated_candidate(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida campos mínimos para virar ground truth."""
    errors = []

    required_fields = [
        "question",
        "expected_document_type",
        "expected_document_name",
        "expected_patient_id",
        "expected_patient_name",
    ]

    for field in required_fields:
        if not normalize_text(candidate.get(field)):
            errors.append(f"missing:{field}")

    expected_terms = parse_expected_terms(candidate.get("expected_terms"))

    if not isinstance(expected_terms, list):
        errors.append("invalid:expected_terms")

    if not normalize_text(candidate.get("expected_answer")):
        errors.append("warning:expected_answer_empty")

    hard_errors = [
        error
        for error in errors
        if not error.startswith("warning:")
    ]

    return len(hard_errors) == 0, errors


def build_curated_ground_truth_record(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Converte candidato curado em linha de ground truth."""
    return {
        "question": normalize_text(candidate.get("question")),
        "expected_chunk_id": normalize_text(candidate.get("expected_chunk_id")),
        "expected_document_type": normalize_text(candidate.get("expected_document_type")),
        "expected_document_name": normalize_text(candidate.get("expected_document_name")),
        "expected_patient_id": normalize_text(candidate.get("expected_patient_id")),
        "expected_patient_name": normalize_text(candidate.get("expected_patient_name")),
        "expected_terms": parse_expected_terms(candidate.get("expected_terms")),
        "expected_answer": normalize_text(candidate.get("expected_answer")),
        "evaluation_scope": "retrieval_answer",
        "dataset_type": "curated_from_streamlit_interaction",
        "source_interaction_id": candidate.get("source_interaction_id"),
        "source_timestamp_utc": candidate.get("source_timestamp_utc"),
        "curation_status": get_curation_status(candidate),
        "curation_notes": normalize_text(candidate.get("curation_notes")),
    }


def curate_candidates(
    input_jsonl: str,
    output_jsonl: str,
    review_template_jsonl: str,
    use_suggestions: bool,
    include_pending: bool,
    auto_approve_positive: bool,
    fail_on_invalid: bool,
) -> Dict[str, Any]:
    """Executa curadoria dos candidatos."""
    candidates = load_jsonl(input_jsonl)

    review_template_records = []
    curated_records = []
    invalid_records = []
    skipped_records = []

    for candidate in candidates:
        if should_skip_candidate(candidate):
            skipped_records.append({
                "candidate_id": candidate.get("candidate_id"),
                "reason": f"status:{get_curation_status(candidate)}",
            })
            continue

        template_record = build_review_template_record(candidate)
        review_template_records.append(template_record)

        if not should_include_candidate(
            candidate=candidate,
            include_pending=include_pending,
            auto_approve_positive=auto_approve_positive,
        ):
            skipped_records.append({
                "candidate_id": candidate.get("candidate_id"),
                "reason": f"status_not_approved:{get_curation_status(candidate)}",
            })
            continue

        candidate_to_curate = apply_suggestions_if_requested(
            candidate=candidate,
            use_suggestions=use_suggestions,
        )

        is_valid, validation_messages = validate_curated_candidate(candidate_to_curate)

        if not is_valid:
            invalid_records.append({
                "candidate_id": candidate.get("candidate_id"),
                "source_interaction_id": candidate.get("source_interaction_id"),
                "validation_messages": validation_messages,
                "question": candidate.get("question"),
            })

            if fail_on_invalid:
                continue

            continue

        curated_record = build_curated_ground_truth_record(candidate_to_curate)
        curated_record["validation_messages"] = validation_messages

        curated_records.append(curated_record)

    save_jsonl(
        records=review_template_records,
        output_file=review_template_jsonl,
    )

    save_jsonl(
        records=curated_records,
        output_file=output_jsonl,
    )

    return {
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "review_template_jsonl": review_template_jsonl,
        "total_candidates": len(candidates),
        "review_template_records": len(review_template_records),
        "curated_records": len(curated_records),
        "invalid_records": len(invalid_records),
        "skipped_records": len(skipped_records),
        "invalid_details": invalid_records,
        "skipped_details": skipped_records[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cura candidatos derivados do Streamlit e gera curated_ground_truth_dataset.jsonl. "
            "Depende do arquivo gerado por 04_UI_convert_interaction_logs.py."
        )
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Arquivo curation_candidates.jsonl gerado a partir dos logs do Streamlit.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL de ground truth curado.",
    )

    parser.add_argument(
        "--review-template-jsonl",
        default=DEFAULT_REVIEW_TEMPLATE_JSONL,
        help="Arquivo JSONL auxiliar para revisão humana dos candidatos.",
    )

    parser.add_argument(
        "--use-suggestions",
        action="store_true",
        help=(
            "Preenche expected_* usando suggested_expected_* quando os campos estiverem vazios. "
            "Use apenas para acelerar revisão, não para afirmar curadoria automática."
        ),
    )

    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Inclui candidatos pending no output, desde que tenham expected_* válidos.",
    )

    parser.add_argument(
        "--auto-approve-positive",
        action="store_true",
        help="Inclui interações com user_feedback=positive, mesmo sem status aprovado.",
    )

    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Mantém comportamento estrito e reporta inválidos sem incluí-los.",
    )

    args = parser.parse_args()

    summary = curate_candidates(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        review_template_jsonl=args.review_template_jsonl,
        use_suggestions=args.use_suggestions,
        include_pending=args.include_pending,
        auto_approve_positive=args.auto_approve_positive,
        fail_on_invalid=args.fail_on_invalid,
    )

    print("Curadoria concluída")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if summary["curated_records"] == 0:
        print(
            "\nNenhum registro curado foi gerado. "
            "Edite curation_review_template.jsonl, preencha expected_* e marque curation_status como approved."
        )


if __name__ == "__main__":
    main()