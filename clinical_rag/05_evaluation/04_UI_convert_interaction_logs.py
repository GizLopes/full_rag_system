# OBJETIVO PRINCIPAL
# Converter logs reais do Streamlit em candidatos de curadoria para futuro dataset ground truth.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_INPUT_JSONL = "../06_app_ui/interaction_logs.jsonl"
DEFAULT_OUTPUT_JSONL = "curation_candidates.jsonl"


def load_jsonl(input_file: str) -> List[Dict[str, Any]]:
    """Carrega registros JSONL."""
    path = Path(input_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}. "
            "Execute o Streamlit e gere interaction_logs.jsonl primeiro."
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


def get_sources(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Obtém fontes recuperadas de uma interação."""
    sources = record.get("sources")

    if isinstance(sources, list):
        return [
            source
            for source in sources
            if isinstance(source, dict)
        ]

    return []


def get_top_source(record: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna fonte rank 1 ou primeira fonte disponível."""
    sources = get_sources(record)

    if not sources:
        return {}

    ranked_sources = sorted(
        sources,
        key=lambda source: source.get("rank") or 999,
    )

    return ranked_sources[0]


def compact_sources(record: Dict[str, Any], max_sources: int) -> List[Dict[str, Any]]:
    """Reduz fontes para o arquivo de curadoria."""
    sources = get_sources(record)

    compacted = []

    for source in sources[:max_sources]:
        compacted.append({
            "rank": source.get("rank"),
            "score": source.get("score"),
            "chunk_id": source.get("chunk_id"),
            "document_name": source.get("document_name"),
            "document_type": source.get("document_type"),
            "patient_id": source.get("patient_id"),
            "patient_name": source.get("patient_name"),
            "clinical_section": source.get("clinical_section"),
            "chunk_number": source.get("chunk_number"),
            "total_chunks": source.get("total_chunks"),
            "page_start": source.get("page_start"),
            "page_end": source.get("page_end"),
            "s3_uri": source.get("s3_uri"),
            "chunk_strategy": source.get("chunk_strategy"),
            "embedding_strategy": source.get("embedding_strategy"),
            "text_preview": source.get("text_preview"),
        })

    return compacted


def should_include_record(
    record: Dict[str, Any],
    only_needs_curation: bool,
    only_with_feedback: bool,
) -> bool:
    """Define se a interação deve virar candidato."""
    if not normalize_text(record.get("question")):
        return False

    if not normalize_text(record.get("answer")):
        return False

    if only_needs_curation and record.get("needs_curation") is not True:
        return False

    if only_with_feedback and not record.get("user_feedback"):
        return False

    return True


def build_candidate(
    record: Dict[str, Any],
    max_sources: int,
) -> Dict[str, Any]:
    """
    Converte uma interação real em candidato de curadoria.

    Campos expected_* ficam vazios de propósito. Campos suggested_* vêm do top source
    e servem apenas como sugestão para revisão humana.
    """
    top_source = get_top_source(record)

    return {
        "candidate_id": record.get("interaction_id"),
        "source_interaction_id": record.get("interaction_id"),
        "source_timestamp_utc": record.get("timestamp_utc"),
        "candidate_type": "from_streamlit_interaction",
        "dataset_type": "curation_candidate",
        "evaluation_scope": "retrieval_answer",
        "curation_status": "pending",
        "curation_notes": "",
        "question": record.get("question"),
        "generated_answer": record.get("answer"),
        "user_feedback": record.get("user_feedback"),
        "feedback_comment": record.get("feedback_comment"),
        "needs_curation": record.get("needs_curation", False),
        "index_mode": record.get("index_mode"),
        "top_k": record.get("top_k"),
        "latency_seconds": record.get("latency_seconds"),
        "suggested_expected_chunk_id": top_source.get("chunk_id"),
        "suggested_expected_document_type": top_source.get("document_type"),
        "suggested_expected_document_name": top_source.get("document_name"),
        "suggested_expected_patient_id": top_source.get("patient_id"),
        "suggested_expected_patient_name": top_source.get("patient_name"),
        "suggested_expected_page_start": top_source.get("page_start"),
        "suggested_expected_page_end": top_source.get("page_end"),
        "expected_chunk_id": "",
        "expected_document_type": "",
        "expected_document_name": "",
        "expected_patient_id": "",
        "expected_patient_name": "",
        "expected_terms": [],
        "expected_answer": "",
        "sources": compact_sources(
            record=record,
            max_sources=max_sources,
        ),
    }


def convert_interaction_logs(
    input_jsonl: str,
    output_jsonl: str,
    only_needs_curation: bool,
    only_with_feedback: bool,
    max_sources: int,
) -> List[Dict[str, Any]]:
    """Converte interaction_logs.jsonl em curation_candidates.jsonl."""
    interactions = load_jsonl(input_jsonl)

    candidates = []

    for record in interactions:
        if not should_include_record(
            record=record,
            only_needs_curation=only_needs_curation,
            only_with_feedback=only_with_feedback,
        ):
            continue

        candidates.append(
            build_candidate(
                record=record,
                max_sources=max_sources,
            )
        )

    save_jsonl(
        records=candidates,
        output_file=output_jsonl,
    )

    return candidates


def print_summary(candidates: List[Dict[str, Any]], input_jsonl: str, output_jsonl: str) -> None:
    """Imprime resumo da conversão."""
    feedback_counts: Dict[str, int] = {}
    index_counts: Dict[str, int] = {}
    needs_curation_count = 0

    for candidate in candidates:
        feedback = candidate.get("user_feedback") or "not_rated"
        index_mode = candidate.get("index_mode") or "unknown"

        feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        index_counts[index_mode] = index_counts.get(index_mode, 0) + 1

        if candidate.get("needs_curation") is True:
            needs_curation_count += 1

    summary = {
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "total_candidates": len(candidates),
        "needs_curation_count": needs_curation_count,
        "feedback_counts": feedback_counts,
        "index_counts": index_counts,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte logs reais do Streamlit em candidatos de curadoria para ground truth."
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Arquivo interaction_logs.jsonl gerado pelo Streamlit.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL de candidatos de curadoria.",
    )

    parser.add_argument(
        "--only-needs-curation",
        action="store_true",
        help="Inclui apenas interações marcadas como needs_curation=true.",
    )

    parser.add_argument(
        "--only-with-feedback",
        action="store_true",
        help="Inclui apenas interações com feedback preenchido.",
    )

    parser.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="Máximo de fontes preservadas por candidato.",
    )

    args = parser.parse_args()

    candidates = convert_interaction_logs(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        only_needs_curation=args.only_needs_curation,
        only_with_feedback=args.only_with_feedback,
        max_sources=args.max_sources,
    )

    print("Conversão concluída")
    print_summary(
        candidates=candidates,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
    )


if __name__ == "__main__":
    main()