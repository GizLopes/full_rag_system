# OBJETIVO PRINCIPAL
# Centralizar logging em tempo real do Streamlit, registrando interações, feedback e indicação de curadoria em JSONL.

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_LOG_FILE = "interaction_logs.jsonl"
LOG_SCHEMA_VERSION = "1.0"

VALID_FEEDBACK_VALUES = {
    "positive",
    "partial",
    "negative",
    "source_issue",
    None,
}


def utc_now_iso() -> str:
    """Retorna timestamp UTC em ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def generate_interaction_id() -> str:
    """Gera ID único para cada interação real do app."""
    return str(uuid.uuid4())


def ensure_json_serializable(value: Any) -> Any:
    """Converte objetos não serializáveis para string."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def normalize_sources(sources: Any) -> List[Dict[str, Any]]:
    """Garante lista serializável de fontes."""
    if not isinstance(sources, list):
        return []

    normalized_sources = []

    for source in sources:
        if isinstance(source, dict):
            normalized_sources.append({
                key: ensure_json_serializable(value)
                for key, value in source.items()
            })
        else:
            normalized_sources.append({
                "value": ensure_json_serializable(source)
            })

    return normalized_sources


def validate_feedback_value(user_feedback: Optional[str]) -> Optional[str]:
    """Valida feedback salvo pelo Streamlit."""
    if user_feedback not in VALID_FEEDBACK_VALUES:
        raise ValueError(
            f"user_feedback inválido: {user_feedback}. "
            f"Valores permitidos: {sorted(value for value in VALID_FEEDBACK_VALUES if value)}"
        )

    return user_feedback


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Garante estrutura padrão do log.

    Este formato é consumido diretamente pelo 00_streamlit.py e pode depois
    alimentar curadoria e avaliação offline.
    """
    user_feedback = validate_feedback_value(record.get("user_feedback"))

    normalized = {
        "interaction_id": record.get("interaction_id") or generate_interaction_id(),
        "timestamp_utc": record.get("timestamp_utc") or utc_now_iso(),
        "question": record.get("question"),
        "answer": record.get("answer"),
        "index_mode": record.get("index_mode"),
        "top_k": record.get("top_k"),
        "latency_seconds": record.get("latency_seconds"),
        "bucket": record.get("bucket"),
        "region": record.get("region"),
        "embedding_model_id": record.get("embedding_model_id"),
        "llm_model_id": record.get("llm_model_id"),
        "sources": normalize_sources(record.get("sources")),
        "user_feedback": user_feedback,
        "feedback_comment": record.get("feedback_comment"),
        "needs_curation": bool(record.get("needs_curation", False)),
        "feedback_updated_at_utc": record.get("feedback_updated_at_utc"),
        "app_source": record.get("app_source", "streamlit"),
        "log_schema_version": record.get("log_schema_version", LOG_SCHEMA_VERSION),
    }

    known_fields = set(normalized.keys())

    extra_fields = {
        key: ensure_json_serializable(value)
        for key, value in record.items()
        if key not in known_fields
    }

    if extra_fields:
        normalized["extra"] = extra_fields

    return {
        key: ensure_json_serializable(value)
        for key, value in normalized.items()
    }


def build_interaction_record(
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    index_mode: str,
    top_k: int,
    latency_seconds: float,
    bucket: str,
    region: str,
    embedding_model_id: str,
    llm_model_id: str,
    user_feedback: Optional[str] = None,
    feedback_comment: Optional[str] = None,
    needs_curation: bool = False,
) -> Dict[str, Any]:
    """
    Cria registro padronizado para o Streamlit.

    O 00_streamlit.py chama esta função logo após gerar a resposta do RAG.
    """
    return normalize_record({
        "question": question,
        "answer": answer,
        "sources": sources,
        "index_mode": index_mode,
        "top_k": top_k,
        "latency_seconds": latency_seconds,
        "bucket": bucket,
        "region": region,
        "embedding_model_id": embedding_model_id,
        "llm_model_id": llm_model_id,
        "user_feedback": user_feedback,
        "feedback_comment": feedback_comment,
        "needs_curation": needs_curation,
        "app_source": "streamlit",
        "log_schema_version": LOG_SCHEMA_VERSION,
    })


def append_interaction(
    record: Dict[str, Any],
    log_file: str = DEFAULT_LOG_FILE,
) -> Dict[str, Any]:
    """
    Salva uma interação real em JSONL.

    Retorna o registro normalizado já com interaction_id e timestamp.
    """
    normalized = normalize_record(record)

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    return normalized


def load_interactions(
    log_file: str = DEFAULT_LOG_FILE,
    limit: Optional[int] = None,
    newest_first: bool = True,
) -> List[Dict[str, Any]]:
    """Lê interações do JSONL."""
    path = Path(log_file)

    if not path.exists():
        return []

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({
                    "interaction_id": f"invalid_line_{line_number}",
                    "timestamp_utc": None,
                    "error": "invalid_json_line",
                    "raw_line": line,
                })

    if newest_first:
        records = list(reversed(records))

    if limit is not None:
        return records[:limit]

    return records


def count_interactions(log_file: str = DEFAULT_LOG_FILE) -> int:
    """Conta interações registradas no JSONL."""
    path = Path(log_file)

    if not path.exists():
        return 0

    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def get_interaction_by_id(
    interaction_id: str,
    log_file: str = DEFAULT_LOG_FILE,
) -> Optional[Dict[str, Any]]:
    """Busca uma interação pelo interaction_id."""
    if not interaction_id:
        return None

    records = load_interactions(
        log_file=log_file,
        newest_first=False,
    )

    for record in records:
        if record.get("interaction_id") == interaction_id:
            return record

    return None


def overwrite_interactions(
    records: List[Dict[str, Any]],
    log_file: str = DEFAULT_LOG_FILE,
) -> None:
    """
    Reescreve o JSONL inteiro com segurança simples.

    Usado para atualizar feedback de uma interação já salva.
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_path.replace(path)


def update_interaction_feedback(
    interaction_id: str,
    user_feedback: Optional[str] = None,
    feedback_comment: Optional[str] = None,
    needs_curation: Optional[bool] = None,
    log_file: str = DEFAULT_LOG_FILE,
) -> Optional[Dict[str, Any]]:
    """
    Atualiza feedback de uma interação já registrada pelo Streamlit.

    user_feedback permitido:
    - positive
    - partial
    - negative
    - source_issue
    """
    if not interaction_id:
        return None

    validate_feedback_value(user_feedback)

    records = load_interactions(
        log_file=log_file,
        newest_first=False,
    )

    updated_record = None

    for record in records:
        if record.get("interaction_id") != interaction_id:
            continue

        if user_feedback is not None:
            record["user_feedback"] = user_feedback

        if feedback_comment is not None:
            record["feedback_comment"] = feedback_comment

        if needs_curation is not None:
            record["needs_curation"] = bool(needs_curation)

        record["feedback_updated_at_utc"] = utc_now_iso()
        record["log_schema_version"] = record.get("log_schema_version", LOG_SCHEMA_VERSION)

        updated_record = normalize_record(record)
        record.clear()
        record.update(updated_record)
        break

    if updated_record is None:
        return None

    overwrite_interactions(
        records=records,
        log_file=log_file,
    )

    return updated_record


def filter_interactions_for_curation(
    log_file: str = DEFAULT_LOG_FILE,
) -> List[Dict[str, Any]]:
    """Retorna interações marcadas para curadoria."""
    records = load_interactions(
        log_file=log_file,
        newest_first=True,
    )

    return [
        record
        for record in records
        if record.get("needs_curation") is True
    ]


def export_curation_candidates_jsonl(
    output_file: str = "curation_candidates.jsonl",
    log_file: str = DEFAULT_LOG_FILE,
) -> int:
    """Exporta interações marcadas para curadoria em JSONL."""
    candidates = filter_interactions_for_curation(log_file=log_file)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in candidates:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(candidates)


def summarize_interactions(log_file: str = DEFAULT_LOG_FILE) -> Dict[str, Any]:
    """Gera resumo simples para inspeção local."""
    records = load_interactions(
        log_file=log_file,
        newest_first=False,
    )

    feedback_counts: Dict[str, int] = {}
    index_counts: Dict[str, int] = {}
    curation_count = 0

    for record in records:
        feedback = record.get("user_feedback") or "not_rated"
        index_mode = record.get("index_mode") or "unknown"

        feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        index_counts[index_mode] = index_counts.get(index_mode, 0) + 1

        if record.get("needs_curation") is True:
            curation_count += 1

    return {
        "log_file": log_file,
        "total_interactions": len(records),
        "feedback_counts": feedback_counts,
        "index_counts": index_counts,
        "needs_curation_count": curation_count,
    }


def main() -> None:
    """
    Utilitário local do logger.

    Não cria interação fake. Serve apenas para consultar ou exportar logs reais
    gerados pelo Streamlit.
    """
    parser = argparse.ArgumentParser(
        description="Utilitário para consultar logs reais do Clinical RAG Streamlit."
    )

    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help="Arquivo JSONL de interações.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Exibe resumo do log.",
    )

    parser.add_argument(
        "--export-curation",
        default=None,
        help="Exporta interações marcadas para curadoria para o arquivo informado.",
    )

    args = parser.parse_args()

    if args.export_curation:
        total = export_curation_candidates_jsonl(
            output_file=args.export_curation,
            log_file=args.log_file,
        )
        print(f"Candidatos de curadoria exportados: {total}")
        print(args.export_curation)
        return

    if args.summary:
        print(json.dumps(
            summarize_interactions(log_file=args.log_file),
            ensure_ascii=False,
            indent=2,
        ))
        return

    print(json.dumps(
        summarize_interactions(log_file=args.log_file),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()