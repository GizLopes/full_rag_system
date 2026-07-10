# OBJETIVO PRINCIPAL
# Exibir dashboard Streamlit dev para comparar baseline vs semantic usando arquivos JSONL de avaliação.

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


DEFAULT_RETRIEVAL_JSONL = "../05_evaluation/retrieval_eval_results.jsonl"
DEFAULT_ANSWER_JSONL = "../05_evaluation/answer_eval_results.jsonl"
DEFAULT_JUDGE_JSONL = "../05_evaluation/llm_judge_results.jsonl"


METRIC_LABELS = {
    "recall_at_k": "Recall@K",
    "mrr": "MRR",
    "precision_at_k": "Precision@K",
    "avg_best_score": "Avg best score",
    "hit_count": "Hits",
    "miss_count": "Misses",
    "total_questions": "Questions",
}


TWO_DECIMAL_COLUMNS = {
    "recall_at_k": st.column_config.NumberColumn("Recall@K", format="%.2f"),
    "mrr": st.column_config.NumberColumn("MRR", format="%.2f"),
    "precision_at_k": st.column_config.NumberColumn("Precision@K", format="%.2f"),
    "best_score": st.column_config.NumberColumn("Best score", format="%.2f"),
    "avg_best_score": st.column_config.NumberColumn("Avg best score", format="%.2f"),
    "baseline": st.column_config.NumberColumn("Baseline", format="%.2f"),
    "semantic": st.column_config.NumberColumn("Semantic", format="%.2f"),
    "delta_semantic_minus_baseline": st.column_config.NumberColumn("Delta absoluto", format="%.2f"),
    "delta_percent_semantic_minus_baseline": st.column_config.NumberColumn("Delta percentual", format="%.2f"),
    "score": st.column_config.NumberColumn("Score", format="%.2f"),
    "avg_score": st.column_config.NumberColumn("Avg score", format="%.2f"),
    "pass_rate": st.column_config.NumberColumn("Pass rate", format="%.2f"),
    "overall_score": st.column_config.NumberColumn("Overall score", format="%.2f"),
    "groundedness_score": st.column_config.NumberColumn("Groundedness", format="%.2f"),
    "correctness_score": st.column_config.NumberColumn("Correctness", format="%.2f"),
    "citation_quality_score": st.column_config.NumberColumn("Citation quality", format="%.2f"),
    "completeness_score": st.column_config.NumberColumn("Completeness", format="%.2f"),
    "avg_overall_score": st.column_config.NumberColumn("Avg overall", format="%.2f"),
    "avg_groundedness": st.column_config.NumberColumn("Avg groundedness", format="%.2f"),
    "avg_correctness": st.column_config.NumberColumn("Avg correctness", format="%.2f"),
    "avg_citation_quality": st.column_config.NumberColumn("Avg citation quality", format="%.2f"),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Clinical RAG Dev Evaluation",
        page_icon="📊",
        layout="wide",
    )


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .dev-metric-card {
            background: transparent;
            border-radius: 14px;
            padding: 0.2rem 0.1rem 0.2rem 0.1rem;
            min-height: 150px;
        }
        .dev-metric-label {
            font-size: 0.95rem;
            color: #444;
            margin-bottom: 0.4rem;
        }
        .dev-metric-value {
            font-size: 3rem;
            font-weight: 700;
            color: #2f3340;
            line-height: 1.0;
            margin-bottom: 0.8rem;
        }
        .dev-pill-row {
            display: flex;
            gap: 0.55rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .dev-pill-green {
            display: inline-block;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            background: #dff3e6;
            color: #16803c;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .dev-pill-cyan {
            display: inline-block;
            padding: 0.32rem 0.75rem;
            border-radius: 999px;
            background: #dff4ff;
            color: #0b67a3;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .dev-metric-baseline {
            font-size: 1rem;
            color: #5b6470;
            margin-top: 0.55rem;
        }
        .formula-block {
            background: #f8f9fb;
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
            border: 1px solid #e6e9ef;
        }
        .judge-note {
            background: #f0fbff;
            border: 1px solid #bee8ff;
            color: #17435f;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_jsonl_file(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        return []

    records = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({
                    "error": "invalid_json_line",
                    "line_number": line_number,
                    "raw_line": line,
                })

    return records


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def round_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rounded = df.copy()

    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].round(2)

    return rounded


def get_numeric_column_config(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        column: config
        for column, config in TWO_DECIMAL_COLUMNS.items()
        if column in df.columns
    }


def extract_retrieval_rows(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for record in records:
        if record.get("error"):
            continue

        expected = record.get("expected") or {}
        hit_result = record.get("hit_result") or {}

        rows.append({
            "index_mode": record.get("index_mode"),
            "question": record.get("question"),
            "hit": bool(record.get("hit")),
            "hit_rank": record.get("hit_rank"),
            "recall_at_k": safe_float(record.get("recall_at_k")),
            "mrr": safe_float(record.get("mrr")),
            "precision_at_k": safe_float(record.get("precision_at_k")),
            "best_score": safe_float(record.get("best_score")),
            "expected_document_name": expected.get("expected_document_name"),
            "expected_document_type": expected.get("expected_document_type"),
            "expected_patient_id": expected.get("expected_patient_id"),
            "expected_patient_name": expected.get("expected_patient_name"),
            "expected_terms": " | ".join(expected.get("expected_terms") or []),
            "hit_document_name": hit_result.get("document_name"),
            "hit_document_type": hit_result.get("document_type"),
            "hit_patient_id": hit_result.get("patient_id"),
            "hit_patient_name": hit_result.get("patient_name"),
            "hit_page_start": hit_result.get("page_start"),
            "hit_page_end": hit_result.get("page_end"),
            "hit_chunk_number": hit_result.get("chunk_number"),
            "rank_checks": record.get("rank_checks"),
            "retrieved": record.get("retrieved"),
        })

    return pd.DataFrame(rows)


def build_summary_df(question_df: pd.DataFrame) -> pd.DataFrame:
    if question_df.empty:
        return pd.DataFrame()

    summaries = []

    for index_mode, group in question_df.groupby("index_mode", dropna=False):
        total_questions = len(group)
        hit_count = int(group["hit"].sum())
        miss_count = total_questions - hit_count

        summaries.append({
            "index_mode": index_mode,
            "total_questions": total_questions,
            "hit_count": hit_count,
            "miss_count": miss_count,
            "recall_at_k": round(group["recall_at_k"].fillna(0).mean(), 2),
            "mrr": round(group["mrr"].fillna(0).mean(), 2),
            "precision_at_k": round(group["precision_at_k"].fillna(0).mean(), 2),
            "avg_best_score": round(group["best_score"].fillna(0).mean(), 2),
        })

    return pd.DataFrame(summaries)


def get_summary_value(summary_df: pd.DataFrame, index_mode: str, metric: str) -> Any:
    if summary_df.empty:
        return None

    filtered = summary_df[summary_df["index_mode"] == index_mode]

    if filtered.empty:
        return None

    value = filtered.iloc[0].get(metric)

    if pd.isna(value):
        return None

    return value


def calculate_delta(baseline_value: Any, semantic_value: Any) -> Optional[float]:
    try:
        if baseline_value is None or semantic_value is None:
            return None

        return round(float(semantic_value) - float(baseline_value), 2)
    except (TypeError, ValueError):
        return None


def calculate_delta_percent(baseline_value: Any, semantic_value: Any) -> Optional[float]:
    try:
        if baseline_value is None or semantic_value is None:
            return None

        baseline_value = float(baseline_value)
        semantic_value = float(semantic_value)

        if baseline_value == 0:
            return None

        return round(((semantic_value - baseline_value) / baseline_value) * 100.0, 2)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def format_metric_value(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_delta_value(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        value = float(value)
        arrow = "↑" if value >= 0 else "↓"
        return f"{arrow} {abs(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent_delta(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        value = float(value)
        arrow = "↑" if value >= 0 else "↓"
        return f"{arrow} {abs(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def build_metric_comparison_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in summary_df.iterrows():
        for metric in ["recall_at_k", "mrr", "precision_at_k", "avg_best_score"]:
            rows.append({
                "metric": METRIC_LABELS.get(metric, metric),
                "index_mode": row.get("index_mode"),
                "value": round(float(row.get(metric) or 0), 2),
            })

    return pd.DataFrame(rows)


def build_side_by_side_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []

    for metric in [
        "recall_at_k",
        "mrr",
        "precision_at_k",
        "avg_best_score",
        "hit_count",
        "miss_count",
        "total_questions",
    ]:
        baseline_value = get_summary_value(summary_df, "baseline", metric)
        semantic_value = get_summary_value(summary_df, "semantic", metric)

        rows.append({
            "metric": METRIC_LABELS.get(metric, metric),
            "baseline": baseline_value,
            "semantic": semantic_value,
            "delta_semantic_minus_baseline": calculate_delta(
                baseline_value,
                semantic_value,
            ),
            "delta_percent_semantic_minus_baseline": calculate_delta_percent(
                baseline_value,
                semantic_value,
            ),
        })

    return round_numeric_columns(pd.DataFrame(rows))


def render_header() -> None:
    st.title("Clinical RAG Dev Evaluation")
    st.caption(
        "Dashboard de desenvolvimento para comparar baseline vs semantic usando retrieval_eval_results.jsonl."
    )


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Arquivos de avaliação")

    retrieval_path = st.sidebar.text_input(
        "retrieval_eval_results.jsonl",
        value=DEFAULT_RETRIEVAL_JSONL,
    )

    answer_path = st.sidebar.text_input(
        "answer_eval_results.jsonl",
        value=DEFAULT_ANSWER_JSONL,
    )

    judge_path = st.sidebar.text_input(
        "llm_judge_results.jsonl",
        value=DEFAULT_JUDGE_JSONL,
    )

    st.sidebar.divider()

    refresh = st.sidebar.button(
        "Recarregar arquivos",
        use_container_width=True,
    )

    st.sidebar.caption(
        "Este app deve ficar em 06_app_ui/ e lê resultados de 05_evaluation/."
    )

    return {
        "retrieval_path": retrieval_path,
        "answer_path": answer_path,
        "judge_path": judge_path,
        "refresh": refresh,
    }


def render_custom_metric_card(
    label: str,
    current_value: Any,
    baseline_value: Any,
    absolute_delta: Any,
    percent_delta: Any,
) -> None:
    current_value_text = format_metric_value(current_value)
    absolute_delta_text = format_delta_value(absolute_delta)
    percent_delta_text = format_percent_delta(percent_delta)
    baseline_text = format_metric_value(baseline_value)

    html = f"""
    <div class="dev-metric-card">
        <div class="dev-metric-label">{label}</div>
        <div class="dev-metric-value">{current_value_text}</div>
        <div class="dev-pill-row">
            <span class="dev-pill-green">{absolute_delta_text}</span>
            <span class="dev-pill-cyan">{percent_delta_text}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_metric_cards(summary_df: pd.DataFrame) -> None:
    st.subheader("Resumo executivo")

    if summary_df.empty:
        st.warning("Nenhum resumo encontrado.")
        return

    baseline_recall = get_summary_value(summary_df, "baseline", "recall_at_k")
    semantic_recall = get_summary_value(summary_df, "semantic", "recall_at_k")
    baseline_mrr = get_summary_value(summary_df, "baseline", "mrr")
    semantic_mrr = get_summary_value(summary_df, "semantic", "mrr")
    baseline_precision = get_summary_value(summary_df, "baseline", "precision_at_k")
    semantic_precision = get_summary_value(summary_df, "semantic", "precision_at_k")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_custom_metric_card(
            label="Semantic Recall@K",
            current_value=semantic_recall,
            baseline_value=baseline_recall,
            absolute_delta=calculate_delta(baseline_recall, semantic_recall),
            percent_delta=calculate_delta_percent(baseline_recall, semantic_recall),
        )

    with col2:
        render_custom_metric_card(
            label="Semantic MRR",
            current_value=semantic_mrr,
            baseline_value=baseline_mrr,
            absolute_delta=calculate_delta(baseline_mrr, semantic_mrr),
            percent_delta=calculate_delta_percent(baseline_mrr, semantic_mrr),
        )

    with col3:
        render_custom_metric_card(
            label="Semantic Precision@K",
            current_value=semantic_precision,
            baseline_value=baseline_precision,
            absolute_delta=calculate_delta(baseline_precision, semantic_precision),
            percent_delta=calculate_delta_percent(baseline_precision, semantic_precision),
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Baseline Recall@K", format_metric_value(baseline_recall))

    with col5:
        st.metric("Baseline MRR", format_metric_value(baseline_mrr))

    with col6:
        st.metric("Baseline Precision@K", format_metric_value(baseline_precision))


def render_bar_chart(summary_df: pd.DataFrame) -> None:
    st.subheader("Comparativo baseline vs semantic")

    comparison_df = build_metric_comparison_df(summary_df)

    if comparison_df.empty:
        st.warning("Sem dados suficientes para gráfico.")
        return

    pivot = comparison_df.pivot(
        index="metric",
        columns="index_mode",
        values="value",
    ).round(2)

    st.bar_chart(pivot, use_container_width=True)


def render_side_by_side_table(summary_df: pd.DataFrame) -> None:
    with st.expander("Tabela comparativa de métricas", expanded=True):
        table = build_side_by_side_metrics(summary_df)

        if table.empty:
            st.warning("Sem métricas para exibir.")
            return

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config=get_numeric_column_config(table),
        )


def render_question_detail(filtered_df: pd.DataFrame) -> None:
    if filtered_df.empty:
        return

    st.markdown("#### Detalhe da pergunta")

    question_options = [
        f"{row['index_mode']} | {'HIT' if row['hit'] else 'MISS'} | {row['question']}"
        for _, row in filtered_df.iterrows()
    ]

    selected_label = st.selectbox(
        "Selecione uma linha para inspecionar",
        options=question_options,
    )

    selected_index = question_options.index(selected_label)
    selected_row = filtered_df.iloc[selected_index].to_dict()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Esperado**")
        st.json({
            "expected_patient_id": selected_row.get("expected_patient_id"),
            "expected_patient_name": selected_row.get("expected_patient_name"),
            "expected_document_name": selected_row.get("expected_document_name"),
            "expected_document_type": selected_row.get("expected_document_type"),
            "expected_terms": selected_row.get("expected_terms"),
        })

    with col2:
        st.markdown("**Hit encontrado**")
        st.json({
            "hit": selected_row.get("hit"),
            "hit_rank": selected_row.get("hit_rank"),
            "hit_document_name": selected_row.get("hit_document_name"),
            "hit_document_type": selected_row.get("hit_document_type"),
            "hit_patient_id": selected_row.get("hit_patient_id"),
            "hit_patient_name": selected_row.get("hit_patient_name"),
            "hit_page_start": selected_row.get("hit_page_start"),
            "hit_page_end": selected_row.get("hit_page_end"),
            "hit_chunk_number": selected_row.get("hit_chunk_number"),
        })

    with st.expander("Rank checks"):
        st.json(selected_row.get("rank_checks"))

    with st.expander("Chunks recuperados"):
        st.json(selected_row.get("retrieved"))


def render_question_comparison(question_df: pd.DataFrame) -> None:
    st.subheader("Análise por pergunta")

    if question_df.empty:
        st.warning("Nenhum detalhe por pergunta encontrado.")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        available_indexes = sorted(question_df["index_mode"].dropna().unique())
        selected_index = st.multiselect(
            "Índice",
            options=available_indexes,
            default=available_indexes,
        )

    with col2:
        hit_filter = st.selectbox(
            "Resultado",
            options=["Todos", "Hits", "Misses"],
            index=0,
        )

    with col3:
        patient_filter = st.text_input(
            "Filtro por paciente ou pergunta",
            value="",
        )

    filtered = question_df.copy()

    if selected_index:
        filtered = filtered[filtered["index_mode"].isin(selected_index)]

    if hit_filter == "Hits":
        filtered = filtered[filtered["hit"] == True]

    if hit_filter == "Misses":
        filtered = filtered[filtered["hit"] == False]

    if patient_filter.strip():
        needle = patient_filter.strip().lower()
        filtered = filtered[
            filtered["question"].fillna("").str.lower().str.contains(needle, regex=False)
            | filtered["expected_patient_name"].fillna("").str.lower().str.contains(needle, regex=False)
            | filtered["expected_patient_id"].fillna("").str.lower().str.contains(needle, regex=False)
        ]

    display_columns = [
        "index_mode",
        "hit",
        "hit_rank",
        "recall_at_k",
        "mrr",
        "precision_at_k",
        "best_score",
        "question",
        "expected_patient_name",
        "expected_document_name",
        "hit_document_name",
        "hit_page_start",
        "hit_page_end",
    ]

    display_df = round_numeric_columns(filtered[display_columns])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=get_numeric_column_config(display_df),
    )

    render_question_detail(filtered)


def render_answer_eval(answer_records: List[Dict[str, Any]]) -> None:
    st.subheader("Answer evaluation")

    if not answer_records:
        st.info("answer_eval_results.jsonl ainda não encontrado ou vazio.")
        return

    rows = []

    for record in answer_records:
        evaluation = record.get("evaluation") or {}
        components = evaluation.get("score_components") or {}

        rows.append({
            "index_mode": record.get("index_mode"),
            "question": record.get("question"),
            "passed": bool(evaluation.get("passed")),
            "score": safe_float(evaluation.get("score")),
            "terms": components.get("answer_contains_expected_terms"),
            "patient": components.get("answer_mentions_patient"),
            "document": components.get("answer_mentions_document"),
            "citation": components.get("answer_has_citation_signal"),
            "source_match": components.get("source_match"),
            "failure_reasons": " | ".join(evaluation.get("failure_reasons") or []),
        })

    df = round_numeric_columns(pd.DataFrame(rows))

    if df.empty:
        st.info("Sem registros de answer evaluation.")
        return

    summary = (
        df.groupby("index_mode", dropna=False)
        .agg(
            total=("question", "count"),
            passed=("passed", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    summary["pass_rate"] = summary["passed"] / summary["total"]
    summary = round_numeric_columns(summary)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config=get_numeric_column_config(summary),
    )

    chart_df = summary.set_index("index_mode")[["pass_rate", "avg_score"]].round(2)
    st.bar_chart(chart_df, use_container_width=True)

    with st.expander("Detalhe por resposta"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=get_numeric_column_config(df),
        )


def render_llm_judge(judge_records: List[Dict[str, Any]]) -> None:
    st.subheader("LLM as Judge")

    if not judge_records:
        st.info("llm_judge_results.jsonl ainda não encontrado ou vazio.")
        return

    rows = []

    for record in judge_records:
        judge = record.get("llm_judge") or {}

        rows.append({
            "index_mode": record.get("index_mode"),
            "question": record.get("question"),
            "pass": bool(judge.get("pass")),
            "overall_score": safe_float(judge.get("overall_score")),
            "groundedness_score": safe_float(judge.get("groundedness_score")),
            "correctness_score": safe_float(judge.get("correctness_score")),
            "citation_quality_score": safe_float(judge.get("citation_quality_score")),
            "completeness_score": safe_float(judge.get("completeness_score")),
            "hallucination_risk": judge.get("hallucination_risk"),
            "issues": " | ".join(judge.get("issues") or []),
        })

    df = round_numeric_columns(pd.DataFrame(rows))

    if df.empty:
        st.info("Sem registros de LLM Judge.")
        return

    summary = (
        df.groupby("index_mode", dropna=False)
        .agg(
            total=("question", "count"),
            passed=("pass", "sum"),
            avg_overall_score=("overall_score", "mean"),
            avg_groundedness=("groundedness_score", "mean"),
            avg_correctness=("correctness_score", "mean"),
            avg_citation_quality=("citation_quality_score", "mean"),
        )
        .reset_index()
    )
    summary["pass_rate"] = summary["passed"] / summary["total"]
    summary = round_numeric_columns(summary)

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config=get_numeric_column_config(summary),
    )

    render_llm_judge_quality_section(df, summary)

    with st.expander("Detalhe LLM Judge"):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=get_numeric_column_config(df),
        )


def render_llm_judge_quality_section(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.markdown("#### Groundedness, correctness, citation quality e risco de alucinação")

    st.markdown(
        """
        <div class="judge-note">
        Groundedness mede se a resposta está sustentada pelas fontes. Correctness mede aderência ao ground truth.
        Citation quality mede qualidade de citação de documento, página e chunk. Hallucination risk classifica o risco de resposta não suportada.
        </div>
        """,
        unsafe_allow_html=True,
    )

    baseline_groundedness = get_summary_value(summary, "baseline", "avg_groundedness")
    semantic_groundedness = get_summary_value(summary, "semantic", "avg_groundedness")
    baseline_correctness = get_summary_value(summary, "baseline", "avg_correctness")
    semantic_correctness = get_summary_value(summary, "semantic", "avg_correctness")
    baseline_citation = get_summary_value(summary, "baseline", "avg_citation_quality")
    semantic_citation = get_summary_value(summary, "semantic", "avg_citation_quality")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_custom_metric_card(
            label="Semantic groundedness",
            current_value=semantic_groundedness,
            baseline_value=baseline_groundedness,
            absolute_delta=calculate_delta(baseline_groundedness, semantic_groundedness),
            percent_delta=calculate_delta_percent(baseline_groundedness, semantic_groundedness),
        )

    with col2:
        render_custom_metric_card(
            label="Semantic correctness",
            current_value=semantic_correctness,
            baseline_value=baseline_correctness,
            absolute_delta=calculate_delta(baseline_correctness, semantic_correctness),
            percent_delta=calculate_delta_percent(baseline_correctness, semantic_correctness),
        )

    with col3:
        render_custom_metric_card(
            label="Semantic citation quality",
            current_value=semantic_citation,
            baseline_value=baseline_citation,
            absolute_delta=calculate_delta(baseline_citation, semantic_citation),
            percent_delta=calculate_delta_percent(baseline_citation, semantic_citation),
        )

    chart_df = summary.set_index("index_mode")[
        [
            "avg_overall_score",
            "avg_groundedness",
            "avg_correctness",
            "avg_citation_quality",
        ]
    ].round(2)

    st.bar_chart(chart_df, use_container_width=True)

    risk_counts = (
        df.groupby(["index_mode", "hallucination_risk"], dropna=False)
        .size()
        .reset_index(name="count")
    )

    if not risk_counts.empty:
        st.markdown("#### Risco de alucinação")
        risk_pivot = risk_counts.pivot(
            index="hallucination_risk",
            columns="index_mode",
            values="count",
        ).fillna(0)
        st.bar_chart(risk_pivot, use_container_width=True)

        st.dataframe(
            risk_counts,
            use_container_width=True,
            hide_index=True,
        )


def render_formula_reference() -> None:
    st.divider()
    st.subheader("Fórmulas de referência")
    st.caption("Cálculos usados para as métricas exibidas neste dashboard.")

    st.markdown(
        """
        <div class="formula-block">
        <b>Recall@K</b><br>
        Mede se o retrieval encontrou pelo menos uma evidência relevante entre os K primeiros resultados.<br><br>
        Para uma pergunta q:<br>
        Recall@K(q) = 1, se existe pelo menos um item relevante no Top-K; caso contrário, 0.<br><br>
        Agregado:<br>
        Recall@K = soma dos acertos no Top-K / número total de perguntas.
        </div>

        <div class="formula-block">
        <b>MRR</b><br>
        Mede quão cedo o primeiro resultado relevante aparece no ranking.<br><br>
        Para uma pergunta q:<br>
        Reciprocal Rank(q) = 1 / rank do primeiro item relevante.<br><br>
        Agregado:<br>
        MRR = média dos reciprocal ranks de todas as perguntas.<br>
        Se não houver item relevante, a contribuição da pergunta é 0.
        </div>

        <div class="formula-block">
        <b>Precision@K</b><br>
        Mede a concentração de resultados relevantes dentro do Top-K.<br><br>
        Para uma pergunta q:<br>
        Precision@K(q) = número de itens relevantes no Top-K / K.<br><br>
        Agregado:<br>
        Precision@K = média da Precision@K de todas as perguntas.
        </div>

        <div class="formula-block">
        <b>Avg best score</b><br>
        Média do maior score de similaridade recuperado em cada pergunta.<br><br>
        Avg best score = soma dos best scores / número total de perguntas.
        </div>

        <div class="formula-block">
        <b>Hit count e Miss count</b><br>
        Hit count = número de perguntas com pelo menos um acerto no Top-K.<br>
        Miss count = número de perguntas sem acerto no Top-K.<br>
        Total questions = Hit count + Miss count.
        </div>

        <div class="formula-block">
        <b>Delta absoluto e delta percentual</b><br>
        Delta absoluto = métrica semantic − métrica baseline.<br><br>
        Delta percentual = ((métrica semantic − métrica baseline) / métrica baseline) × 100.
        </div>

        <div class="formula-block">
        <b>LLM as Judge</b><br>
        Groundedness = nota de 0 a 5 para suporte da resposta nas fontes recuperadas.<br>
        Correctness = nota de 0 a 5 para aderência da resposta ao ground truth esperado.<br>
        Citation quality = nota de 0 a 5 para qualidade da citação de documento, página e chunk.<br>
        Hallucination risk = classificação low, medium ou high para risco de informação não suportada pelas fontes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Notação matemática"):
        st.latex(r"Recall@K = \frac{\sum_{q=1}^{N} hit_q}{N}")
        st.latex(r"MRR = \frac{1}{N} \sum_{q=1}^{N} \frac{1}{rank_q}")
        st.latex(r"Precision@K = \frac{1}{N} \sum_{q=1}^{N} \frac{relevant\_items@K_q}{K}")
        st.latex(r"AvgBestScore = \frac{1}{N} \sum_{q=1}^{N} best\_score_q")
        st.latex(r"\Delta_{abs} = metric_{semantic} - metric_{baseline}")
        st.latex(r"\Delta_{\%} = \frac{metric_{semantic} - metric_{baseline}}{metric_{baseline}} \times 100")


def render_raw_records(records: List[Dict[str, Any]]) -> None:
    with st.expander("JSONL bruto do retrieval_eval_results.jsonl"):
        st.json(records)


def render_app() -> None:
    configure_page()
    inject_custom_css()
    render_header()

    settings = render_sidebar()

    if settings["refresh"]:
        st.cache_data.clear()
        st.rerun()

    retrieval_records = load_jsonl_file(settings["retrieval_path"])

    if not retrieval_records:
        st.error(
            "retrieval_eval_results.jsonl não encontrado ou vazio. "
            "Rode primeiro: python 01_evaluate_retrieval.py"
        )
        st.code(settings["retrieval_path"])
        return

    question_df = extract_retrieval_rows(retrieval_records)
    summary_df = build_summary_df(question_df)

    render_metric_cards(summary_df)
    render_bar_chart(summary_df)
    render_side_by_side_table(summary_df)

    st.divider()
    render_question_comparison(question_df)

    st.divider()
    answer_records = load_jsonl_file(settings["answer_path"])
    render_answer_eval(answer_records)

    st.divider()
    judge_records = load_jsonl_file(settings["judge_path"])
    render_llm_judge(judge_records)

    render_formula_reference()
    render_raw_records(retrieval_records)


if __name__ == "__main__":
    render_app()