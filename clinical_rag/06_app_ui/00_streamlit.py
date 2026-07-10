# OBJETIVO PRINCIPAL
# Executar Streamlit para RAG clínico com logging integrado em tempo real via interaction_logger.py.

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
import faiss
import numpy as np
import streamlit as st
from botocore.exceptions import ClientError

from interaction_logger import (
    append_interaction,
    build_interaction_record,
    count_interactions,
    update_interaction_feedback,
)


DEFAULT_BUCKET = "clinical-rag-database-789065179500-us-east-1-an"
DEFAULT_REGION = "us-east-1"

BASELINE_INDEX_PREFIX = "index/"
BASELINE_INDEX_FILE = "clinical_faiss.index"
BASELINE_METADATA_FILE = "clinical_faiss_metadata.pkl"

SEMANTIC_INDEX_PREFIX = "index_semantic/"
SEMANTIC_INDEX_FILE = "clinical_faiss_semantic.index"
SEMANTIC_METADATA_FILE = "clinical_faiss_semantic_metadata.pkl"

DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 900
DEFAULT_TEMPERATURE = 0.0

INTERACTION_LOG_FILE = "interaction_logs.jsonl"


def configure_page() -> None:
    """Configura página Streamlit."""
    st.set_page_config(
        page_title="Clinical RAG",
        page_icon="🩺",
        layout="wide",
    )


@st.cache_resource(show_spinner=False)
def get_s3_client(region: str):
    """Cria cliente S3."""
    return boto3.client("s3", region_name=region)


@st.cache_resource(show_spinner=False)
def get_bedrock_client(region: str):
    """Cria cliente Bedrock Runtime."""
    return boto3.client("bedrock-runtime", region_name=region)


def resolve_index_config(index_mode: str) -> Tuple[str, str, str]:
    """Resolve arquivos do índice baseline ou semantic."""
    if index_mode == "baseline":
        return BASELINE_INDEX_PREFIX, BASELINE_INDEX_FILE, BASELINE_METADATA_FILE

    if index_mode == "semantic":
        return SEMANTIC_INDEX_PREFIX, SEMANTIC_INDEX_FILE, SEMANTIC_METADATA_FILE

    raise ValueError(f"index_mode inválido: {index_mode}")


def download_from_s3(
    s3_client,
    bucket: str,
    s3_key: str,
    local_file: Path,
    use_local_cache: bool = False,
) -> None:
    """
    Baixa arquivo do S3.

    Por padrão, sempre sobrescreve o arquivo local para evitar usar índice ou metadados antigos.
    """
    if use_local_cache and local_file.exists():
        return

    try:
        s3_client.download_file(
            bucket,
            s3_key,
            str(local_file),
        )
    except ClientError as exc:
        raise RuntimeError(f"Erro ao baixar s3://{bucket}/{s3_key}: {exc}") from exc


def load_faiss_artifacts(
    s3_client,
    bucket: str,
    index_mode: str,
    use_local_cache: bool,
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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza vetor para usar inner product como similaridade por cosseno."""
    norm = np.linalg.norm(vector, axis=1, keepdims=True)
    norm[norm == 0] = 1

    return vector / norm


def embed_query(
    bedrock_client,
    query: str,
    model_id: str,
) -> np.ndarray:
    """Gera embedding da query usando Titan Embeddings."""
    payload = {
        "inputText": query,
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


def search_faiss(
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    question: str,
    bedrock_client,
    embedding_model_id: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Executa busca FAISS."""
    query_vector = embed_query(
        bedrock_client=bedrock_client,
        query=question,
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


def compact_source(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduz fonte recuperada para UI e logs."""
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
        "text_preview": get_text(result)[:1200],
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


def invoke_claude_answer(
    bedrock_client,
    model_id: str,
    question: str,
    context: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Gera resposta final com Claude usando somente contexto recuperado."""
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
6. Não dê aconselhamento médico. Este é um dataset sintético para demonstração de RAG.

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


def run_rag(
    question: str,
    bucket: str,
    region: str,
    index_mode: str,
    top_k: int,
    embedding_model_id: str,
    llm_model_id: str,
    max_tokens: int,
    temperature: float,
    use_local_cache: bool,
) -> Dict[str, Any]:
    """
    Executa pipeline RAG completo e registra a interação real via interaction_logger.py.
    """
    start_time = time.time()

    s3_client = get_s3_client(region)
    bedrock_client = get_bedrock_client(region)

    index, metadata = load_faiss_artifacts(
        s3_client=s3_client,
        bucket=bucket,
        index_mode=index_mode,
        use_local_cache=use_local_cache,
    )

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

    latency_seconds = round(time.time() - start_time, 4)

    sources = [
        compact_source(result)
        for result in retrieved_results
    ]

    interaction_record = build_interaction_record(
        question=question,
        answer=answer,
        sources=sources,
        index_mode=index_mode,
        top_k=top_k,
        latency_seconds=latency_seconds,
        bucket=bucket,
        region=region,
        embedding_model_id=embedding_model_id,
        llm_model_id=llm_model_id,
    )

    saved_record = append_interaction(
        record=interaction_record,
        log_file=INTERACTION_LOG_FILE,
    )

    return saved_record


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Renderiza fontes no Streamlit."""
    if not sources:
        st.warning("Nenhuma fonte recuperada.")
        return

    for source in sources:
        if isinstance(source.get("score"), float):
            title = (
                f"Fonte {source.get('rank')} | "
                f"{source.get('document_name')} | "
                f"score {source.get('score'):.4f}"
            )
        else:
            title = f"Fonte {source.get('rank')} | {source.get('document_name')}"

        with st.expander(title, expanded=source.get("rank") == 1):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Documento**")
                st.write(source.get("document_name"))
                st.markdown("**Tipo**")
                st.write(source.get("document_type"))

            with col2:
                st.markdown("**Paciente**")
                st.write(source.get("patient_name"))
                st.markdown("**Patient ID**")
                st.write(source.get("patient_id"))

            with col3:
                st.markdown("**Página**")
                st.write(f"{source.get('page_start')} - {source.get('page_end')}")
                st.markdown("**Chunk**")
                st.write(f"{source.get('chunk_number')} / {source.get('total_chunks')}")

            st.markdown("**Prévia do chunk**")
            st.text_area(
                label=f"chunk_preview_{source.get('rank')}",
                value=source.get("text_preview") or "",
                height=180,
                label_visibility="collapsed",
                disabled=True,
            )

            st.markdown("**S3 URI**")
            st.code(source.get("s3_uri") or "N/A")


def render_feedback_panel(result: Dict[str, Any]) -> None:
    """Renderiza feedback em tempo real e atualiza interaction_logs.jsonl."""
    interaction_id = result.get("interaction_id")

    if not interaction_id:
        st.warning("Interação sem interaction_id. Feedback não pode ser registrado.")
        return

    st.subheader("Feedback da resposta")

    feedback_options = {
        "Correta": "positive",
        "Parcial": "partial",
        "Incorreta": "negative",
        "Fonte incorreta": "source_issue",
    }

    feedback_label = st.radio(
        "Como você avalia essa resposta?",
        options=list(feedback_options.keys()),
        horizontal=True,
        key=f"feedback_label_{interaction_id}",
    )

    feedback_comment = st.text_area(
        "Comentário opcional para curadoria",
        key=f"feedback_comment_{interaction_id}",
        placeholder="Exemplo: recuperou o paciente certo, mas citou o documento errado...",
        height=90,
    )

    needs_curation = st.checkbox(
        "Marcar para curadoria",
        value=feedback_label in ["Parcial", "Incorreta", "Fonte incorreta"],
        key=f"needs_curation_{interaction_id}",
    )

    if st.button(
        "Salvar feedback",
        key=f"save_feedback_{interaction_id}",
        type="secondary",
    ):
        updated = update_interaction_feedback(
            interaction_id=interaction_id,
            user_feedback=feedback_options[feedback_label],
            feedback_comment=feedback_comment,
            needs_curation=needs_curation,
            log_file=INTERACTION_LOG_FILE,
        )

        if updated:
            st.session_state["last_result"] = updated
            st.success("Feedback registrado no interaction_logs.jsonl.")
        else:
            st.error("Não encontrei a interação para atualizar o feedback.")


def render_sidebar() -> Dict[str, Any]:
    """Renderiza configurações laterais."""
    st.sidebar.header("Configurações")

    bucket = st.sidebar.text_input(
        "Bucket S3",
        value=DEFAULT_BUCKET,
    )

    region = st.sidebar.text_input(
        "Região AWS",
        value=DEFAULT_REGION,
    )

    index_mode = st.sidebar.selectbox(
        "Índice FAISS",
        options=["baseline", "semantic"],
        index=0,
    )

    top_k = st.sidebar.slider(
        "Top-K",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    max_tokens = st.sidebar.slider(
        "Max tokens",
        min_value=300,
        max_value=2000,
        value=DEFAULT_MAX_TOKENS,
        step=100,
    )

    temperature = st.sidebar.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.1,
    )

    use_local_cache = st.sidebar.checkbox(
        "Usar cache local dos índices",
        value=False,
        help="Desmarcado por padrão para sempre baixar artefatos atualizados do S3.",
    )

    with st.sidebar.expander("Modelos Bedrock"):
        embedding_model_id = st.text_input(
            "Embedding model",
            value=DEFAULT_EMBEDDING_MODEL_ID,
        )

        llm_model_id = st.text_area(
            "LLM model ou inference profile ARN",
            value=DEFAULT_LLM_MODEL_ID,
            height=120,
        )

    st.sidebar.divider()
    st.sidebar.metric(
        "Interações registradas",
        count_interactions(INTERACTION_LOG_FILE),
    )

    st.sidebar.caption(
        "O log é gravado em tempo real via interaction_logger.py."
    )

    return {
        "bucket": bucket,
        "region": region,
        "index_mode": index_mode,
        "top_k": top_k,
        "embedding_model_id": embedding_model_id,
        "llm_model_id": llm_model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_local_cache": use_local_cache,
    }


def clear_result_and_question() -> None:
    """Limpa resultado exibido e caixa de pergunta."""
    st.session_state.pop("last_result", None)
    st.session_state["question_input"] = ""


def render_app() -> None:
    """Renderiza aplicação Streamlit."""
    configure_page()

    st.title("🩺 Clinical RAG")
    st.caption(
        "Consulta documentos clínicos sintéticos via Bedrock."
    )

    settings = render_sidebar()

    default_question = "Qual foi o resultado do/a paciente INSIRA NOME O COMPLETO?"

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = default_question

    question = st.text_area(
        "Pergunta clínica",
        key="question_input",
        height=110,
        placeholder="Digite uma pergunta sobre os documentos clínicos sintéticos...",
    )

    col_run, col_clear = st.columns([1, 4])

    with col_run:
        run_clicked = st.button(
            "Executar RAG",
            type="primary",
            use_container_width=True,
        )

    with col_clear:
        st.button(
            "Limpar resultado",
            use_container_width=False,
            on_click=clear_result_and_question,
        )

    if run_clicked:
        clean_question = question.strip()

        if not clean_question:
            st.error("Digite uma pergunta antes de executar.")
            return

        with st.spinner("Executando retrieval, geração de resposta e logging em tempo real..."):
            try:
                result = run_rag(
                    question=clean_question,
                    bucket=settings["bucket"],
                    region=settings["region"],
                    index_mode=settings["index_mode"],
                    top_k=settings["top_k"],
                    embedding_model_id=settings["embedding_model_id"],
                    llm_model_id=settings["llm_model_id"],
                    max_tokens=settings["max_tokens"],
                    temperature=settings["temperature"],
                    use_local_cache=settings["use_local_cache"],
                )
                st.session_state["last_result"] = result
                st.success(
                    f"RAG executado e interação registrada. ID: {result.get('interaction_id')}"
                )

            except Exception as exc:
                st.error(f"Erro ao executar RAG: {exc}")
                return

    result = st.session_state.get("last_result")

    if not result:
        st.info("Digite uma pergunta e clique em Executar RAG.")
        return

    st.divider()

    st.subheader("Resposta")
    st.markdown(result.get("answer") or "Sem resposta gerada.")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Índice", result.get("index_mode"))

    with metric_col2:
        st.metric("Top-K", result.get("top_k"))

    with metric_col3:
        st.metric("Latência", f"{result.get('latency_seconds')}s")

    with metric_col4:
        st.metric("Interaction ID", str(result.get("interaction_id", ""))[:8])

    st.subheader("Fontes recuperadas")
    render_sources(result.get("sources") or [])

    st.divider()
    render_feedback_panel(result)

    with st.expander("Registro JSON da interação"):
        st.json(result)


if __name__ == "__main__":
    render_app()