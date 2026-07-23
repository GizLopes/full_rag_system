# Clinical Knowledge RAG Assistant
End-to-end clinical RAG with synthetic PDF documents, Amazon S3, chunking, Amazon Bedrock Titan Embeddings,
FAISS, Claude via Bedrock, LangGraph, agents, offline evaluation, LLM as a Judge, and Streamlit.

The project evolves from a baseline RAG to more advanced techniques, including semantic chunking, query rewriting, hybrid search, RAG Fusion, Corrective RAG, ReAct Agent, retrieval evaluation, answer evaluation, real-log curation, and a comparative dashboard (baseline vs. semantic).

## Case Objective
Demonstrate a practical clinical RAG architecture using synthetic data to:
- Retrieve clinical evidence from PDF documents.
- Generate responses citing sources, pages, and chunks.
- Compare chunking and retrieval strategies.
- Evaluate quality using metrics such as Recall@K, MRR, and Precision@K.
- Evaluate responses against criteria including groundedness, correctness, citation quality, and hallucination risk.
- Log real interactions via Streamlit for future curation.

## Files
- consultas_ambulatoriais_032026
- hemograma_e_bioquimica_032026
- ressonancia_coluna
- parecer_cardiologista
- alta_hospitalar

## Details
- csv/: consolidated data by topic, with 10 patients per file.
- pdf/: simulated documents for ingestion via RAG, including reports with simple visual elements.

## Project Structure

```text
clinical_rag/
│
├── 00_ground_truth/
│   ├── csv/
│
├── 01_database/
│   └── synthetic clinical PDF documents
│
├── 02_processing/
│   ├── 00_upload_s3.py
│   ├── 01_chunking_baseline.py
│   ├── 02_chunking_semantic.py
│   ├── 03_generate_embeddings_baseline.py
│   ├── 03_generate_embeddings_semantic.py
│   └── 04_indexing.py
│
├── 03_retrieval/
│   ├── 01_baseline_rag.py
│   ├── 02_query_rewrite.py
│   ├── 03_hybrid_search.py
│   ├── 04_rag_fusion.py
│   └── 05_corrective_rag.py
│
├── 04_agentic/
│   ├── 01_langgraph_rag.py
│   └── 02_react_agent.py
│
├── 05_evaluation/
│   ├── 00_ground_truth_evaluation_dataset.py
│   ├── 01_evaluate_retrieval.py
│   ├── 02_evaluate_answer.py
│   ├── 03_llm_as_judge.py
│   ├── 04_UI_convert_interaction_logs.py
│   ├── 05_UI_curate_ground_truth_dataset.py
│   └── 06_UI_merge_evaluation_datasets.py
│
└── 06_app_ui/
    ├── 00_streamlit.py
    ├── interaction_logger.py
    ├── 02_dev_evaluation_dashboard.py
    └── interaction_logs.jsonl
```

## Synthetic Documents

Simulated clinical files used in the case:

```text
consultas_ambulatoriais_032026.pdf
hemograma_e_bioquimica_032026.pdf
ressonancia_coluna.pdf
parecer_cardiologista.pdf
alta_hospitalar.pdf
```

## Main Components

### 00_ground_truth/
Contains synthetic data consolidated by topic and simulated CSVs.

### 01_database/
Contains the synthetic clinical PDFs used as the document base for RAG ingestion.
This layer represents the controlled source of data used in training and testing.

### 02_processing/
Responsible for ingestion, chunking, embeddings, and indexing.

Main files:
```text
00_upload_s3.py
```
Uploads PDFs to the S3 bucket.

```text
01_chunking_baseline.py
```
Creates baseline chunks by page and tokens.

```text
02_chunking_semantic.py
```
Creates semantic chunks by patient, clinical section, and document structure.

```text
03_generate_embeddings_baseline.py
```
Generates Titan embeddings for baseline chunks.

```text
03_generate_embeddings_semantic.py
```
Generates Titan embeddings for semantic chunks.

```text
04_indexing.py
```
Creates baseline and semantic FAISS indices and saves artifacts locally and to S3.

## S3 Artifacts
Bucket used in the project:
```text
clinical-rag-database
```

Main prefixes:
```text
rag-database/
chunks/
chunks_semantic/
embeddings/
embeddings_semantic/
index/
index_semantic/
```

## 03_retrieval/
Contains retrieval and response generation strategies:

```text
01_baseline_rag.py
```
Executes baseline RAG with Titan Embeddings, FAISS, and Claude.

```text
02_query_rewrite.py
```
Rewrites clinical queries to improve retrieval.

```text
03_hybrid_search.py
```
Combines FAISS vector search with BM25 lexical search and Reciprocal Rank Fusion.

```text
04_rag_fusion.py
```
Generates multiple queries, executes retrieval per query, and consolidates results using RRF.

```text
05_corrective_rag.py
```
Evaluates retrieved context, rewrites the query if necessary, and executes a new retrieval.

## 04_agentic/
Contains agentic orchestration:
```text
01_langgraph_rag.py
```
Orchestrates the RAG flow with LangGraph:
```text
retrieve
evaluate
rewrite, if necessary
answer
```

Can also generate a dynamic visual graph of the answer, evidence, evaluation, and sources.
```text
02_react_agent.py
```

Executes ReAct agent with tools:
```text
retrieve_baseline
retrieve_semantic
query_rewrite
final_answer
```

## 05_evaluation/
Contains offline RAG evaluation.
```text
00_ground_truth_evaluation_dataset.py
```

Generates ground truth dataset in JSONL for offline evaluation.
Output:
```text
ground_truth_evaluation_dataset.jsonl
```

```text
01_evaluate_retrieval.py
```

Evaluates baseline versus semantic retrieval using:
```text
Recall@K
MRR
Precision@K
Hit rank
Best score
```

Main output:
```text
retrieval_eval_results.jsonl
```

```text
02_evaluate_answer.py
```

Evaluates the final RAG answer against the ground truth.
Criteria:
```text
expected terms
patient
document
citation
compatible source
```

Output:
```text
answer_eval_results.jsonl
```

```text
03_llm_as_judge.py
```

Uses Claude as a judge to evaluate:
```text
groundedness
correctness
citation quality
completeness
hallucination risk
overall score
```

Output:
```text
llm_judge_results.jsonl
```

```text
04_UI_convert_interaction_logs.py
```

Converts real Streamlit logs into candidates for curation.
Input:
```text
../06_app_ui/interaction_logs.jsonl
```

Output:
```text
curation_candidates.jsonl
```

```text
05_UI_curate_ground_truth_dataset.py
```

Transforms candidates derived from Streamlit into a curated ground truth.
Outputs:
```text
curation_review_template.jsonl
curated_ground_truth_dataset.jsonl
```

```text
06_UI_merge_evaluation_datasets.py
```

Merges the seed dataset with the real curated dataset.
Inputs:
```text
ground_truth_evaluation_dataset.jsonl
curated_ground_truth_dataset.jsonl
```

Output:
```text
full_ground_truth_evaluation_dataset.jsonl
```

## 06_app_ui/
Contains Streamlit interfaces and real interaction logging.
```text
00_streamlit.py
```
Main interface for clinical questions.
Flow:
```text
clinical question
baseline or semantic FAISS retrieval
Claude via Bedrock
response with sources
logging in interaction_logs.jsonl
user feedback
curation flagging
```

```text
interaction_logger.py
```
Module used by Streamlit to:
```text
create interaction_id
log to interaction_logs.jsonl
count interactions
update feedback
flag curation
export candidates
```

```text
02_dev_evaluation_dashboard.py
```
Development dashboard to compare baseline versus semantic.
Reads:
```text
../05_evaluation/retrieval_eval_results.jsonl
../05_evaluation/answer_eval_results.jsonl
../05_evaluation/llm_judge_results.jsonl
```

Displays:
```text
Recall@K
MRR
Precision@K
Avg best score
Hit count
Miss count
Groundedness
Correctness
Citation quality
Hallucination risk
```

## Step-by-Step Guide
### 1. Upload PDFs to S3
```powershell
cd clinical_rag/02_processing
python 00_upload_s3.py
```

### 2. Generate chunks
```powershell
python 01_chunking_baseline.py
python 02_chunking_semantic.py
```

### 3. Generate embeddings
```powershell
python 03_generate_embeddings_baseline.py
python 03_generate_embeddings_semantic.py
```

### 4. Create FAISS indices
```powershell
python 04_indexing_faiss.py
```

Or by index:

```powershell
python 04_indexing_faiss.py --only baseline
python 04_indexing_faiss.py --only semantic
```

### 5. Run baseline retrieval
```powershell
cd ../03_retrieval
python 01_baseline_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 6. Run advanced retrieval strategies
```powershell
python 02_query_rewrite.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 03_hybrid_search.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 04_rag_fusion.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 05_corrective_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 7. Run LangGraph and ReAct Agent
```powershell
cd ../04_agentic
python 01_langgraph_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 02_react_agent.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 8. Run offline evaluation
```powershell
cd ../05_evaluation
python 00_ground_truth_evaluation_dataset.py
python 01_evaluate_retrieval.py
python 02_evaluate_answer.py
python 03_llm_as_judge.py
```

### 9. Run main Streamlit app
```powershell
cd ../06_app_ui
streamlit run 00_streamlit.py
```

### 10. Run dev evaluation dashboard
```powershell
streamlit run 02_dev_evaluation_dashboard.py
```

## Workflow with Real Streamlit Logs
After using the main Streamlit interface and generating `interaction_logs.jsonl`:

```powershell
cd ../05_evaluation
python 04_UI_convert_interaction_logs.py
python 05_UI_curate_ground_truth_dataset.py
python 06_UI_merge_evaluation_datasets.py
```

Then run evaluations using the full dataset:
```powershell
python 01_evaluate_retrieval.py --eval-file full_ground_truth_evaluation_dataset.jsonl
python 02_evaluate_answer.py --eval-file full_ground_truth_evaluation_dataset.jsonl
python 03_llm_as_judge.py
```

## Test Questions
```text
Quais medicamentos o paciente P001 utiliza atualmente?
Qual foi o último resultado de creatinina da Ana Ribeiro?
O paciente P007 possui alergia registrada?
Qual foi a conclusão da ressonância da Gabriela Lima?
O parecer cardiológico do Henrique Rocha aponta risco alto?
Há indicação cirúrgica registrada para o paciente P002?
Qual foi o resultado da creatinina da paciente Gabriela Lima?
Qual foi a pressão arterial de Gabriela Lima no parecer cardiologista?
Qual foi a glicemia de Elisa Costa no exame laboratorial?
Qual medicação aparece para Fabio Oliveira na alta hospitalar?
```

## Evaluation Metrics
### Recall@K
Measures whether the expected item appeared among the top K results.
```text
Recall@K = questions with a hit in Top-K / total questions
```

### MRR
Measures how early the first relevant result appears.
```text
MRR = average of 1 / rank of the first relevant result
```

### Precision@K
Measures the proportion of relevant results within Top-K.
```text
Precision@K = relevant items in Top-K / K
```

### Groundedness
Measures whether the answer is supported by the retrieved sources.

### Correctness
Measures whether the answer is correct relative to the ground truth.

### Citation quality
Measures whether document, page, and chunk were cited correctly.

### Hallucination risk
Classifies the risk of the answer containing information unsupported by sources.

## Important Notes
- The project uses synthetic data.
- There are no real patient data.
- It must not be used for diagnosis, real clinical triage, or medical decision-making.
- The objective is to demonstrate architecture, retrieval, evaluation, and traceability.
- The dev dashboard uses JSONL files, not JSON, for primary evaluations.
- The `06_app_ui/` folder replaces the legacy `06_app/`.
