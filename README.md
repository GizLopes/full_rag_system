# Clinical Knowledge RAG Assistant
RAG clínico end-to-end com documentos sintéticos em PDF, Amazon S3, chunking, Amazon Bedrock Titan Embeddings,
FAISS, Claude via Bedrock, LangGraph, agentes, avaliação offline, LLM as a Judge e Streamlit.

O projeto evolui de um RAG baseline para técnicas mais avançadas, incluindo chunking semântico, query rewriting, hybrid search,
RAG Fusion, Corrective RAG, ReAct Agent, avaliação de retrieval, avaliação de resposta, curadoria de logs reais e dashboard comparativo
baseline versus semantic.

## Objetivo do case
Demonstrar uma arquitetura prática de RAG clínico usando dados sintéticos para:
- Recuperar evidências clínicas a partir de documentos PDF.
- Gerar respostas com citação de fontes, páginas e chunks.
- Comparar estratégias de chunking e retrieval.
- Avaliar qualidade com métricas como Recall@K, MRR e Precision@K.
- Avaliar respostas com critérios de groundedness, correctness, citation quality e hallucination risk.
- Registrar interações reais via Streamlit para futura curadoria.

## Arquivos
- consultas_ambulatoriais_032026
- hemograma_e_bioquimica_032026
- ressonancia_coluna
- parecer_cardiologista
- alta_hospitalar
## Detalhes
- csv/: dados consolidados por tema, com 10 pacientes por arquivo.
- pdf/: documentos simulados para ingestao via RAG, incluindo laudos com elementos visuais simples.

## Estrutura do projeto

```text
clinical_rag/
│
├── 00_ground_truth/
│   ├── csv/
│
├── 01_database/
│   └── documentos clínicos sintéticos em PDF
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

## Documentos sintéticos

Arquivos clínicos simulados usados no case:

```text
consultas_ambulatoriais_032026.pdf
hemograma_e_bioquimica_032026.pdf
ressonancia_coluna.pdf
parecer_cardiologista.pdf
alta_hospitalar.pdf
```

## Componentes principais

### 00_ground_truth/
Contém dados sintéticos consolidados por tema e CSVs simulados.

### 01_database/
Contém os PDFs clínicos sintéticos usados como base documental para ingestão no RAG.
Essa camada representa a origem controlada dos dados usados no treinamento e nos testes.

### 02_processing/
Responsável por ingestão, chunking, embeddings e indexação.

Arquivos principais:
```text
00_upload_s3.py
```
Envia PDFs para o bucket S3.

```text
01_chunking_baseline.py
```
Cria chunks baseline por página e tokens.

```text
02_chunking_semantic.py
```
Cria chunks semânticos por paciente, seção clínica e estrutura documental.

```text
03_generate_embeddings_baseline.py
```
Gera embeddings Titan para chunks baseline.

```text
03_generate_embeddings_semantic.py
```
Gera embeddings Titan para chunks semânticos.

```text
04_indexing.py
```
Cria índices FAISS baseline e semantic e salva os artefatos localmente e no S3.

## Artefatos no S3
Bucket usado no projeto:
```text
clinical-rag-database
```

Principais prefixos:
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
Contém as estratégias de recuperação e geração de resposta:

```text
01_baseline_rag.py
```
Executa RAG baseline com Titan Embeddings, FAISS e Claude.

```text
02_query_rewrite.py
```
Reescreve perguntas clínicas para melhorar o retrieval.

```text
03_hybrid_search.py
```
Combina busca vetorial FAISS com busca lexical BM25 e Reciprocal Rank Fusion.

```text
04_rag_fusion.py
```
Gera múltiplas queries, executa retrieval por query e consolida resultados com RRF.

```text
05_corrective_rag.py
```
Avalia contexto recuperado, reescreve a query se necessário e executa nova recuperação.

## 04_agentic/
Contém orquestração agentic:
```text
01_langgraph_rag.py
```
Orquestra o fluxo RAG com LangGraph:
```text
retrieve
evaluate
rewrite, se necessário
answer
```

Também pode gerar grafo visual dinâmico da resposta, evidências, avaliação e fontes.
```text
02_react_agent.py
```

Executa agente ReAct com ferramentas:
```text
retrieve_baseline
retrieve_semantic
query_rewrite
final_answer
```

## 05_evaluation/
Contém avaliação offline do RAG.
```text
00_ground_truth_evaluation_dataset.py
```

Gera dataset ground truth em JSONL para avaliação offline.
Saída:
```text
ground_truth_evaluation_dataset.jsonl
```

```text
01_evaluate_retrieval.py
```

Avalia retrieval baseline versus semantic com:
```text
Recall@K
MRR
Precision@K
Hit rank
Best score
```

Saída principal:
```text
retrieval_eval_results.jsonl
```

```text
02_evaluate_answer.py
```

Avalia a resposta final do RAG contra o ground truth.
Critérios:
```text
termos esperados
paciente
documento
citação
fonte compatível
```

Saída:
```text
answer_eval_results.jsonl
```

```text
03_llm_as_judge.py
```

Usa Claude como juiz para avaliar:
```text
groundedness
correctness
citation quality
completeness
hallucination risk
overall score
```

Saída:
```text
llm_judge_results.jsonl
```

```text
04_UI_convert_interaction_logs.py
```

Converte logs reais do Streamlit em candidatos para curadoria.
Entrada:
```text
../06_app_ui/interaction_logs.jsonl
```

Saída:
```text
curation_candidates.jsonl
```

```text
05_UI_curate_ground_truth_dataset.py
```

Transforma candidatos derivados do Streamlit em ground truth curado.
Saídas:
```text
curation_review_template.jsonl
curated_ground_truth_dataset.jsonl
```

```text
06_UI_merge_evaluation_datasets.py
```

Une o dataset seed com o dataset curado real.
Entradas:
```text
ground_truth_evaluation_dataset.jsonl
curated_ground_truth_dataset.jsonl
```

Saída:
```text
full_ground_truth_evaluation_dataset.jsonl
```

## 06_app_ui/
Contém as interfaces Streamlit e logging real das interações.
```text
00_streamlit.py
```
Interface principal para perguntas clínicas.
Fluxo:
```text
pergunta clínica
retrieval FAISS baseline ou semantic
Claude via Bedrock
resposta com fontes
registro em interaction_logs.jsonl
feedback do usuário
marcação para curadoria
```

```text
interaction_logger.py
```
Módulo usado pelo Streamlit para:
```text
criar interaction_id
registrar interaction_logs.jsonl
contar interações
atualizar feedback
marcar curadoria
exportar candidatos
```

```text
02_dev_evaluation_dashboard.py
```
Dashboard de desenvolvimento para comparar baseline versus semantic.
Lê:
```text
../05_evaluation/retrieval_eval_results.jsonl
../05_evaluation/answer_eval_results.jsonl
../05_evaluation/llm_judge_results.jsonl
```

Mostra:
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

## Passo a passo simplificado
### 1. Subir PDFs no S3
```powershell
cd clinical_rag/02_processing
python 00_upload_s3.py
```

### 2. Gerar chunks
```powershell
python 01_chunking_baseline.py
python 02_chunking_semantic.py
```

### 3. Gerar embeddings
```powershell
python 03_generate_embeddings_baseline.py
python 03_generate_embeddings_semantic.py
```

### 4. Criar índices FAISS
```powershell
python 04_indexing_faiss.py
```

Ou por índice:

```powershell
python 04_indexing_faiss.py --only baseline
python 04_indexing_faiss.py --only semantic
```

### 5. Rodar retrieval baseline
```powershell
cd ../03_retrieval
python 01_baseline_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 6. Rodar estratégias avançadas de retrieval
```powershell
python 02_query_rewrite.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 03_hybrid_search.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 04_rag_fusion.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 05_corrective_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 7. Rodar LangGraph e ReAct Agent
```powershell
cd ../04_agentic
python 01_langgraph_rag.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
python 02_react_agent.py --question "Qual foi o resultado da creatinina da paciente Gabriela Lima?"
```

### 8. Rodar avaliação offline
```powershell
cd ../05_evaluation
python 00_ground_truth_evaluation_dataset.py
python 01_evaluate_retrieval.py
python 02_evaluate_answer.py
python 03_llm_as_judge.py
```

### 9. Rodar Streamlit principal
```powershell
cd ../06_app_ui
streamlit run 00_streamlit.py
```

### 10. Rodar dashboard dev de avaliação
```powershell
streamlit run 02_dev_evaluation_dashboard.py
```

## Fluxo com logs reais do Streamlit
Depois de usar o Streamlit principal e gerar `interaction_logs.jsonl`:

```powershell
cd ../05_evaluation
python 04_UI_convert_interaction_logs.py
python 05_UI_curate_ground_truth_dataset.py
python 06_UI_merge_evaluation_datasets.py
```

Depois rode avaliações com o dataset completo:
```powershell
python 01_evaluate_retrieval.py --eval-file full_ground_truth_evaluation_dataset.jsonl
python 02_evaluate_answer.py --eval-file full_ground_truth_evaluation_dataset.jsonl
python 03_llm_as_judge.py
```

## Perguntas de teste
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

## Métricas de avaliação
### Recall@K
Mede se o item esperado apareceu entre os K primeiros resultados.
```text
Recall@K = perguntas com acerto no Top-K / total de perguntas
```

### MRR
Mede quão cedo aparece o primeiro resultado relevante.
```text
MRR = média de 1 / rank do primeiro resultado relevante
```

### Precision@K
Mede a proporção de resultados relevantes dentro do Top-K.
```text
Precision@K = itens relevantes no Top-K / K
```

### Groundedness
Mede se a resposta está suportada pelas fontes recuperadas.

### Correctness
Mede se a resposta está correta em relação ao ground truth.

### Citation quality
Mede se documento, página e chunk foram citados corretamente.

### Hallucination risk
Classifica o risco de a resposta conter informação não sustentada pelas fontes.

## Observações importantes
- O projeto usa dados sintéticos.
- Não há dados reais de pacientes.
- Não deve ser usado para diagnóstico, triagem clínica real ou decisão médica.
- O objetivo é demonstrar arquitetura, retrieval, avaliação e rastreabilidade.
- O dashboard dev usa arquivos JSONL, não JSON, para as avaliações principais.
- A pasta `06_app_ui/` substitui a antiga `06_app/`.
