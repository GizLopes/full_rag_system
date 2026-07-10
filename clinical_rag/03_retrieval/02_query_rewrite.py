# OBJETIVO PRINCIPAL
# Reescreve perguntas clínicas para melhorar retrieval, extraindo paciente,
# termos clínicos, tipo documental, filtros e queries alternativas

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError


DEFAULT_REGION = "us-east-1"
DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_OUTPUT_FILE = "query_rewrite_result.json"


DOCUMENT_TYPE_HINTS = {
    "hemograma_e_bioquimica": [
        "hemograma",
        "bioquimica",
        "bioquímica",
        "creatinina",
        "glicemia",
        "hemoglobina",
        "leucocitos",
        "leucócitos",
        "plaquetas",
        "exame",
        "laboratorial",
        "laboratoriais",
    ],
    "consultas_ambulatoriais": [
        "consulta",
        "consultas",
        "ambulatorial",
        "ambulatoriais",
        "queixa",
        "historia",
        "história",
        "conduta",
        "medicacoes atuais",
        "medicações atuais",
    ],
    "alta_hospitalar": [
        "alta",
        "hospitalar",
        "prescricao alta",
        "prescrição alta",
        "orientacoes",
        "orientações",
        "restricoes",
        "restrições",
        "diagnostico alta",
        "diagnóstico alta",
    ],
    "parecer_cardiologista": [
        "cardiologista",
        "cardiologia",
        "parecer",
        "ecg",
        "pressao arterial",
        "pressão arterial",
        "risco cardiovascular",
    ],
    "ressonancia_coluna": [
        "ressonancia",
        "ressonância",
        "coluna",
        "lombar",
        "cervical",
        "toracica",
        "torácica",
        "laudo",
        "imagem",
    ],
}


def normalize_text(text: str) -> str:
    """Normaliza espaços sem destruir o texto original da pergunta."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()


def infer_document_types_by_keywords(question: str) -> List[str]:
    """Infere tipos documentais candidatos por palavras-chave simples."""
    question_lower = normalize_text(question).lower()
    inferred = []

    for document_type, keywords in DOCUMENT_TYPE_HINTS.items():
        if any(keyword in question_lower for keyword in keywords):
            inferred.append(document_type)

    return inferred


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extrai o primeiro objeto JSON válido retornado pelo LLM."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("A resposta do LLM não contém JSON válido.")

    return json.loads(match.group(0))


def build_rewrite_prompt(
    question: str,
    keyword_document_types: List[str],
) -> str:
    """Cria prompt para reescrita de query clínica."""
    keyword_hints = keyword_document_types or ["nenhum tipo documental inferido por regra"]

    return f"""
Você é um componente de Query Rewrite para um sistema RAG clínico de treinamento.

Sua tarefa é transformar a pergunta do usuário em uma consulta melhor para recuperação vetorial e textual.

Contexto do corpus:
- Os documentos são sintéticos e clínicos.
- Existem documentos de consultas ambulatoriais, hemograma e bioquímica, parecer cardiologista, ressonância de coluna e alta hospitalar.
- Os chunks possuem metadados como document_type, document_name, page_start, page_end, patient_id, patient_name e clinical_section quando disponíveis.
- A saída será usada por FAISS, busca híbrida e/ou retrieval semântico.

Tipos documentais candidatos por regra:
{json.dumps(keyword_hints, ensure_ascii=False)}

Pergunta original:
{question}

Regras:
1. Não responda à pergunta clínica.
2. Não invente resultado clínico.
3. Apenas reescreva a consulta para melhorar retrieval.
4. Extraia entidades explícitas, como nome do paciente, patient_id, exame, medicação, data, documento ou seção clínica.
5. Se não houver patient_id, deixe null.
6. Se o tipo documental for incerto, use lista vazia.
7. Gere consultas alternativas curtas, úteis para busca vetorial e busca textual.
8. Retorne apenas JSON válido, sem markdown.

Formato obrigatório:
{{
  "original_question": "...",
  "normalized_question": "...",
  "rewritten_question": "...",
  "patient_name": null,
  "patient_id": null,
  "clinical_terms": [],
  "target_document_types": [],
  "target_sections": [],
  "expanded_queries": [],
  "keyword_query": "...",
  "retrieval_filters": {{
    "patient_name": null,
    "patient_id": null,
    "document_type": [],
    "clinical_section": []
  }},
  "reasoning_summary": "Resumo curto do objetivo da reescrita, sem responder à pergunta."
}}
""".strip()


def invoke_claude_for_rewrite(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """Invoca Claude via Bedrock e retorna JSON de query rewrite."""
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
        raise RuntimeError("Claude não retornou conteúdo.")

    raw_text = content[0].get("text", "").strip()

    if not raw_text:
        raise RuntimeError("Claude retornou texto vazio.")

    return extract_json_from_text(raw_text)


def build_rule_based_fallback(
    question: str,
    keyword_document_types: List[str],
) -> Dict[str, Any]:
    """Fallback simples caso o LLM falhe."""
    normalized_question = normalize_text(question)

    patient_id_match = re.search(r"\bP\d{3}\b", normalized_question, flags=re.IGNORECASE)
    patient_id = patient_id_match.group(0).upper() if patient_id_match else None

    patient_name = None

    name_patterns = [
        r"paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"de\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"da\s+paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
        r"do\s+paciente\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)+)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, normalized_question)

        if match:
            patient_name = match.group(1).strip(" ?.,;:")
            break

    clinical_terms = []

    for document_type in keyword_document_types:
        for keyword in DOCUMENT_TYPE_HINTS.get(document_type, []):
            if keyword.lower() in normalized_question.lower():
                clinical_terms.append(keyword)

    clinical_terms = sorted(set(clinical_terms))

    rewritten_parts = []

    if patient_name:
        rewritten_parts.append(f"paciente {patient_name}")

    if patient_id:
        rewritten_parts.append(f"patient_id {patient_id}")

    if clinical_terms:
        rewritten_parts.append(" ".join(clinical_terms))

    rewritten_parts.append(normalized_question)

    rewritten_question = " | ".join(rewritten_parts)

    expanded_queries = [
        normalized_question,
        rewritten_question,
    ]

    if patient_name and clinical_terms:
        expanded_queries.append(f"{patient_name} {' '.join(clinical_terms)}")

    if patient_id and clinical_terms:
        expanded_queries.append(f"{patient_id} {' '.join(clinical_terms)}")

    return {
        "original_question": question,
        "normalized_question": normalized_question,
        "rewritten_question": rewritten_question,
        "patient_name": patient_name,
        "patient_id": patient_id,
        "clinical_terms": clinical_terms,
        "target_document_types": keyword_document_types,
        "target_sections": [],
        "expanded_queries": expanded_queries,
        "keyword_query": " ".join(
            item for item in [patient_id, patient_name, *clinical_terms]
            if item
        ),
        "retrieval_filters": {
            "patient_name": patient_name,
            "patient_id": patient_id,
            "document_type": keyword_document_types,
            "clinical_section": [],
        },
        "reasoning_summary": "Fallback local aplicado para extrair entidades e termos úteis para retrieval.",
    }


def rewrite_query(
    question: str,
    region: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    use_llm: bool,
) -> Dict[str, Any]:
    """Executa query rewrite usando LLM ou fallback local."""
    normalized_question = normalize_text(question)
    keyword_document_types = infer_document_types_by_keywords(normalized_question)

    if not use_llm:
        return build_rule_based_fallback(
            question=normalized_question,
            keyword_document_types=keyword_document_types,
        )

    bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    prompt = build_rewrite_prompt(
        question=normalized_question,
        keyword_document_types=keyword_document_types,
    )

    try:
        result = invoke_claude_for_rewrite(
            bedrock_client=bedrock_client,
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except (ClientError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Aviso: falha no LLM para query rewrite. Usando fallback local. Erro: {exc}")
        result = build_rule_based_fallback(
            question=normalized_question,
            keyword_document_types=keyword_document_types,
        )

    if not result.get("target_document_types") and keyword_document_types:
        result["target_document_types"] = keyword_document_types

    retrieval_filters = result.get("retrieval_filters") or {}
    if not retrieval_filters.get("document_type") and keyword_document_types:
        retrieval_filters["document_type"] = keyword_document_types
    result["retrieval_filters"] = retrieval_filters

    return result


def save_json(result: Dict[str, Any], output_file: Path) -> None:
    """Salva resultado em JSON local."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reescreve perguntas clínicas para melhorar retrieval em RAG."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Pergunta original do usuário.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--llm-model-id",
        default=DEFAULT_LLM_MODEL_ID,
        help="Modelo LLM ou ARN do inference profile no Amazon Bedrock.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens gerados pelo LLM.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperatura do LLM.",
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Usa apenas fallback local baseado em regras.",
    )

    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Arquivo JSON local com a query reescrita.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Não salva o resultado em arquivo local.",
    )

    args = parser.parse_args()

    print("Iniciando query rewrite")
    print(f"Pergunta original: {args.question}")
    print(f"Usar LLM: {not args.no_llm}")

    result = rewrite_query(
        question=args.question,
        region=args.region,
        model_id=args.llm_model_id,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_llm=not args.no_llm,
    )

    print("\nResultado:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not args.no_save:
        output_file = Path(args.output_file)
        save_json(result, output_file)
        print(f"\nArquivo salvo: {output_file}")


if __name__ == "__main__":
    main()