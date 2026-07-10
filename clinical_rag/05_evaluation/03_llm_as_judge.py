# OBJETIVO PRINCIPAL
# Avaliar respostas do RAG clínico com LLM as Judge, medindo groundedness, correctness, citation quality e risco de alucinação.

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3


DEFAULT_REGION = "us-east-1"
DEFAULT_INPUT_JSONL = "answer_eval_results.jsonl"
DEFAULT_OUTPUT_JSONL = "llm_judge_results.jsonl"

DEFAULT_LLM_MODEL_ID = "arn:aws:bedrock:us-east-1:789065179500:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_MAX_TOKENS = 1200
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PASS_THRESHOLD = 4.0


def normalize_text(value: Any) -> str:
    """Normaliza texto básico para logs e validações."""
    return str(value or "").strip()


def load_jsonl(input_file: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Carrega registros JSONL."""
    path = Path(input_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}. "
            "Execute antes: python 02_evaluate_answer.py"
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

            if limit and len(records) >= limit:
                break

    return records


def save_jsonl(records: List[Dict[str, Any]], output_file: str) -> None:
    """Salva registros em JSONL."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_sources(sources: List[Dict[str, Any]], max_sources: int = 5) -> List[Dict[str, Any]]:
    """Reduz fontes para o prompt do juiz."""
    compacted = []

    for source in sources[:max_sources]:
        compacted.append({
            "rank": source.get("rank"),
            "score": source.get("score"),
            "document_name": source.get("document_name"),
            "document_type": source.get("document_type"),
            "patient_id": source.get("patient_id"),
            "patient_name": source.get("patient_name"),
            "clinical_section": source.get("clinical_section"),
            "page_start": source.get("page_start"),
            "page_end": source.get("page_end"),
            "chunk_number": source.get("chunk_number"),
            "total_chunks": source.get("total_chunks"),
            "text_preview": source.get("text_preview"),
        })

    return compacted


def build_judge_prompt(record: Dict[str, Any], pass_threshold: float) -> str:
    """Monta prompt do LLM as Judge."""
    question = record.get("question")
    expected = record.get("expected") or {}
    answer = record.get("answer")
    sources = compact_sources(record.get("sources") or [])
    rule_based_evaluation = record.get("evaluation") or {}

    return f"""
Você é um avaliador técnico de qualidade para um sistema RAG clínico sintético.

Avalie a RESPOSTA usando somente:
1. Pergunta
2. Ground truth esperado
3. Fontes recuperadas
4. Resposta gerada

Não use conhecimento externo.
Não assuma fatos clínicos fora das fontes.
Não seja permissivo com fonte errada, paciente errado ou valor clínico inventado.

Pergunta:
{question}

Ground truth esperado:
{json.dumps(expected, ensure_ascii=False, indent=2)}

Resposta gerada:
{answer}

Fontes recuperadas:
{json.dumps(sources, ensure_ascii=False, indent=2)}

Avaliação determinística anterior:
{json.dumps(rule_based_evaluation, ensure_ascii=False, indent=2)}

Critérios:
- groundedness_score: 0 a 5, mede se a resposta é suportada pelas fontes.
- correctness_score: 0 a 5, mede se a resposta atende ao ground truth.
- citation_quality_score: 0 a 5, mede se documento, página e chunk foram citados corretamente.
- completeness_score: 0 a 5, mede se a resposta cobre o que foi perguntado sem excesso.
- hallucination_risk: low, medium ou high.
- expected_evidence_found: true se as fontes contêm a evidência esperada.
- source_supports_answer: true se a resposta está suportada pelas fontes.
- pass: true apenas se overall_score >= {pass_threshold}, hallucination_risk não for high e source_supports_answer for true.

Retorne apenas JSON válido, sem markdown, no formato:

{{
  "groundedness_score": 0,
  "correctness_score": 0,
  "citation_quality_score": 0,
  "completeness_score": 0,
  "overall_score": 0,
  "hallucination_risk": "low",
  "expected_evidence_found": false,
  "source_supports_answer": false,
  "pass": false,
  "issues": ["..."],
  "rationale": "explicação curta"
}}
""".strip()


def invoke_claude_text(
    bedrock_client,
    model_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Invoca Claude via Bedrock."""
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


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extrai JSON de resposta textual."""
    text = normalize_text(text)

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("A resposta do juiz não contém JSON válido.")

    return json.loads(match.group(0))


def clamp_score(value: Any) -> float:
    """Garante score entre 0 e 5."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0

    return max(0.0, min(5.0, score))


def normalize_judge_result(judge_result: Dict[str, Any], pass_threshold: float) -> Dict[str, Any]:
    """Padroniza saída do juiz."""
    groundedness = clamp_score(judge_result.get("groundedness_score"))
    correctness = clamp_score(judge_result.get("correctness_score"))
    citation_quality = clamp_score(judge_result.get("citation_quality_score"))
    completeness = clamp_score(judge_result.get("completeness_score"))

    calculated_overall = round(
        (groundedness + correctness + citation_quality + completeness) / 4,
        4,
    )

    overall_score = clamp_score(judge_result.get("overall_score", calculated_overall))

    hallucination_risk = str(judge_result.get("hallucination_risk", "medium")).lower()

    if hallucination_risk not in {"low", "medium", "high"}:
        hallucination_risk = "medium"

    expected_evidence_found = bool(judge_result.get("expected_evidence_found"))
    source_supports_answer = bool(judge_result.get("source_supports_answer"))

    passed = (
        overall_score >= pass_threshold
        and hallucination_risk != "high"
        and source_supports_answer
    )

    issues = judge_result.get("issues")

    if not isinstance(issues, list):
        issues = []

    return {
        "groundedness_score": groundedness,
        "correctness_score": correctness,
        "citation_quality_score": citation_quality,
        "completeness_score": completeness,
        "overall_score": overall_score,
        "hallucination_risk": hallucination_risk,
        "expected_evidence_found": expected_evidence_found,
        "source_supports_answer": source_supports_answer,
        "pass": passed,
        "issues": [str(issue) for issue in issues],
        "rationale": normalize_text(judge_result.get("rationale")),
    }


def fallback_judge_result(error: Exception) -> Dict[str, Any]:
    """Retorna avaliação de erro quando o LLM Judge falha."""
    return {
        "groundedness_score": 0.0,
        "correctness_score": 0.0,
        "citation_quality_score": 0.0,
        "completeness_score": 0.0,
        "overall_score": 0.0,
        "hallucination_risk": "medium",
        "expected_evidence_found": False,
        "source_supports_answer": False,
        "pass": False,
        "issues": [f"llm_judge_error:{error}"],
        "rationale": "Falha ao executar LLM as Judge.",
    }


def judge_record(
    record: Dict[str, Any],
    bedrock_client,
    llm_model_id: str,
    max_tokens: int,
    temperature: float,
    pass_threshold: float,
) -> Dict[str, Any]:
    """Avalia um registro individual."""
    prompt = build_judge_prompt(
        record=record,
        pass_threshold=pass_threshold,
    )

    try:
        raw_response = invoke_claude_text(
            bedrock_client=bedrock_client,
            model_id=llm_model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        parsed = extract_json_from_text(raw_response)
        judge = normalize_judge_result(
            judge_result=parsed,
            pass_threshold=pass_threshold,
        )
        judge["raw_judge_response"] = raw_response

    except Exception as exc:
        judge = fallback_judge_result(exc)
        judge["raw_judge_response"] = ""

    return {
        "evaluation_type": "llm_as_judge",
        "index_mode": record.get("index_mode"),
        "question": record.get("question"),
        "expected": record.get("expected"),
        "answer": record.get("answer"),
        "rule_based_evaluation": record.get("evaluation"),
        "llm_judge": judge,
        "sources": record.get("sources"),
    }


def print_summary(records: List[Dict[str, Any]]) -> None:
    """Imprime resumo agregado por índice."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        index_mode = record.get("index_mode") or "unknown"
        grouped.setdefault(index_mode, []).append(record)

    summaries = []

    for index_mode, items in grouped.items():
        total = len(items)
        passed = sum(1 for item in items if item["llm_judge"]["pass"])
        avg_overall = (
            sum(item["llm_judge"]["overall_score"] for item in items) / total
            if total
            else 0.0
        )
        high_risk = sum(
            1
            for item in items
            if item["llm_judge"]["hallucination_risk"] == "high"
        )

        summaries.append({
            "index_mode": index_mode,
            "total_questions": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_overall_score": round(avg_overall, 4),
            "high_hallucination_risk_count": high_risk,
        })

    print("\nResumo final")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Executa LLM as Judge sobre respostas avaliadas do RAG clínico."
    )

    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Arquivo JSONL gerado pelo 02_evaluate_answer.py.",
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL de saída com julgamento do LLM.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help="Região AWS.",
    )

    parser.add_argument(
        "--llm-model-id",
        default=DEFAULT_LLM_MODEL_ID,
        help="Modelo LLM ou ARN do inference profile no Bedrock.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Máximo de tokens do juiz.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperatura do juiz.",
    )

    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=DEFAULT_PASS_THRESHOLD,
        help="Nota mínima de overall_score para passar.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita quantidade de registros avaliados.",
    )

    args = parser.parse_args()

    records = load_jsonl(
        input_file=args.input_jsonl,
        limit=args.limit,
    )

    if not records:
        print("Nenhum registro encontrado para julgamento.")
        return

    print("Iniciando LLM as Judge")
    print(f"Input: {args.input_jsonl}")
    print(f"Output: {args.output_jsonl}")
    print(f"Registros: {len(records)}")
    print(f"Pass threshold: {args.pass_threshold}")

    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)

    judged_records = []

    for idx, record in enumerate(records, start=1):
        print("\n" + "=" * 90)
        print(f"Julgando {idx}/{len(records)}")
        print(f"Index mode: {record.get('index_mode')}")
        print(f"Pergunta: {record.get('question')}")

        judged = judge_record(
            record=record,
            bedrock_client=bedrock_client,
            llm_model_id=args.llm_model_id,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            pass_threshold=args.pass_threshold,
        )

        judged_records.append(judged)

        judge = judged["llm_judge"]
        print(f"Pass: {judge['pass']}")
        print(f"Overall score: {judge['overall_score']}")
        print(f"Hallucination risk: {judge['hallucination_risk']}")

        if judge["issues"]:
            print("Issues:")
            print(json.dumps(judge["issues"], ensure_ascii=False, indent=2))

    save_jsonl(judged_records, args.output_jsonl)
    print_summary(judged_records)

    print("\nArquivo gerado:")
    print(args.output_jsonl)


if __name__ == "__main__":
    main()