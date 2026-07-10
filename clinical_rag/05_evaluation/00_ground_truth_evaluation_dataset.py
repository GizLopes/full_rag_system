# OBJETIVO PRINCIPAL
# Gerar dataset ground truth para avaliação offline do retrieval clínico, com perguntas, documentos, pacientes e termos esperados.

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_OUTPUT_JSONL = "ground_truth_evaluation_dataset.jsonl"

GROUND_TRUTH_EXAMPLES: List[Dict[str, Any]] = [
    {
        "question": "Qual foi o resultado da creatinina da paciente Gabriela Lima?",
        "expected_chunk_id": "",
        "expected_document_type": "hemograma_e_bioquimica",
        "expected_document_name": "hemograma_e_bioquimica_032026.pdf",
        "expected_patient_id": "P007",
        "expected_patient_name": "Gabriela Lima",
        "expected_terms": ["creatinina_mg_dl", "1.46"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual medicação atual aparece para Carla Mendes na consulta ambulatorial?",
        "expected_chunk_id": "",
        "expected_document_type": "consultas_ambulatoriais",
        "expected_document_name": "consultas_ambulatoriais_032026.pdf",
        "expected_patient_id": "P003",
        "expected_patient_name": "Carla Mendes",
        "expected_terms": ["medicacoes_atuais", "Sulfato ferroso"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual foi a prescrição de alta de Bruno Almeida?",
        "expected_chunk_id": "",
        "expected_document_type": "alta_hospitalar",
        "expected_document_name": "alta_hospitalar.pdf",
        "expected_patient_id": "P002",
        "expected_patient_name": "Bruno Almeida",
        "expected_terms": ["prescricao_alta", "Metformina 850 mg"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual foi o risco cardiovascular de Henrique Rocha no parecer cardiologista?",
        "expected_chunk_id": "",
        "expected_document_type": "parecer_cardiologista",
        "expected_document_name": "parecer_cardiologista.pdf",
        "expected_patient_id": "P008",
        "expected_patient_name": "Henrique Rocha",
        "expected_terms": ["risco_cardiovascular", "alto"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual laudo de ressonância aparece para Diego Santos?",
        "expected_chunk_id": "",
        "expected_document_type": "ressonancia_coluna",
        "expected_document_name": "ressonancia_coluna.pdf",
        "expected_patient_id": "P004",
        "expected_patient_name": "Diego Santos",
        "expected_terms": [],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual alergia foi registrada para Ana Ribeiro na consulta ambulatorial?",
        "expected_chunk_id": "",
        "expected_document_type": "consultas_ambulatoriais",
        "expected_document_name": "consultas_ambulatoriais_032026.pdf",
        "expected_patient_id": "P001",
        "expected_patient_name": "Ana Ribeiro",
        "expected_terms": ["Alergias", "Dipirona"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual foi a creatinina de João Pereira no exame de hemograma e bioquímica?",
        "expected_chunk_id": "",
        "expected_document_type": "hemograma_e_bioquimica",
        "expected_document_name": "hemograma_e_bioquimica_032026.pdf",
        "expected_patient_id": "P010",
        "expected_patient_name": "Joao Pereira",
        "expected_terms": ["creatinina_mg_dl", "0.77"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual medicação aparece para Fabio Oliveira na alta hospitalar?",
        "expected_chunk_id": "",
        "expected_document_type": "alta_hospitalar",
        "expected_document_name": "alta_hospitalar.pdf",
        "expected_patient_id": "P006",
        "expected_patient_name": "Fabio Oliveira",
        "expected_terms": ["prescricao_alta", "Enalapril 10 mg"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual foi a glicemia de Elisa Costa no exame laboratorial?",
        "expected_chunk_id": "",
        "expected_document_type": "hemograma_e_bioquimica",
        "expected_document_name": "hemograma_e_bioquimica_032026.pdf",
        "expected_patient_id": "P005",
        "expected_patient_name": "Elisa Costa",
        "expected_terms": ["glicemia_mg_dl", "129"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
    {
        "question": "Qual foi a pressão arterial de Gabriela Lima no parecer cardiologista?",
        "expected_chunk_id": "",
        "expected_document_type": "parecer_cardiologista",
        "expected_document_name": "parecer_cardiologista.pdf",
        "expected_patient_id": "P007",
        "expected_patient_name": "Gabriela Lima",
        "expected_terms": ["pressao_arterial", "141/83 mmHg"],
        "evaluation_scope": "retrieval",
        "dataset_type": "ground_truth_seed",
    },
]


def validate_examples(records: List[Dict[str, Any]]) -> None:
    """Valida campos mínimos do dataset."""
    required_fields = [
        "question",
        "expected_document_type",
        "expected_document_name",
        "expected_patient_id",
        "expected_patient_name",
        "expected_terms",
    ]

    for index, record in enumerate(records, start=1):
        missing_fields = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing_fields:
            raise ValueError(
                f"Exemplo {index} sem campos obrigatórios: {missing_fields}"
            )

        if not record["question"].strip():
            raise ValueError(f"Exemplo {index} está sem pergunta.")

        if not isinstance(record["expected_terms"], list):
            raise ValueError(
                f"Exemplo {index} precisa usar lista em expected_terms."
            )


def save_jsonl(records: List[Dict[str, Any]], output_file: Path) -> None:
    """Salva dataset em JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera ground truth evaluation dataset para avaliação de retrieval do RAG clínico."
    )

    parser.add_argument(
        "--output-jsonl",
        default=DEFAULT_OUTPUT_JSONL,
        help="Arquivo JSONL de saída.",
    )

    args = parser.parse_args()

    validate_examples(GROUND_TRUTH_EXAMPLES)

    save_jsonl(GROUND_TRUTH_EXAMPLES, Path(args.output_jsonl))

    print("Ground truth evaluation dataset gerado")
    print(f"Tipo: ground_truth_seed")
    print(f"Escopo: retrieval")
    print(f"Total de exemplos: {len(GROUND_TRUTH_EXAMPLES)}")
    print(f"JSONL: {args.output_jsonl}")

if __name__ == "__main__":
    main()