import boto3
from pathlib import Path

s3 = boto3.client("s3")

bucket = "clinical-rag-database-789065179500-us-east-1-an"

source_dir = Path(__file__).resolve().parent.parent / "01_database"

files = list(source_dir.glob("*.pdf"))

if not files:
    raise FileNotFoundError(
        f"Nenhum PDF encontrado em: {source_dir}"
    )

for file in files:
    s3.upload_file(
        str(file),
        bucket,
        f"rag-database/{file.name}"
    )

    print(f"Enviado: {file.name}")

print(f"Upload concluído: {len(files)} arquivo(s).")

#CORREÇÃO - Se Arquivos encontrados retornar [], o problema está no caminho ou no padrão dos arquivos
#Obs.: o terminal estava aberto em outra pasta
#from pathlib import Path

#print("Diretório atual:", Path.cwd())
#print("Pasta existe:", Path("clinical_rag/01_database").exists())
#print("Arquivos encontrados:", list(Path("clinical_rag/01_database").glob("*.pdf")))