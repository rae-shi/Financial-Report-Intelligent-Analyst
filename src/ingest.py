import os
from dotenv import load_dotenv
import pdfplumber
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
CONNECTION_STRING = os.getenv("DATABASE_URL")
# Vectorizing to PGVector
vector_store = PGVector(
    connection=CONNECTION_STRING,
    embeddings=OpenAIEmbeddings(),
    collection_name="nvidia_docs"
)

def ingest_report(pdf_path):
    print(f"Reading {pdf_path}...")
    
    # Extracting Text + Tables
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_str = ""
            for table in tables:
                for row in table:
                    table_str += "| " + " | ".join([str(c).strip() if c else "" for c in row]) + " |\n"
            full_text += f"\n{page.extract_text()}\n{table_str}"

    # Chunking for Financial Context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    vector_store.add_texts(chunks)
    print(f"Done! {len(chunks)} chunks are now in your database.")

if __name__ == "__main__":
    ingest_report("data/nvidia_report.pdf")