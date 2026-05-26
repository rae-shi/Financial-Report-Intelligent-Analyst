import os
from dotenv import load_dotenv
from langchain_core.documents import Document
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

def extract_financial_data(pdf_path):
    text_pages = []
    table_documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            current_page = page_num + 1

            raw_text = page.extract_text()
            if raw_text:
                page_doc = Document(
                    page_content=raw_text,
                    metadata={
                        "source": pdf_path,
                        "page": current_page,
                        "type": "text"
                    }
                )
                text_pages.append(page_doc)

            tables = page.extract_tables()
            for table in tables:
                table_str = ""

                for row in table:
                    table_str += "| " + " | ".join([str(c).strip() if c else "" for c in row]) + " |\n"

                table_doc = Document(
                    page_content=table_str,
                    metadata={
                        "source": pdf_path,
                        "page": current_page,
                        "type": "table"
                    }
                )
                table_documents.append(table_doc)
    return text_pages, table_documents

def ingest_report(pdf_path):
    print(f"Reading {pdf_path}...")

    text_pages, table_documents = extract_financial_data(pdf_path)

    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    text_chunks= text_splitter.split_documents(text_pages)

    if table_documents:
        print(f"Ingesting {len(table_documents)} table into PGVector...")
        vector_store.add_documents(table_documents)
    
    if text_chunks:
        print(f"Ingesting {len(text_chunks)} text chunks into PGVector...")
        vector_store.add_documents(text_chunks)
    
    print("Ingestion complete!")    

if __name__ == "__main__":
    ingest_report("data/nvidia_10_k.pdf")
