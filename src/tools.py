from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

CONNECTION_STRING = os.getenv("DATABASE_URL")

vector_store = PGVector(
    connection=CONNECTION_STRING,
    embeddings=OpenAIEmbeddings(),
    collection_name="nvidia_docs"
)

@tool
def calculate_financial_ratio(numerator: float, denominator: float) -> float:
    """
    Calculates a financial ratio (like Debt-to-Equity, Net Profit Margin, or ROE) 
    by dividing the numerator by the denominator. Use this for ANY mathematical 
    calculation to ensure 100% precision and avoid AI hallucinations.
    """
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)

@tool
def search_financial_docs(query: str) -> str:
    """
    Searches the uploaded 10-K financial reports. Use this to find line items like 
    'Total Revenue', 'Net Income', or 'Risk Factors'. This tool automatically retrieves 
    both narrative text and structured financial markdown tables.
    """
    
    # 1. Search for the top 3 most relevant TEXT chunks
    text_results = vector_store.similarity_search(
        query, 
        k=3, 
        filter={"type": "text"}
    )
    
    # 2. Search for the top 2 most relevant TABLES
    table_results = vector_store.similarity_search(
        query, 
        k=2, 
        filter={"type": "table"}
    )
    
    # 3. Format the output so the LLM clearly understands what it is reading
    formatted_output = "### SEARCH RESULTS ###\n\n"
    
    formatted_output += "--- NARRATIVE CONTEXT (TEXT) ---\n"
    if not text_results:
        formatted_output += "No relevant text found.\n"
    else:
        for i, doc in enumerate(text_results):
            source = doc.metadata.get('source', 'Unknown Document')
            page = doc.metadata.get('page', 'Unknown Page')
            formatted_output += f"[Text Result {i+1} from {source} (Page {page})]:\n{doc.page_content}\n\n"
            
    formatted_output += "--- FINANCIAL DATA (TABLES) ---\n"
    if not table_results:
        formatted_output += "No relevant tables found.\n"
    else:
        for i, doc in enumerate(table_results):
            source = doc.metadata.get('source', 'Unknown Document')
            page = doc.metadata.get('page', 'Unknown Page')
            formatted_output += f"[Table Result {i+1} from {source} (Page {page})]:\n{doc.page_content}\n\n"
            
    return formatted_output

# Make sure this is at the bottom of your file
tools = [calculate_financial_ratio, search_financial_docs]