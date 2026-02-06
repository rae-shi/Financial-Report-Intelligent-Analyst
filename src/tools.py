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
    Searches the uploaded 10-K financial reports and company documents for 
    specific data points. Use this to find line items like 'Total Revenue', 
    'Net Income', 'Total Debt', or 'Risk Factors' before performing analysis.
    """
    
    # Retrieve the top 5 most relevant chunks from the 10-K
    results = vector_store.similarity_search(query, k=5)
    
    # Combine results into a single string for the Analyst to read
    return "\n\n".join([doc.page_content for doc in results])

# function
tools = [calculate_financial_ratio, search_financial_docs]