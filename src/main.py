from fastapi import FastAPI
from .graph import run_agent

app = FastAPI()

@app.post("/query")
def query(query: str):
    result = run_agent(query)
    return {"result": result}
