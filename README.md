# Financial Report Intelligent Analyst

Built an agentic RAG engine to audit 10-K reports. It tries to solve two problems: scrambled PDF tables and math hallucinations.

Used `pdfplumber` to force tables into Markdown such that the agent understands rows and columns.

Instead of guessing ratios, the LLM has to find the raw numbers via search and then hand them off to a local Python script for the actual calculation.

## Set up

1. `docker compose up -d`
2. Ingest: `python3 -m src.ingest` ( put the pdf under `/data`)
3. Run: `python3 -m src.main`
