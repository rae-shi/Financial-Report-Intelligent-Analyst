import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from tools import tools

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent_executor = create_react_agent(model, tools)

def run_agent(query: str):
    response = agent_executor.invoke({"messages": [("user", query)]})
    # The last message in the state is the agent's final answer
    return response["messages"][-1].content