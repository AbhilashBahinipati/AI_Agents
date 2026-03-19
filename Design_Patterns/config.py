import os
from pathlib import Path
from dotenv import load_dotenv
 
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "ollama"

def get_llm(temperature=0.0):
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="llama3.2",
            temperature=temperature
        )