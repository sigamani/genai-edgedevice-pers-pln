from langchain_openai import ChatOpenAI

def openai_runner(prompt: str) -> str:
    llm = ChatOpenAI(temperature=0.3)
    return llm.invoke(prompt).content