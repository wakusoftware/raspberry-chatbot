from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def chat_completion_streaming(prompt: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512)
    for chunk in llm.stream(prompt):
        print(chunk.content, end="")


if __name__ == "__main__":
    chat_completion_streaming("What is the meaning of life?")
