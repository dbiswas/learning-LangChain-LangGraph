from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain_openai import ChatOpenAI
import os

load_dotenv()

def main():

    # print("Hello from langchain-langgraph!")
    # print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))
    # print("Google API Key:", os.getenv("GOOGLE_API_KEY"))

    information_text = """
    Langchain-LangGraph is a project that integrates Langchain and LangGraph to create a powerful language processing and graph-based knowledge representation system.
    It leverages the capabilities of Langchain for natural language understanding and generation, and LangGraph for graph-based data modeling and querying.
    The project aims to provide an efficient and scalable solution for building intelligent applications that require advanced language and graph processing capabilities.
    """

    summary_template = f"""
    Summarize the following information:
    {information_text}
    Summary:
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information_text"],
        template=summary_template
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information_text": information_text})
    print("Summary:", response.content)




if __name__ == "__main__":
    main()
