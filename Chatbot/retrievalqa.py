"""
RetrievalQA Chain for MEDAI Chatbot

This module sets up the LangChain ConversationalRetrievalChain
using Groq (LLaMA3) and the local FAISS vectorstore.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from Chatbot.vectorstorebuilder import get_vectorstore


# Define log path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "retrieval_qa.log")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_prompt_template() -> PromptTemplate:
    """
    Returns the prompt template for medical assistant responses.

    Returns:
        PromptTemplate: Custom instruction-based prompt.
    """
    template = """
You are a medical assistant chatbot for skin-related conditions.
Use only the provided context to answer user questions.

Instructions:
- If the user describes SYMPTOMS:
  1. Suggest 3 possible skin diseases with brief summaries.
  2. Ask: "Would you like to know more about any of these?"

- If the user asks about a DISEASE:
  1. Explain the disease, common symptoms, and treatments (maximum 6 sentences).
  2. End with: "This tool is for awareness only. If you experience these symptoms, please consult a doctor."

If no information is found, reply:
"Sorry, the information is not currently available in our medical knowledge base."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}
"""
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )


def get_retrieval_chain():
    """
    Builds and returns the ConversationalRetrievalChain object.

    Returns:
        ConversationalRetrievalChain: LangChain Retrieval QA pipeline.
    """
    try:
        logger.info("Initializing vectorstore and LLM...")
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.1
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        prompt = get_prompt_template()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

        logger.info("ConversationalRetrievalChain initialized.")
        return chain

    except Exception as e:
        logger.critical("Error setting up RetrievalQA chain: %s", str(e))
        raise


if __name__ == "__main__":
    chain = get_retrieval_chain()
    print("Type your question (type 'exit' to quit):")

    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"question": query})
            print("\nBot:", result.get("answer", result))
        except Exception as e:
            print("\nBot: Something went wrong.")
            logger.error("Chatbot error: %s", str(e))
