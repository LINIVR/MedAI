"""
RetrievalQA Chain for medical chatbot.
Supports conversational memory, follow-up questions, and confirmation of symptoms.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from .vectorstore_builder import get_vectorstore

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/retrieval_qa.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger("retrieval_qa")

# Load environment variables
load_dotenv()

def get_prompt_template() -> PromptTemplate:
    """
    Returns the main prompt template for the chatbot.
    """
    template = """
You are a medical assistant chatbot for skin-related conditions.
Answer strictly based on the provided context. Do not guess or hallucinate.

Instructions:
- If the user provides SYMPTOMS (e.g., "I have itching and redness"):
  1. Retrieve and list the top 3 most likely diseases matching the symptoms from the database.
  2. For each disease, provide:
     - Disease Name
     - One-sentence summary
  3. Ask: "Would you like to know more about any of these diseases? If yes, please specify."

- If the user asks about a DISEASE:
  1. Respond with:
     - Disease Description
     - Common Symptoms
     - Treatments or Medications
  2. Keep it concise (~6 sentences).
  3. Add: "This tool is for awareness only. If you experience these symptoms, please consult a doctor."

Do not mention that you are an AI or that you are searching a database.
If no relevant info is found:
Reply: "Sorry, the information is not currently available in our medical knowledge base."

Context:
{context}

Chat History:
{chat_history}

User Question:
{question}
"""
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

def get_retrieval_chain():
    """
    Sets up and returns a ConversationalRetrievalChain with memory.
    """
    try:
        logger.info("Loading vectorstore...")
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.1
        )
        logger.info("LLM initialized.")

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        prompt = get_prompt_template()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        logger.info("ConversationalRetrievalChain created successfully.")
        return chain

    except Exception as e:
        logger.critical("Failed to set up RetrievalQA chain: %s", str(e))
        raise

if __name__ == "__main__":
    chain = get_retrieval_chain()
    print("Type your questions (type 'exit' or 'quit' to stop):")
    while True:
        user_question = input("\nYou: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        try:
            result = chain.invoke({"question": user_question})
            print("\nBot:", result.get("answer", result))
        except Exception as err:
            print("\nBot: Sorry, something went wrong. Please try again.")
            logger.error("Error during chatbot response: %s", str(err))
