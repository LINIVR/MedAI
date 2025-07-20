"""
RetrievalQA Chain for MEDAI Chatbot

Builds a ConversationalRetrievalChain using:
- Groq (LLaMA3) LLM
- FAISS vectorstore from embedded PDFs
- Custom prompt for skin disease chatbot

Returns source documents internally for test tracking.
"""

import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .vectorstorebuilder import get_vectorstore
from medai_logger import get_logger

load_dotenv()


logger = get_logger("retrievalqa")


def get_prompt_template() -> PromptTemplate:
    """
    Returns the custom prompt template for medical Q&A.
    """
    template = """
You are a medical assistant chatbot specialized in skin-related conditions.  
You must answer only using the information provided in the context below.

Instructions:

- If the user describes SYMPTOMS (e.g., "I have redness and itching"):
  1. Search the context for skin diseases that match those symptoms.
  2. Suggest up to 3 possible diseases mentioned in the context.
  3. Ask: "Would you like to know more about any of these?"

- If the user asks about a DISEASE (e.g., "What is eczema?"):
  1. Provide a clear and complete description using only the context.
  2. Include the definition, common symptoms, treatments, and any medications.
  3. End with:  
     "This tool is for awareness only. If you experience these symptoms, please consult a doctor."

Important Rules:
- Do NOT use your own knowledge.
- Do NOT guess or hallucinate any information.
- ONLY use medically relevant information from the context.
- ❗ If any part of the context includes unrelated content (e.g., disclaimers, footnotes, headers, page numbers, or non-medical text), ignore it and do NOT include it in your answer.
- If the answer is not found in the context, reply:  
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


def get_retrieval_chain() -> ConversationalRetrievalChain:
    """
    Initializes the RetrievalQA pipeline with memory and custom prompt.
    """
    try:
        logger.info("Initializing vectorstore and LLM...")
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(
            search_kwargs={ "k": 10})

        llm = ChatGroq(
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.0
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  
        )

        prompt = get_prompt_template()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        logger.info("ConversationalRetrievalChain initialized successfully.")
        return chain

    except Exception as e:
        logger.critical("Error initializing RetrievalQA chain: %s", str(e))
        raise


if __name__ == "__main__":
    print("Chatbot Interaction\n")
    chain = get_retrieval_chain()

    while True:
        user_query = input(" You: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        try:
            result = chain.invoke({"question": user_query})
            print("\n Bot:", result.get("answer", "No response."))

            if "source_documents" in result:
                for doc in result["source_documents"]:
                    print(" Source:", doc.metadata.get("source", "unknown"))
                    print(" Content snippet:", doc.page_content[:200])
                    print("-" * 50)

        except Exception as err:
            print(" Error during response:", err)
            logger.error("Response generation failed: %s", str(err))
