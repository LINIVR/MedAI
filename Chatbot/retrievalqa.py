"""
RetrievalQA Chain for MEDAI Chatbot

Builds a ConversationalRetrievalChain using:
- Groq  LLM
- FAISS vectorstore from embedded PDFs
- Custom prompt for skin disease chatbot


"""
import os
import sys
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .vectorstorebuilder import get_vectorstore
from medai_logger import get_logger

load_dotenv()

logger = get_logger("retrievalqa")




def get_prompt_template() -> PromptTemplate:
    """Returns the custom prompt template for medical Q&A."""
    template = """
You are a medical assistant chatbot specialized in skin-related conditions.  
You must answer only using the information provided in the context below.

Instructions:

- If the user describes SYMPTOMS (e.g., "I have redness and itching"):
  1. Search the context for skin diseases that match those symptoms.
  2. Suggest up to 3 possible diseases mentioned in the context.
  

- If the user asks about a DISEASE (e.g., "What is eczema?"):
  1. Provide a clear and complete description using only the context.(minimum 5 sentence.)
  2. If required information is missing, explicitly state: "This information is not available in the provided context."
  

Important Rules:
- Do NOT use your own knowledge.
- Do NOT guess or hallucinate any information.
- ONLY use medically relevant information from the context.
- If any part of the context includes unrelated content (e.g., disclaimers, footnotes, headers, or non-medical text), ignore it and do NOT include it in your answer.
- If no relevant information is found, reply: "Sorry, the information is not currently available in our medical knowledge base."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

def get_retrieval_chain() -> ConversationalRetrievalChain:
    """
    Initializes the RetrievalQA pipeline with custom prompt.

    Returns:
        ConversationalRetrievalChain: The initialized chain.

    
    """
    try:
        logger.info("Initializing vectorstore...")
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        logger.info("Initializing LLM...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        llm = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=0.0,
            api_key=api_key
        )

        prompt = get_prompt_template()

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
        )

        logger.info("ConversationalRetrievalChain initialized successfully.")
        return chain

    except Exception as e:
        logger.critical("Failed to initialize RetrievalQA chain: %s", str(e))
        raise RuntimeError(f"Chain initialization failed: {str(e)}") from e

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Retrieves or creates chat history for a given session.

    Args:
        session_id (str): Identifier for the chat session.

    Returns:
        InMemoryChatMessageHistory: The session's chat history.
    """
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]

# In-memory store for chat histories
chat_histories = {}

def chat(query: str, session_id: str = "default_session") -> dict:
    """
    Runs a chat query with the RAG system and returns the answer and sources.

    Args:
        query (str): The user's question.
        session_id (str): Identifier for the chat session (default: "default_session").

    Returns:
        dict: Contains the answer and source documents.

    """
    try:
        if not query.strip():
            logger.warning("Empty query received.")
            return {"answer": "Please provide a valid query.", "source_documents": []}

        chain = get_retrieval_chain()
        qa_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            output_messages_key="answer",
            history_messages_key="chat_history"
        )

        logger.info("Processing query: %s", query)
        result = qa_with_history.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )

        # Deduplicate source documents based on content
        unique_docs = []
        seen_content = set()
        for doc in result.get("source_documents", []):
            content = doc.page_content
            if content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content)

        logger.info("Query processed successfully, retrieved %d unique documents.", len(unique_docs))
        return {
            "answer": result.get("answer", "No response generated."),
            "source_documents": unique_docs
        }

    except Exception as e:
        logger.error("Failed to process query '%s': %s", query, str(e))
        raise RuntimeError(f"Query processing failed: {str(e)}") from e

if __name__ == "__main__":
    print("Welcome to the Medical Assistant Chatbot for Skin Conditions!")
    print("Type 'exit' or 'quit' to stop.\n")

    session_id = "default_session"
    try:
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                logger.info("Chat session terminated by user.")
                print("Goodbye!")
                break

            result = chat(user_query, session_id)
            print("\nBot:", result["answer"])

            # Output full retrieved chunks 
            if result["source_documents"]:
                print("\nRetrieved Source Documents:")
                for i, doc in enumerate(result["source_documents"], 1):
                    source = doc.metadata.get("source", "Unknown source")
                    content = doc.page_content.replace("\n", " ") 
                    print(f"Document {i}:")
                    print(f"  Source: {source}")
                    print(f"  Content: {content}")
                    print("-" * 50)
            else:
                print("\nNo source documents retrieved.")
                logger.warning("No source documents found for query: %s", user_query)

    except KeyboardInterrupt:
        logger.info("Chat session interrupted by user.")
        print("\nSession interrupted. Goodbye!")
    except Exception as e:
        logger.critical("Unexpected error in chat loop: %s", str(e))
        print(f"\nAn unexpected error occurred: {str(e)}")