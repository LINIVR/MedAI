"""
Vectorstore Builder for MEDAI

Loads PDF documents from `data/`, splits them into text chunks,
generates embeddings using HuggingFace, and stores them in a FAISS index.
"""

import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medai_logger import get_logger

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")


logger = get_logger("vectorstorebuilder")


os.makedirs(DB_FAISS_PATH, exist_ok=True)

def build_vectorstore():
    """
    Build and save a FAISS vectorstore from PDF documents.
    """
    try:
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info("Loaded %d documents.", len(documents))

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        logger.info("Split into %d chunks.", len(chunks))

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore saved at '%s'.", DB_FAISS_PATH)

    except Exception as e:
        logger.error("Error building vectorstore: %s", str(e))


def get_vectorstore():
    """
    Load and return the FAISS vectorstore from local storage.

    Returns:
        FAISS: A loaded FAISS vectorstore.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vectorstore loaded from '%s'.", DB_FAISS_PATH)
        return vectorstore

    except Exception as e:
        logger.error("Failed to load vectorstore: %s", str(e))
        raise


if __name__ == "__main__":
    build_vectorstore()
