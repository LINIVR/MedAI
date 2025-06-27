"""
Builds FAISS vector store from PDF documents using HuggingFace embeddings.
Handles PDF loading, chunking, embedding, and vector storage.

"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# logging setup
logging.basicConfig(
    filename=os.path.join(log_dir, "vectorstore_builder.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger("vectorstore_builder")


DATA_PATH = os.path.join(BASE_DIR, "data")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")

def load_pdf_files(data_dir):
    """Load PDF files from a specified directory."""
    try:
        loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info("Loaded %d PDF pages from %s.", len(documents), data_dir)
        return documents
    except Exception as e:
        logger.error("Failed to load PDF files: %s", str(e))
        raise

def create_chunks(documents):
    """Chunk loaded documents for vectorization."""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        logger.info("Created %d chunks from documents.", len(chunks))
        return chunks
    except Exception as e:
        logger.error("Failed to split documents: %s", str(e))
        raise

def get_embedding_model():
    """Return HuggingFace Embedding model instance."""
    try:
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded HuggingFace embedding model.")
        return model
    except Exception as e:
        logger.error("Failed to load embedding model: %s", str(e))
        raise

def create_and_save_vectorstore(chunks, embedding_model, save_path):
    """Create FAISS vector store from chunks and save it locally."""
    try:
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(save_path)
        logger.info("Saved FAISS vectorstore to %s.", save_path)
    except Exception as e:
        logger.error("Failed to create/save FAISS vectorstore: %s", str(e))
        raise

def run_pipeline():
    """Execute the vectorization pipeline from PDFs to FAISS."""
    try:
        documents = load_pdf_files(DATA_PATH)
        chunks = create_chunks(documents)
        embedding = get_embedding_model()
        create_and_save_vectorstore(chunks, embedding, DB_FAISS_PATH)
        logger.info("Vector building pipeline completed successfully.")
    except Exception as e:
        logger.critical("Pipeline execution failed: %s", str(e))

def get_vectorstore():
    """Load the existing FAISS vectorstore for use in retrieval."""
    try:
        embedding_model = get_embedding_model()
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("Loaded FAISS vectorstore from local path.")
        return vectorstore
    except Exception as e:
        logger.error("Failed to load FAISS vectorstore: %s", str(e))
        raise

if __name__ == "__main__":
    run_pipeline()
