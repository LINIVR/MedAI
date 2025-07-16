"""
Report Summarizer Module for MEDAI

Extracts text from PDF or image files (JPG, PNG)
and summarizes it using a Hugging Face transformer model.

Uses global logger from medai_logger.py and is compatible
with Docker, Hugging Face Spaces, and unit testing.
"""
import os
import sys
from typing import Optional

from PyPDF2 import PdfReader
from PIL import Image
from pytesseract import image_to_string
from transformers import pipeline


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medai_logger import get_logger
logger = get_logger("report_summarizer")



MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
logger = get_logger("report_summarizer")

_summarizer = None

def get_summarizer():
    """
    Loads and caches the Hugging Face summarization pipeline.

    Returns:
        transformers.Pipeline: Hugging Face summarization pipeline.
    """
    global _summarizer
    if _summarizer is None:
        logger.info(f"Loading summarization model: {MODEL_NAME}")
        try:
            _summarizer = pipeline("summarization", model=MODEL_NAME)
        except Exception as e:
            logger.exception("Failed to load summarizer model")
            raise RuntimeError(f"Could not load model: {e}")
    return _summarizer

def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Extracts text from a PDF or image file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        Optional[str]: Extracted text or None on failure.
    """
    try:
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            logger.info("Extracted %d characters from PDF", len(text))
            return text.strip()

        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            text = image_to_string(Image.open(file_path))
            logger.info("Extracted %d characters from image", len(text))
            return text.strip()

        else:
            logger.warning("Unsupported file type: %s", file_path)
            return None

    except Exception as e:
        logger.exception("Failed to extract text")
        return None

def summarize_report(file_path: str) -> str:
    """
    Summarizes a medical report from PDF or image file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        str: Summary text or error message.
    """
    logger.info("Processing file for summarization: %s", file_path)

    text = extract_text_from_file(file_path)
    if not text:
        return "No readable text found in the uploaded file."

    summarizer = get_summarizer()

    try:
        # Trim text to fit token limits (~3000 chars)
        text = text[:3000]
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        result = summary[0]["summary_text"] if summary else "Summary could not be generated."
        logger.info("Summarization completed successfully.")
        return result

    except Exception as e:
        logger.exception("Summarization failed.")
        return f"An error occurred during summarization: {e}"

if __name__ == "__main__":
    test_path = "report/Report.PNG"  

    if os.path.exists(test_path):
        print("\n--- Summary ---\n")
        print(summarize_report(test_path))
    else:
        logger.error("File not found: %s", test_path)
        print(f"File not found: {test_path}")
