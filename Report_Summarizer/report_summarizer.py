"""
Report Summarizer Module for MEDAI

Extracts text from PDF or image files (JPG, PNG)
and summarizes it using a Hugging Face transformer.

Model Used: sshleifer/distilbart-cnn-12-6
"""

import os
import logging
from transformers import pipeline
from PyPDF2 import PdfReader
from pytesseract import image_to_string
from PIL import Image

# Constants
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
LOG_FILE = "logs/report_summarizer.log"

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger("report_summarizer")

# Internal summarizer cache
_summarizer = None


def get_summarizer():
    """
    Loads and caches the Hugging Face summarization pipeline.

    Returns:
        pipeline: A summarization pipeline.
    """
    global _summarizer
    if _summarizer is None:
        logger.info("Loading summarization model: %s", MODEL_NAME)
        _summarizer = pipeline("summarization", model=MODEL_NAME)
    return _summarizer


def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a PDF or image file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        str: Extracted text.
    """
    try:
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info("Extracted text from PDF (%d chars)", len(text))
            return text.strip()

        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            text = image_to_string(Image.open(file_path))
            logger.info("Extracted text from image (%d chars)", len(text))
            return text.strip()

        else:
            logger.warning("Unsupported file type: %s", file_path)
            return ""

    except Exception as e:
        logger.error("Failed to extract text: %s", str(e))
        return ""


def summarize_report(file_path: str) -> str:
    """
    Summarizes the content of a medical report (PDF or image).

    Args:
        file_path (str): Path to the input file.

    Returns:
        str: A summarized version of the extracted text.
    """
    try:
        logger.info("Summarizing report: %s", file_path)
        summarizer = get_summarizer()
        text = extract_text_from_file(file_path)

        if not text:
            return "No text found in the provided file."

        # Hugging Face models usually accept <= 1024 tokens; trim if needed
        text = text[:3000]

        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        summary_text = summary[0]["summary_text"] if summary else "Summary could not be generated."

        logger.info("Summarization completed.")
        return summary_text

    except Exception as e:
        logger.exception("Summarization failed.")
        return f"An error occurred during summarization: {e}"


if __name__ == "__main__":
    print(f"Using summarization model: {MODEL_NAME}")
    file_path = "report/Report.PNG"  # Update path as needed

    if os.path.exists(file_path):
        summary = summarize_report(file_path)
        print("\n--- Summary ---\n")
        print(summary)
    else:
        print(f"File not found: {file_path}")
        logger.error("File not found: %s", file_path)
