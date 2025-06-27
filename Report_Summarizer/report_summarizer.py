"""
Report Summarizer Module

Extracts text from PDF or image files and summarizes it using a Hugging Face transformer model.

"""

import os
import logging
from transformers import pipeline
from PyPDF2 import PdfReader
from pytesseract import image_to_string
from PIL import Image

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/report_summarizer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger("report_summarizer")


def summarize_report(file_path: str) -> str:
    """
    Summarizes the content of a blood report or scan report from an image or PDF file
    using a Hugging Face transformer summarization pipeline.

    Args:
        file_path (str): The path to the image or PDF file to be summarized.

    Returns:
        str: The summarized text output, or an error message if summarization fails.
    """
    try:
        logger.info("Starting summarization for file: %s", file_path)
        summarizer = pipeline("summarization")

        # Extract text from file
        if file_path.lower().endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            logger.info("Extracted text from PDF: %d characters", len(text))
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            text = image_to_string(Image.open(file_path))
            logger.info("Extracted text from image: %d characters", len(text))
        else:
            logger.error("Unsupported file type: %s", file_path)
            return "Unsupported file type. Please upload a PDF or an image file."

        if not text.strip():
            logger.warning("No text found in the provided file: %s", file_path)
            return "No text found in the provided file."

        # Summarize the extracted text
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        summary_text = summary[0]['summary_text'] if summary else "Summary could not be generated."
        logger.info("Summarization successful for file: %s", file_path)
        return summary_text

    except Exception as e:
        logger.error("Summarization failed for file %s: %s", file_path, str(e))
        return f"An error occurred during summarization: {e}"