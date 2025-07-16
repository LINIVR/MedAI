"""
unit tests for the Report_Summarizer module in MEDAI.


"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Report_Summarizer.report_summarizer import (
    extract_text_from_file,
    summarize_report,
)


# Create a temporary directory and dummy image for testing

TEMP_DIR = PROJECT_ROOT / "tests" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FAKE_IMG = TEMP_DIR / "sample.png"

# Generate a white image with simple black text "BP 140/90"
canvas = Image.new("RGB", (300, 120), "white")
text_layer = ImageDraw.Draw(canvas)
text_layer.text((10, 50), "BP 140/90 mmHg", fill="black")
canvas.save(FAKE_IMG)

print("\nRunning basic tests for Report_Summarizer...\n")


# 1. Test OCR text extraction from the dummy image

img_text = extract_text_from_file(str(FAKE_IMG))
assert isinstance(img_text, str), "Image OCR did not return a string"
print("✅ Image text extraction passed (length:", len(img_text), ")")


# 2. Test summarization using the same image

summary_text = summarize_report(str(FAKE_IMG))
assert isinstance(summary_text, str) and len(summary_text) > 10, "Summarization failed"
print("✅ Summarization passed (length:", len(summary_text), ")")


# Cleanup temporary files

FAKE_IMG.unlink(missing_ok=True)
TEMP_DIR.rmdir()

print("\nAll tests ran successfully.\n")
