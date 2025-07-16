"""
Unit test for Skin Disease Classifier module in MEDAI

Test Case:
- Upload a dummy image
- Perform YOLOv11s classification
- Verify output structure


"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd

# Set up import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from Skin_Disease_Classifier.skin_disease_classifier import (
    predict,
    save_uploaded_image
)

# Temp test directory
TEMP_DIR = PROJECT_ROOT / "tests" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create dummy test image
TEST_IMAGE_PATH = TEMP_DIR / "test_image.jpg"
img = Image.new("RGB", (224, 224), color="white")
draw = ImageDraw.Draw(img)
draw.text((50, 100), "Skin test", fill="black")
img.save(TEST_IMAGE_PATH)

print("\nRunning test: Skin Disease Classifier\n")

try:
    # Run prediction
    df_result = predict(TEST_IMAGE_PATH)

    assert isinstance(df_result, pd.DataFrame), "Output is not a DataFrame"
    assert "class" in df_result.columns, "Missing 'class' column"
    assert "confidence" in df_result.columns, "Missing 'confidence' column"
    assert not df_result.empty, "Prediction result is empty"

    print(" Prediction output is a valid DataFrame with results")
    print(df_result.head())

except Exception as e:
    print(" Test failed:", str(e))

finally:
    TEST_IMAGE_PATH.unlink(missing_ok=True)
    TEMP_DIR.rmdir()

print("\nTest completed.\n")
