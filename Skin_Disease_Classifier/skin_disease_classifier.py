"""
Skin Disease Classifier Module

This module handles:
- Loading the fine-tuned YOLOv11s model
- Saving uploaded skin images
- Running classification
- Returning predictions with class probabilities
"""

import os
import gdown
import logging
from pathlib import Path
import pandas as pd
from ultralytics import YOLO



MODEL_PATH = "Skin_Disease_Classifier/model/yolo11s_best_model.pt"
GOOGLE_DRIVE_ID = "1hMuswhlLydskPf8MdPI61400GUAoA3y7"

# Download the model if it doesn't already exist
if not os.path.exists(MODEL_PATH):
    print(" Downloading YOLOv11s model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


# Define base and log paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_PATH = os.path.join(BASE_DIR, "model", "yolo11s_best_model.pt")
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "skin_disease_classifier.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

# Global model cache
_model = None


def load_model():
    """
    Load YOLOv11s model from disk (once per session).

    Returns:
        YOLO: Loaded model instance.
    """
    global _model
    if _model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
            logging.info("Loading YOLOv11s model from: %s", MODEL_PATH)
            _model = YOLO(MODEL_PATH)
        except Exception as e:
            logging.error("Model loading failed: %s", str(e))
            raise RuntimeError("Failed to load YOLOv11s model.") from e
    return _model


def save_uploaded_image(uploaded_file) -> str:
    """
    Save uploaded image to a temporary directory.

    Args:
        uploaded_file: Streamlit uploaded file object.

    Returns:
        str: Absolute path to saved image.
    """
    try:
        temp_dir = Path(BASE_DIR) / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logging.info("Image uploaded and saved at: %s", temp_path)
        return str(temp_path)

    except Exception as e:
        logging.error("Failed to save uploaded image: %s", str(e))
        raise RuntimeError("Image saving failed.") from e


def predict(image_path: str) -> pd.DataFrame:
    """
    Perform classification on a skin image using YOLOv11s.

    Args:
        image_path (str): Absolute path to the input image.

    Returns:
        pd.DataFrame: Sorted DataFrame with 'class' and 'confidence' columns.
    """
    try:
        model = load_model()
        logging.info("Running classification on: %s", image_path)
        results = model.predict(image_path)

        if not results or results[0].probs is None:
            logging.warning("No classification result returned.")
            return pd.DataFrame()

        class_names = list(model.names.values())
        confidences = results[0].probs.data.tolist()

        df = pd.DataFrame({
            "class": class_names,
            "confidence": [round(float(c), 4) for c in confidences]
        }).sort_values(by="confidence", ascending=False).reset_index(drop=True)

        logging.info("Prediction result: %s", df.to_dict(orient="records"))
        return df

    except Exception as e:
        logging.error("Prediction failed: %s", str(e))
        raise RuntimeError("Skin disease classification failed.") from e


if __name__ == "__main__":
    test_image = os.path.join(BASE_DIR, "test_images", "BCC4.jpeg")
    if not os.path.exists(test_image):
        print("Test image not found:", test_image)
    else:
        print("Running test on:", test_image)
        try:
            df_result = predict(test_image)
            print(df_result)
        except Exception as err:
            print("Error during test prediction:", err)
