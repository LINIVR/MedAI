"""
Skin Disease Classifier Module for MEDAI

Handles:
- Loading a fine-tuned YOLOv11s model
- Saving uploaded skin images
- Performing image-based classification
- Returning prediction results


"""

import os
import sys
from pathlib import Path
from typing import Union

import pandas as pd
import gdown
from ultralytics import YOLO



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medai_logger import get_logger


logger = get_logger("skin_disease_classifier")


MODEL_FILENAME = "yolo11s_best_model.pt"
MODEL_PATH = Path(__file__).resolve().parent / "model" / MODEL_FILENAME
GDRIVE_ID = "1hMuswhlLydskPf8MdPI61400GUAoA3y7"


_model = None


def download_model():
    """
    Downloads the YOLOv11s model from Google Drive if not found locally.
    """
    if not MODEL_PATH.exists():
        logger.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(MODEL_PATH), quiet=False)
        logger.info("Model download complete.")


def load_model() -> YOLO:
    """
    Loads and caches the YOLOv11s model.

    Returns:
        YOLO: Loaded model instance
    """
    global _model
    if _model is None:
        download_model()
        try:
            logger.info("Loading YOLO model from: %s", MODEL_PATH)
            _model = YOLO(str(MODEL_PATH))
        except Exception as e:
            logger.exception("Model loading failed.")
            raise RuntimeError("Failed to load YOLOv11s model") from e
    return _model


def save_uploaded_image(uploaded_file) -> str:
    """
    Saves uploaded image to a temporary folder.

    Args:
        uploaded_file: File-like object from Streamlit

    Returns:
        str: Path to the saved image
    """
    try:
        temp_dir = Path(__file__).resolve().parent / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info("Image saved at: %s", temp_path)
        return str(temp_path)

    except Exception as e:
        logger.exception("Failed to save uploaded image.")
        raise RuntimeError("Image saving failed") from e


def predict(image_path: Union[str, Path]) -> pd.DataFrame:
    """
    Runs inference on the input image using YOLOv11s.

    Args:
        image_path (str | Path): Path to the image

    Returns:
        pd.DataFrame: Prediction results with 'class' and 'confidence'
    """
    try:
        model = load_model()
        logger.info("Running prediction on: %s", image_path)
        results = model.predict(source=str(image_path))

        if not results or results[0].probs is None:
            logger.warning("No predictions found.")
            return pd.DataFrame()

        class_names = list(model.names.values())
        confidences = results[0].probs.data.tolist()

        df = pd.DataFrame({
            "class": class_names,
            "confidence": [round(float(c), 4) for c in confidences]
        }).sort_values(by="confidence", ascending=False).reset_index(drop=True)

        logger.info("Prediction complete: %s", df.to_dict(orient="records"))
        return df

    except Exception as e:
        logger.exception("Prediction failed.")
        raise RuntimeError("Skin disease prediction failed") from e


if __name__ == "__main__":
    test_img = Path(__file__).resolve().parent / "test_images" / "BCC4.jpeg"
    if not test_img.exists():
        print("Test image not found:", test_img)
    else:
        print("Running test prediction on:", test_img)
        try:
            output = predict(test_img)
            print(output)
        except Exception as err:
            print("Error:", err)
