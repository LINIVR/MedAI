"""
Skin Disease Classifier

Loads and runs a fine-tuned YOLOv11s model for skin disease detection.
Assumes the model is located in 'Project/Skin_Disease_Classifier/model/yolov11s.pt'.
Supports image uploads and resizing using OpenCV.
"""

import pandas as pd
import logging
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# Logging setup
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/skin_disease_classifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "yolo11s_best_model.pt")

_model = None

def load_model():
    """
    Load the YOLOv11s model from disk.

    Returns:
        A loaded YOLOv11s model object.
    """
    global _model
    if _model is None:
        try:
            logging.info("Loading YOLOv11s model from: %s", MODEL_PATH)
            _model = YOLO(MODEL_PATH)
        except Exception as e:
            logging.error("Model loading failed: %s", str(e))
            raise RuntimeError("Failed to load YOLOv11s model.") from e
    return _model

def save_and_resize_image(uploaded_file, target_size=(224, 224)) -> str:
    """
    Save uploaded image and resize to the model's input dimensions.

    Args:
        uploaded_file: Uploaded file from Streamlit (BytesIO).
        target_size: Tuple (width, height) for resizing.

    Returns:
        str: Path to the resized image.
    """
    try:
        temp_dir = Path("Skin_Disease_Classifier/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / "input.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = cv2.imread(str(temp_path))
        if img is None:
            raise ValueError("Uploaded image could not be read by OpenCV.")

        resized = cv2.resize(img, target_size)
        cv2.imwrite(str(temp_path), resized)
        logging.info("Image uploaded and resized to: %s", temp_path)
        return str(temp_path)

    except Exception as e:
        logging.error("Failed to process uploaded image: %s", str(e))
        raise RuntimeError("Image preprocessing failed.") from e

def predict(image_path: str) -> pd.DataFrame:
    """
    Run classification on the provided image and return label probabilities.

    Args:
        image_path (str): Full path to the image file.

    Returns:
        pd.DataFrame: Prediction probabilities per class.
    """
    try:
        model = load_model()
        logging.info("Running classification on image: %s", image_path)
        results = model.predict(image_path)

        # Check for classification output
        probs = results[0].probs
        if probs is None:
            logging.warning("No classification result returned.")
            return pd.DataFrame()

        # Convert to pandas DataFrame
        class_names = list(model.names.values())
        scores = probs.data.tolist()
        df = pd.DataFrame({"class": class_names, "confidence": scores})
        df["confidence"] = df["confidence"].apply(lambda x: round(float(x), 4))
        df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)

        logging.info("Predicted classes: %s", df.to_dict(orient="records"))
        return df

    except Exception as e:
        logging.error("Classification failed: %s", str(e))
        raise RuntimeError("Skin disease classification failed.") from e

if __name__ == "__main__":
    test_img_path = "test_images/BCC4.jpeg"
    if not os.path.exists(test_img_path):
        print("Image not found at:", test_img_path)
    else:
        try:
            result_df = predict(test_img_path)
            print("Prediction Results:\n", result_df)
        except Exception as error:
            print("Error:", error)
