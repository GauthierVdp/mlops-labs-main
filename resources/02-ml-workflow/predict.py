import mlflow
import mlflow.sklearn
from PIL import Image
import numpy as np
import os

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "AppleOrangeClassifier" 
IMAGE_SIZE = (64, 64) 
CLASS_MAPPING = {0: "apple", 1: "orange"} 

APPLE_IMAGE_PATH = "dataset/val/apples/apple8.jpeg"
ORANGE_IMAGE_PATH = "dataset/val/oranges/orange8.jpeg"


def load_image(path):
    """Laad afbeelding en converteer naar numpy array (flattened RGB)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bestand niet gevonden: {path}")
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.array(img).flatten()

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/latest"
    print(f"Laden model van URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    X_new = np.array([
        load_image(APPLE_IMAGE_PATH),
        load_image(ORANGE_IMAGE_PATH)
    ])

    preds = model.predict(X_new)
    print("Voorspellingen:")
    for path, pred in zip([APPLE_IMAGE_PATH, ORANGE_IMAGE_PATH], preds):
        label_name = CLASS_MAPPING.get(pred, f"unknown({pred})")
        print(f"{path} -> {label_name}")

if __name__ == "__main__":
    main()
