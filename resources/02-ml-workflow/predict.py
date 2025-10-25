import argparse
import os
import sys
from pathlib import Path
import shutil
import tempfile

import mlflow
import mlflow.pyfunc
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests

MLFLOW_TRACKING_URI = "http://localhost:5000"
DEFAULT_MODEL_NAME = "FruitClassifier"

DEFAULT_APPLE_URL = "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"
DEFAULT_ORANGE_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg"

IMAGE_SIZE = (128, 128)


def download_image(url: str, dst: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, stream=True, timeout=30, headers=headers)
    resp.raise_for_status()
    with open(dst, "wb") as fh:
        shutil.copyfileobj(resp.raw, fh)


def load_and_preprocess(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_model_with_fallback(model_uri: str):
    """Try pyfunc first (most generic), then keras flavor as fallback."""
    def find_local_h5(limit_parents=3):
        p = Path(__file__).resolve()
        for i in range(limit_parents + 1):
            candidate = p.parents[i] / "fruit_model.h5"
            if candidate.exists():
                return str(candidate)
        cwd_candidate = Path(os.getcwd()) / "fruit_model.h5"
        if cwd_candidate.exists():
            return str(cwd_candidate)
        return None

    h5_path = find_local_h5()
    if h5_path is not None:
        try:
            print(f"Found local Keras H5 model at {h5_path} — loading with tf.keras.models.load_model...")
            keras_model = tf.keras.models.load_model(h5_path)
            print("Loaded local H5 Keras model")
            return keras_model
        except Exception as e_h5:
            print(f"Failed to load local H5 model at {h5_path}: {e_h5} — continuing to mlflow loaders...")
    e_pyfunc = None
    e_keras = None
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print("Loaded model via mlflow.pyfunc")
        return model
    except Exception as exc:
        e_pyfunc = exc
        print(f"pyfunc load failed: {e_pyfunc}; trying keras loader...")
    try:
        model = mlflow.keras.load_model(model_uri)
        print("Loaded model via mlflow.keras")
        return model
    except Exception as exc:
        e_keras = exc
    def find_local_model_artifact():
        candidates = [
            os.path.join(os.getcwd(), "resources", "02-ml-workflow", "mlruns"),
            os.path.join(os.getcwd(), "mlruns"),
        ]
        for base in candidates:
            if not os.path.exists(base):
                continue
            for exp in os.listdir(base):
                exp_path = os.path.join(base, exp)
                if not os.path.isdir(exp_path):
                    continue
                for run in os.listdir(exp_path):
                    run_path = os.path.join(exp_path, run)
                    if not os.path.isdir(run_path):
                        continue
                    possible = os.path.join(run_path, "artifacts", "model")
                    if os.path.exists(possible):
                        return possible
        return None

    local_model_path = find_local_model_artifact()
    if local_model_path:
        try:
            print(f"Attempting to load model from local artifact path: {local_model_path}")
            try:
                model = mlflow.pyfunc.load_model(local_model_path)
                print("Loaded local model via mlflow.pyfunc")
                return model
            except Exception as e_local_py:
                print(f"local pyfunc load failed: {e_local_py}; trying keras loader...")
            model = mlflow.keras.load_model(local_model_path)
            print("Loaded local model via mlflow.keras")
            return model
        except Exception as e_local:
            e_keras = e_local if e_keras is None else e_keras
    msg_parts = []
    if e_pyfunc is not None:
        msg_parts.append(f"pyfunc error: {e_pyfunc}")
    if e_keras is not None:
        msg_parts.append(f"keras error: {e_keras}")
    if local_model_path is None:
        msg_parts.append("no local mlruns/model artifact found in project")
    raise RuntimeError("Failed to load model; " + " ; ".join(msg_parts))


def predict_and_print(model, img_array, label_name="item"):
    try:
        preds = model.predict(img_array)
    except Exception:
        try:
            preds = model.predict(np.vstack([img_array]))
        except Exception as e:
            raise

    if isinstance(preds, np.ndarray):
        val = float(np.asarray(preds).ravel()[0])
    else:
        print(f"{label_name} prediction (raw): {preds}")
        return

    print(f"{label_name} prediction (probability of class 1): {val:.4f}")
    cls = "class 1 (positive)" if val >= 0.5 else "class 0 (negative)"
    print(f"Interpreted as: {cls} (threshold 0.5)")


def main(argv):
    p = argparse.ArgumentParser(description="Classify a single image as apple or orange")
    p.add_argument("--image", required=True, help="Path or URL to the image to classify")
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    args = p.parse_args(argv)

    mlflow.set_tracking_uri(args.tracking_uri)
    model_uri = f"models:/{args.model_name}/latest"

    img_input = args.image
    with tempfile.TemporaryDirectory() as tmpdir:
        if img_input.startswith("http://") or img_input.startswith("https://"):
            img_path = os.path.join(tmpdir, "image.jpg")
            download_image(img_input, img_path)
        else:
            img_path = img_input

        if not os.path.exists(img_path):
            repo_img = os.path.join(os.getcwd(), "resources", "02-ml-workflow", "sample_apple.jpg")
            if os.path.exists(repo_img):
                print("Provided image not found; falling back to repository sample_apple.jpg")
                img_path = repo_img
            else:
                raise FileNotFoundError(f"Image not found: {img_input}")

        try:
            model = load_model_with_fallback(model_uri)
        except Exception as e:
            print(f"ERROR loading model {model_uri}: {e}")
            sys.exit(2)

        img_array = load_and_preprocess(img_path)
        try:
            preds = model.predict(img_array)
        except Exception:
            preds = model.predict(np.vstack([img_array]))
        prob = float(np.asarray(preds).ravel()[0])
        label = "orange" if prob >= 0.5 else "apple"
        print(f"Prediction: {label} (probability of orange={prob:.4f})")


if __name__ == "__main__":
    main(sys.argv[1:])
