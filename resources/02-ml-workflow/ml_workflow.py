import os
import mlflow
import mlflow.sklearn
from prefect import task, flow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from PIL import Image
import numpy as np

DATA_DIR = "dataset"
MODEL_FILE = "model.joblib"
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "AppleOrangeClassifier"
REGISTERED_MODEL_NAME = "AppleOrangeClassifier"
IMAGE_SIZE = (64, 64)

def image_to_features(path):
    """Converteer afbeelding naar eenvoudige features (flattened RGB)"""
    img = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.array(img).flatten()

def load_image_dataset():
    """Laad alle afbeeldingen en labels"""
    X, y = [], []
    classes = {"apples": 0, "oranges": 1}
    for label_name, label in classes.items():
        for split in ["train", "val", "test"]:
            folder = os.path.join(DATA_DIR, split, label_name)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    X.append(image_to_features(os.path.join(folder, fname)))
                    y.append(label)
    return np.array(X), np.array(y)

@task
def load_data_task():
    X, y = load_image_dataset()
    print(f"Data geladen: {len(X)} afbeeldingen")
    return X, y

@task
def preprocess_data_task(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print("Data gepreprocessed!")
    return X_train, y_train, X_val, y_val, X_test, y_test

@task
def train_model_task(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    dump(model, MODEL_FILE)
    print("Model lokaal opgeslagen!")
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
    print("Model geregistreerd in MLflow!")
    return MODEL_FILE, model

@task
def evaluate_model_task(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc}")

@flow(name="ML Workflow")
def ml_workflow_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.autolog()
    
    X, y = load_data_task()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data_task(X, y)
    model_file, model = train_model_task(X_train, y_train, X_val, y_val)
    evaluate_model_task(model, X_test, y_test)

if __name__ == "__main__":
    ml_workflow_flow()
