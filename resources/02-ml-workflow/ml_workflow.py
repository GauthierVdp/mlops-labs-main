from prefect import task, flow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import os
import mlflow
import mlflow.keras
import psutil

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fruit-classification"

DATASET_DIR = os.path.join(os.getcwd(), "resources", "02-ml-workflow", "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
EPOCHS = 15

@task
def preprocess_data():
    # Data augmentation for training to improve generalization
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation / test should only be rescaled
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

@task(cache_key_fn=None)
def train_model(train_gen, val_gen):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # callbacks: checkpoint best model and early stopping
    checkpoint = ModelCheckpoint('fruit_model.h5', save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=[checkpoint, earlystop])

    # If checkpoint saved a best model, load it back so downstream tasks use the best weights
    if os.path.exists('fruit_model.h5'):
        try:
            print('Loading best model from fruit_model.h5')
            model = tf.keras.models.load_model('fruit_model.h5')
        except Exception as e:
            print(f'Could not reload fruit_model.h5 after training: {e}')
    else:
        # ensure we save at least the current model
        model.save('fruit_model.h5')

    return model

@task
def evaluate_model(model, test_gen):
    loss, acc = model.evaluate(test_gen)
    print(f"Test Loss: {loss}, Test Accuracy: {acc}")

@flow
def ml_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.enable_system_metrics_logging()
    mlflow.autolog()

    with mlflow.start_run():
        mlflow.log_params({'batch_size': BATCH_SIZE, 'image_size': IMAGE_SIZE})
        train_gen, val_gen, test_gen = preprocess_data()
        print("Datasets prepared!")
        model = train_model(train_gen, val_gen)
        evaluate_model(model, test_gen)
        # Ensure the trained model is logged as an MLflow artifact at path 'model'
        try:
            mlflow.keras.log_model(model, artifact_path="model")
            print("Logged model artifact to MLflow at artifact_path='model'")
        except Exception as e:
            print(f"Warning: failed to log model artifact to MLflow: {e}")
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        mlflow.log_metric("CPU_usage", cpu)
        mlflow.log_metric("RAM_usage", ram)
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        try:
            mv = mlflow.register_model(model_uri, "FruitClassifier")
            print(f"Registered model: {mv.name}, version: {mv.version}")
        except Exception as e:
            print(f"Model registration failed: {e}")

if __name__ == "__main__":
    ml_pipeline()
