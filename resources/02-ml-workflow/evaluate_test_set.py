import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


DATASET_DIR = os.path.join(os.getcwd(), "resources", "02-ml-workflow", "dataset")
TEST_DIR = os.path.join(DATASET_DIR, "test")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32


def find_local_h5(limit_parents=3):
    p = os.path.abspath(__file__)
    # walk up from script path
    path = os.path.dirname(p)
    for _ in range(limit_parents + 1):
        candidate = os.path.join(path, "fruit_model.h5")
        if os.path.exists(candidate):
            return candidate
        path = os.path.dirname(path)
    # check CWD
    cwd_candidate = os.path.join(os.getcwd(), "fruit_model.h5")
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return None


def main():
    model_path = find_local_h5()
    if model_path is None:
        print("No local fruit_model.h5 found. Exiting.")
        return
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)

    print("Class indices:", gen.class_indices)

    # Predictions
    preds = model.predict(gen, verbose=1)
    probs = np.asarray(preds).ravel()
    pred_labels = (probs >= 0.5).astype(int)
    true_labels = gen.classes

    # Compute simple metrics
    total = len(true_labels)
    correct = (pred_labels == true_labels).sum()
    acc = correct / total if total else 0.0

    # confusion: rows true (0,1), cols pred (0,1)
    cm = np.zeros((2,2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    print(f"Total test samples: {total}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # Show top misclassified examples (by highest confidence)
    mis_idx = np.where(pred_labels != true_labels)[0]
    if mis_idx.size:
        # get filepaths from generator
        filepaths = np.array([os.path.join(TEST_DIR, path) for path in gen.filepaths])
        mis_probs = probs[mis_idx]
        # sort by descending confidence
        order = np.argsort(-np.abs(mis_probs - 0.5))
        print(f"Top {min(10, len(mis_idx))} misclassified examples:")
        for i in order[:10]:
            idx = mis_idx[i]
            print(f"{filepaths[idx]} | true={true_labels[idx]} pred={pred_labels[idx]} prob={probs[idx]:.4f}")
    else:
        print("No misclassifications found.")


if __name__ == '__main__':
    main()
