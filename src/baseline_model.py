# src/baseline_model.py
"""
Baseline Centralized Model for FedShield
Train on 70% data, validate on 20%, test on 10%
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV   = "data/splits/val.csv"
TEST_CSV  = "data/splits/test.csv"

OUT_DIR = "outputs/baseline"
EPOCHS = 30
BATCH_SIZE = 256
RANDOM_STATE = 42
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values

    X_val = val_df.drop(columns=['label']).values
    y_val = val_df['label'].values

    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_training(history):
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== TEST METRICS ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Phish"],
                yticklabels=["Legit","Phish"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1], [0,1], '--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()


def main():
    print("[*] Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("[*] Building model...")
    model = build_model(X_train.shape[1])
    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    print("[*] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    print("[*] Saving model...")
    model.save(os.path.join(OUT_DIR, "model.h5"))

    print("[*] Plotting training curves...")
    plot_training(history)

    print("[*] Evaluating on test set...")
    evaluate_model(model, X_test, y_test)

    print("\n[✓] Baseline training completed successfully.")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
