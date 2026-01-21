"""
Baseline Centralized Model for FedShield
Train on 70% data, validate on 20%, test on 10%

Enhanced with:
- Better model architecture (BatchNorm)
- Model checkpointing (saves best model)
- More comprehensive metrics
- JSON results export
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------- CONFIG ----------------
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV   = "data/splits/val.csv"
TEST_CSV  = "data/splits/test.csv"

OUT_DIR = "outputs/baseline"
MODEL_PATH = os.path.join(OUT_DIR, "best_model.h5")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")

EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
RANDOM_STATE = 42
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data():
    """Load train/val/test splits from CSV files."""
    print("[*] Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=['label']).values
    y_train = train_df['label'].values

    X_val = val_df.drop(columns=['label']).values
    y_val = val_df['label'].values

    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values
    
    print(f"[+] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"    Train labels: {np.bincount(y_train)}")
    print(f"    Val labels:   {np.bincount(y_val)}")
    print(f"    Test labels:  {np.bincount(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(input_dim):
    """
    Build MLP with BatchNorm for better convergence.
    Architecture: 128 -> 64 -> 32 -> 1
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def plot_training(history):
    """Plot training curves: loss, accuracy, AUC."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # AUC
    axes[2].plot(history.history['auc'], label='Train')
    axes[2].plot(history.history['val_auc'], label='Val')
    axes[2].set_title("AUC Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[+] Saved training curves -> {OUT_DIR}/training_curves.png")


def evaluate_model(model, X_test, y_test):
    """Comprehensive evaluation on test set."""
    print("\n[*] Evaluating on test set...")
    
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
        print("[!] Warning: Could not compute ROC-AUC")

    # Print results
    print("\n" + "="*50)
    print("BASELINE MODEL TEST RESULTS")
    print("="*50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=["Legitimate", "Phishing"],
                                zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Phish"],
                yticklabels=["Legit","Phish"],
                cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Baseline Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[+] Saved confusion matrix -> {OUT_DIR}/confusion_matrix.png")

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", linewidth=2, color='darkorange')
        plt.plot([0,1], [0,1], '--', color='navy', linewidth=2, label='Random')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Baseline Model")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=150)
        plt.close()
        print(f"[+] Saved ROC curve -> {OUT_DIR}/roc_curve.png")
    except Exception as e:
        print(f"[!] Could not plot ROC curve: {e}")
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'test_samples': int(len(y_test)),
        'confusion_matrix': cm.tolist()
    }


def save_results(results, history):
    """Save all results to JSON file."""
    output = {
        'test_metrics': results,
        'final_train_metrics': {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'auc': float(history.history['auc'][-1])
        },
        'final_val_metrics': {
            'loss': float(history.history['val_loss'][-1]),
            'accuracy': float(history.history['val_accuracy'][-1]),
            'auc': float(history.history['val_auc'][-1])
        },
        'training_info': {
            'epochs_trained': len(history.history['loss']),
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"[+] Saved results to {RESULTS_JSON}")


def main():
    print("\n" + "="*60)
    print("FEDSHIELD - BASELINE CENTRALIZED MODEL")
    print("="*60 + "\n")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Build model
    print("\n[*] Building model...")
    model = build_model(X_train.shape[1])
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        MODEL_PATH,
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Train
    print("\n[*] Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

    # Load best model
    print(f"\n[*] Loading best model from {MODEL_PATH}...")
    best_model = tf.keras.models.load_model(MODEL_PATH)

    # Plot training curves
    print("\n[*] Plotting training curves...")
    plot_training(history)

    # Evaluate
    results = evaluate_model(best_model, X_test, y_test)
    
    # Save results
    save_results(results, history)

    print("\n" + "="*60)
    print("[✓] Baseline training completed successfully!")
    print("="*60)
    print(f"\nOutputs saved to: {OUT_DIR}/")
    print(f"  - best_model.h5")
    print(f"  - training_curves.png")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - results.json")
    print("\nNext steps:")
    print("  1. Review metrics and visualizations")
    print("  2. Run EDA: python src/eda.py")
    print("  3. Proceed to Federated Learning baseline")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()