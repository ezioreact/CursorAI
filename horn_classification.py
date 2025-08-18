#!/usr/bin/env python3
"""
Horn Sound Classification (Good vs Bad)

This script:
- Loads audio files from a directory structure: data/good, data/bad (configurable via --data_dir)
- Extracts features: MFCC (incl. deltas), spectral centroid, bandwidth, rolloff, contrast,
  zero-crossing rate, SPL (Sound Pressure Level; relative), tonic (fundamental frequency),
  and THD (Total Harmonic Distortion)
- Normalizes features (per feature row using training-set statistics)
- Splits into train/validation/test with stratification
- Defines and trains a CNN on the 2D time-frequency feature maps
- Plots training curves and evaluates on the test set (accuracy, classification report)
- Optionally predicts Good/Bad for a new audio file

Usage example:
  python horn_classification.py --data_dir ./data --epochs 30 \
    --predict ./data/good/example.wav

Expected directory layout:
  data/
    good/*.wav|*.mp3|*.flac
    bad/*.wav|*.mp3|*.flac

Dependencies: see requirements.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# librosa is widely used for audio feature extraction
import librosa
import librosa.display  # noqa: F401 (for potential future visualization)


# ------------------------------
# Reproducibility Configuration
# ------------------------------
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Enable GPU memory growth if available (safer defaults for desktop GPUs)
def enable_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not set memory growth on {gpu}: {exc}")


# ------------------------------
# Audio Utilities
# ------------------------------
def load_audio_mono(
    file_path: str,
    target_sr: int,
    target_duration_sec: float,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample to target_sr, convert to mono, and pad/trim to fixed duration.

    Returns:
        y: audio mono float32, length == target_sr * target_duration_sec
        sr: sample rate (target_sr)
    """
    # Using librosa.load for simplicity; soundfile could be used directly then resampled
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    target_length = int(target_sr * target_duration_sec)

    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode="constant")
    elif len(y) > target_length:
        y = y[:target_length]

    return y.astype(np.float32), target_sr


def compute_relative_spl_db(y: np.ndarray, reference_amplitude: float = 1.0) -> float:
    """Compute a relative SPL in dB using digital amplitude as reference.

    Note: True SPL requires calibrated reference pressure. This computes a relative
    measure sufficient for classification: SPL_dB = 20 * log10(rms / reference_amplitude).
    """
    eps = 1e-10
    rms = float(np.sqrt(np.mean(np.square(y)) + eps))
    spl_db = 20.0 * np.log10(rms / (reference_amplitude + eps) + eps)
    return spl_db


def estimate_tonic_f0_hz(y: np.ndarray, sr: int) -> float:
    """Estimate fundamental frequency (tonic) using librosa.pyin; fallback to 0.0 if unavailable.
    Returns median f0 (Hz) across frames.
    """
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        if f0 is None:
            return 0.0
        finite_f0 = f0[np.isfinite(f0)]
        if finite_f0.size == 0:
            return 0.0
        return float(np.nanmedian(finite_f0))
    except Exception:
        return 0.0


def compute_thd(y: np.ndarray, sr: int) -> float:
    """Compute a simple THD estimate from the magnitude spectrum.

    THD = sqrt(sum(harmonics^2)) / fundamental, using up to first 5 harmonics.
    This is a rough estimate for classification; real THD requires more precise measurement.
    """
    # Apply a Hann window to reduce spectral leakage
    window = np.hanning(len(y))
    windowed = y * window

    # Compute real FFT and magnitude spectrum
    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)

    # Limit search range for fundamental
    min_fund_hz = 50.0
    max_fund_hz = min(4000.0, sr / 2.0)
    valid_idx = np.where((freqs >= min_fund_hz) & (freqs <= max_fund_hz))[0]
    if valid_idx.size == 0:
        return 0.0

    peak_idx = valid_idx[np.argmax(magnitudes[valid_idx])]
    fundamental_mag = float(magnitudes[peak_idx])
    fundamental_hz = float(freqs[peak_idx])
    if fundamental_mag <= 1e-12 or fundamental_hz <= 0:
        return 0.0

    harmonic_magnitudes: List[float] = []
    max_harmonics = 5
    for harmonic_k in range(2, max_harmonics + 1):
        harmonic_freq = harmonic_k * fundamental_hz
        if harmonic_freq >= sr / 2.0:
            break
        harmonic_idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
        harmonic_magnitudes.append(float(magnitudes[harmonic_idx]))

    if not harmonic_magnitudes:
        return 0.0

    thd = float(np.sqrt(np.sum(np.square(harmonic_magnitudes))) / fundamental_mag)
    return thd


# ------------------------------
# Feature Extraction
# ------------------------------
@dataclass
class FeatureConfig:
    sample_rate: int = 22050
    duration_sec: float = 3.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20
    rolloff_percent: float = 0.85


def extract_feature_matrix(y: np.ndarray, sr: int, cfg: FeatureConfig) -> np.ndarray:
    """Extract a stacked 2D feature matrix of shape (num_features, num_frames).

    Stacked rows include:
      - MFCC (n_mfcc)
      - MFCC delta (n_mfcc)
      - MFCC delta-delta (n_mfcc)
      - Spectral centroid (1)
      - Spectral bandwidth (1)
      - Spectral rolloff (1)
      - Spectral contrast (7)
      - Zero-crossing rate (1)
      - SPL (1, tiled across frames)
      - Tonic f0 (1, tiled across frames)
      - THD (1, tiled across frames)
    """
    # Compute shared STFT-based features using consistent hop_length
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        roll_percent=cfg.rolloff_percent,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=y, hop_length=cfg.hop_length
    )

    # Scalars (one per file)
    spl_db = compute_relative_spl_db(y)
    tonic_f0_hz = estimate_tonic_f0_hz(y, sr)
    thd_ratio = compute_thd(y, sr)

    # Determine time axis length to tile scalars
    num_frames = mfcc.shape[1]
    tiled_spl = np.full((1, num_frames), spl_db, dtype=np.float32)
    tiled_tonic = np.full((1, num_frames), tonic_f0_hz, dtype=np.float32)
    tiled_thd = np.full((1, num_frames), thd_ratio, dtype=np.float32)

    # Stack all features
    features_list = [
        mfcc,
        mfcc_delta,
        mfcc_delta2,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_contrast,
        zero_crossing_rate,
        tiled_spl,
        tiled_tonic,
        tiled_thd,
    ]

    feature_matrix = np.vstack(features_list).astype(np.float32)
    return feature_matrix


def load_dataset_features(
    data_dir: str,
    cfg: FeatureConfig,
    allowed_exts: Tuple[str, ...] = (".wav", ".mp3", ".flac"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Scan data_dir for class subfolders ('good', 'bad'), extract features.

    Returns:
        X: np.ndarray of shape (num_samples, num_features, num_frames)
        y: np.ndarray of shape (num_samples,) with labels {0: bad, 1: good}
        file_paths: list of file paths corresponding to X/y
    """
    class_to_label = {"bad": 0, "good": 1}
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    file_paths: List[str] = []

    for class_name, label in class_to_label.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: class directory not found: {class_dir}")
            continue

        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in allowed_exts:
                continue
            try:
                y_audio, sr = load_audio_mono(str(path), cfg.sample_rate, cfg.duration_sec)
                feat = extract_feature_matrix(y_audio, sr, cfg)
                all_features.append(feat)
                all_labels.append(label)
                file_paths.append(str(path))
            except Exception as exc:
                print(f"Failed to process {path}: {exc}")

    if not all_features:
        raise RuntimeError(
            f"No audio files found in {data_dir}. Expected subfolders 'good' and 'bad'."
        )

    # Ensure consistent time dimension across all (should already be consistent due to fixed duration)
    # But pad/truncate just in case any variation occurred due to edge cases.
    num_frames_list = [f.shape[1] for f in all_features]
    target_frames = int(np.median(num_frames_list))
    processed_features: List[np.ndarray] = []
    for f in all_features:
        if f.shape[1] < target_frames:
            pad_width = target_frames - f.shape[1]
            f_padded = np.pad(f, ((0, 0), (0, pad_width)), mode="constant")
            processed_features.append(f_padded)
        else:
            processed_features.append(f[:, :target_frames])

    X = np.stack(processed_features, axis=0)  # (N, F, T)
    y = np.array(all_labels, dtype=np.int64)
    return X, y, file_paths


# ------------------------------
# Normalization Utilities
# ------------------------------
@dataclass
class NormalizationStats:
    feature_mean: np.ndarray  # shape (num_features, 1)
    feature_std: np.ndarray   # shape (num_features, 1)
    target_num_frames: int


def compute_normalization_stats(X_train: np.ndarray) -> NormalizationStats:
    """Compute mean/std per feature row across all training samples and time frames."""
    # X_train shape: (N, F, T)
    feature_means = X_train.mean(axis=(0, 2), keepdims=True).squeeze(axis=0)  # (F, 1)
    feature_stds = X_train.std(axis=(0, 2), keepdims=True).squeeze(axis=0)    # (F, 1)

    # Avoid division by zero
    feature_stds[feature_stds < 1e-8] = 1.0
    target_num_frames = int(X_train.shape[2])
    return NormalizationStats(
        feature_mean=feature_means, feature_std=feature_stds, target_num_frames=target_num_frames
    )


def apply_normalization(X: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    """Apply per-feature standardization using precomputed stats."""
    # Expand dims to broadcast over batch and time dims
    mean = stats.feature_mean[:, np.newaxis]  # (F, 1)
    std = stats.feature_std[:, np.newaxis]    # (F, 1)
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32)


# ------------------------------
# Model Definition
# ------------------------------
def build_cnn_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """Build a simple CNN for 2D time-frequency inputs (F x T x C)."""
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=None)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="horn_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------------
# Training Utilities
# ------------------------------
def compute_class_weights_from_labels(y_train: np.ndarray) -> dict:
    classes = np.array([0, 1], dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def plot_training_history(history: tf.keras.callbacks.History, out_dir: str) -> None:
    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["accuracy"], label="Train Acc")
    plt.plot(epochs, metrics["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved training curves to: {plot_path}")


# ------------------------------
# Prediction Utility
# ------------------------------
def prepare_single_file(
    file_path: str,
    cfg: FeatureConfig,
    norm_stats: NormalizationStats,
) -> np.ndarray:
    y_audio, sr = load_audio_mono(file_path, cfg.sample_rate, cfg.duration_sec)
    feat = extract_feature_matrix(y_audio, sr, cfg)
    # Ensure time dimension matches training expectation via median frames from config
    # Since duration is fixed, frames should match. Still, add a safety resize path.
    feature_frames = feat.shape[1]
    expected_frames = int(norm_stats.target_num_frames)
    if feature_frames < expected_frames:
        pad = expected_frames - feature_frames
        feat = np.pad(feat, ((0, 0), (0, pad)), mode="constant")
    elif feature_frames > expected_frames:
        feat = feat[:, :expected_frames]

    feat_norm = apply_normalization(feat[np.newaxis, ...], norm_stats)  # (1, F, T)
    feat_norm = feat_norm[..., np.newaxis]  # (1, F, T, 1)
    return feat_norm


# ------------------------------
# Main
# ------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Horn sound classification (Good vs Bad)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with 'good' and 'bad' subfolders")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--duration", type=float, default=3.0, help="Target audio duration (seconds)")
    parser.add_argument("--predict", type=str, default="", help="Optional: path to an audio file to predict after training")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save model and figures")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args(argv)

    set_global_seed(args.seed)
    enable_gpu_memory_growth()

    cfg = FeatureConfig(
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
    )

    print("Loading dataset and extracting features... This may take a moment.")
    X, y, paths = load_dataset_features(args.data_dir, cfg)
    num_samples, num_features, num_frames = X.shape
    print(f"Dataset: {num_samples} samples | Features per sample: {num_features} x {num_frames}")
    class_counts = {c: int(np.sum(y == c)) for c in [0, 1]}
    print(f"Class distribution -> bad: {class_counts.get(0, 0)}, good: {class_counts.get(1, 0)}")

    # Train/Val/Test split (e.g., 70/15/15)
    X_temp, X_test, y_temp, y_test, paths_temp, paths_test = train_test_split(
        X, y, paths, test_size=0.15, random_state=args.seed, stratify=y
    )
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X_temp, y_temp, paths_temp, test_size=0.1765, random_state=args.seed, stratify=y_temp
    )  # 0.1765 of 0.85 ~= 0.15 overall

    # Compute normalization stats on training set and apply to all splits
    norm_stats = compute_normalization_stats(X_train)
    X_train = apply_normalization(X_train, norm_stats)
    X_val = apply_normalization(X_val, norm_stats)
    X_test = apply_normalization(X_test, norm_stats)

    # Add channel dimension for Conv2D
    X_train_4d = X_train[..., np.newaxis]
    X_val_4d = X_val[..., np.newaxis]
    X_test_4d = X_test[..., np.newaxis]

    # Compute class weights to address imbalance (e.g., 100 good vs 50 bad)
    class_weight = compute_class_weights_from_labels(y_train)
    print(f"Class weights: {class_weight}")

    # Build and train model
    input_shape = (num_features, num_frames, 1)
    model = build_cnn_model(input_shape)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.out_dir, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train_4d,
        y_train,
        validation_data=(X_val_4d, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=callbacks,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "final_model.keras")
    model.save(model_path)
    print(f"Saved final model to: {model_path}")

    # Persist normalization stats
    stats_path = os.path.join(args.out_dir, "normalization_stats.json")
    stats_payload = {
        "feature_mean": norm_stats.feature_mean.squeeze().tolist(),
        "feature_std": norm_stats.feature_std.squeeze().tolist(),
        "target_num_frames": int(norm_stats.target_num_frames),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f)
    print(f"Saved normalization stats to: {stats_path}")

    # Plot training curves
    plot_training_history(history, args.out_dir)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_pred_prob = model.predict(X_test_4d)
    test_pred = (test_pred_prob.flatten() >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report:")
    target_names = ["bad", "good"]
    print(classification_report(y_test, test_pred, target_names=target_names, digits=4))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, test_pred))

    # Optional prediction on a new file
    if args.predict:
        print(f"\nPredicting on: {args.predict}")
        # Reload stats in case of separate run; we already have them in-memory though
        feat_norm = prepare_single_file(args.predict, cfg, norm_stats)
        pred_prob = float(model.predict(feat_norm)[0][0])
        pred_label = int(pred_prob >= 0.5)
        pred_class = "good" if pred_label == 1 else "bad"
        print(f"Predicted: {pred_class} (prob={pred_prob:.3f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

