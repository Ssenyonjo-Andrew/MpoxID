"""
Deep learning comparison models for MPXV clade classification.

These models are comparison candidates in the full training pipeline. They are
implemented to be CPU-feasible by default through a reduced representation that
samples fixed windows from long genomes and one-hot encodes A/C/G/T/N.

If PyTorch is unavailable, the training code reports the backend as unavailable
and skips DL model fitting rather than failing the whole pipeline.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None
    HAS_TORCH = False


DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class SequenceWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class ConvNet1D(nn.Module):
    def __init__(self, n_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        z = self.features(x)
        return self.head(z)


class ConvBiLSTM(nn.Module):
    def __init__(self, n_classes: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        z = self.conv(x)
        z = z.transpose(1, 2)
        out, _ = self.lstm(z)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


@dataclass
class DLArtifacts:
    model_name: str
    backend: str
    history: Dict[str, List[float]]
    training_time_seconds: float
    inference_time_per_sequence_ms: float
    state_dict_path: Optional[str]
    skipped_reason: Optional[str] = None


class DeepSequenceEncoder:
    def __init__(self, n_chunks: int = 32, chunk_length: int = 128):
        self.n_chunks = int(n_chunks)
        self.chunk_length = int(chunk_length)
        self.total_length = self.n_chunks * self.chunk_length

    def _sample_positions(self, seq_len: int) -> List[Tuple[int, int]]:
        if seq_len <= self.total_length:
            return [(0, min(seq_len, self.total_length))]
        starts = np.linspace(0, max(0, seq_len - self.chunk_length), num=self.n_chunks, dtype=int)
        return [(int(s), int(s + self.chunk_length)) for s in starts]

    def encode_sequence(self, sequence: str) -> np.ndarray:
        seq = (sequence or "").upper()
        canvas = np.full((5, self.total_length), 0.0, dtype=np.float32)
        write_pos = 0
        for start, end in self._sample_positions(len(seq)):
            chunk = seq[start:end]
            for base in chunk[: self.chunk_length]:
                idx = DNA_TO_INDEX.get(base, 4)
                canvas[idx, write_pos] = 1.0
                write_pos += 1
                if write_pos >= self.total_length:
                    break
            if write_pos >= self.total_length:
                break
        while write_pos < self.total_length:
            canvas[4, write_pos] = 1.0
            write_pos += 1
        return canvas

    def transform(self, sequences: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode_sequence(seq) for seq in sequences], axis=0)


class TorchSequenceClassifier:
    def __init__(self, architecture: str, n_classes: int, seed: int = 42, hidden_dim: int = 128, dropout: float = 0.3):
        self.architecture = architecture
        self.n_classes = n_classes
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = "cpu"
        self.model = None
        self.classes_ = None
        self.artifacts: Optional[DLArtifacts] = None
        if HAS_TORCH:
            set_global_seed(seed)
            self.model = self._build_model()
            self.model.to(self.device)

    def _build_model(self):
        if self.architecture == "cnn_1d":
            return ConvNet1D(self.n_classes, hidden_dim=self.hidden_dim, dropout=self.dropout)
        if self.architecture == "cnn_bilstm":
            return ConvBiLSTM(self.n_classes, hidden_dim=self.hidden_dim, dropout=self.dropout)
        raise ValueError(f"Unknown architecture: {self.architecture}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        class_weights: Optional[np.ndarray] = None,
        *,
        batch_size: int = 16,
        max_epochs: int = 20,
        learning_rate: float = 1e-3,
        patience: int = 4,
        out_dir: Optional[Path] = None,
    ) -> "TorchSequenceClassifier":
        if not HAS_TORCH or self.model is None:
            self.artifacts = DLArtifacts(
                model_name=self.architecture,
                backend="unavailable",
                history={},
                training_time_seconds=0.0,
                inference_time_per_sequence_ms=0.0,
                state_dict_path=None,
                skipped_reason="PyTorch is not installed",
            )
            return self

        train_loader = DataLoader(SequenceWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(SequenceWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val = float("inf")
        best_state = None
        epochs_without_improve = 0
        start = time.perf_counter()

        for _epoch in range(max_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(xb)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == yb).sum().item())
                total_seen += len(xb)

            train_loss = total_loss / max(1, total_seen)
            train_acc = total_correct / max(1, total_seen)

            self.model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_seen = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += float(loss.item()) * len(xb)
                    preds = logits.argmax(dim=1)
                    val_correct += int((preds == yb).sum().item())
                    val_seen += len(xb)

            val_loss = val_loss_sum / max(1, val_seen)
            val_acc = val_correct / max(1, val_seen)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    break

        training_time = time.perf_counter() - start
        if best_state is not None:
            self.model.load_state_dict(best_state)

        state_path = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            state_path = str(out_dir / f"{self.architecture}.pt")
            torch.save(self.model.state_dict(), state_path)

        infer_start = time.perf_counter()
        _ = self.predict_proba(X_val[: min(len(X_val), 8)])
        infer_elapsed = time.perf_counter() - infer_start
        per_seq_ms = (infer_elapsed / max(1, min(len(X_val), 8))) * 1000.0

        self.artifacts = DLArtifacts(
            model_name=self.architecture,
            backend="torch",
            history=history,
            training_time_seconds=training_time,
            inference_time_per_sequence_ms=per_seq_ms,
            state_dict_path=state_path,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not HAS_TORCH or self.model is None:
            raise RuntimeError("PyTorch backend unavailable")
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
