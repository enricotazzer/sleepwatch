"""Bidirectional GRU that stages a whole night from its sequence of epoch features.

Inputs are the epoch features (heavy-tailed motion features log-transformed), robust-scaled with
statistics from the training subjects, with missing values set to 0 plus two availability flags
(heart rate present, motion present; the same information the trees get from NaNs). Unknown
labels are ignored in the loss. Training stops early on validation macro-F1 measured on held-out
*training* subjects, and the best epoch's weights are kept. Runs on CPU, which is faster than MPS
for this model size and deterministic.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sleepwatch.constants import UNKNOWN
from sleepwatch.models.hgb import macro_f1

LOG_PREFIXES = ("activity", "enmo", "mag_std", "angle_change")
IGNORE = -100
N_CLASSES = 5


class GRUConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Size chosen on fold 0's validation subjects (training data only): one 48-unit layer
    # matched two 64-unit layers (val macro-F1 0.536 vs 0.539) at about half the cost.
    hidden: int = 48
    layers: int = 1
    dropout: float = 0.2
    input_dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 40
    patience: int = 6
    clip: float = 1.0
    threads: int = 4


class Preprocessor:
    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.log = np.array([c.startswith(LOG_PREFIXES) for c in self.columns])

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.columns].to_numpy(dtype=float, copy=True)
        x[:, self.log] = np.log10(np.clip(x[:, self.log], 1e-5, None))
        return x

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        x = self._raw(df)
        self.center = np.nan_to_num(np.nanmedian(x, axis=0))
        q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
        scale = np.nan_to_num(q75 - q25)
        self.scale = np.where(scale > 1e-9, scale, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        z = np.clip((self._raw(df) - self.center) / self.scale, -10, 10)
        flags = np.column_stack(
            [np.isfinite(df[c].to_numpy(dtype=float)) for c in ("hr_mean", "activity")]
        )
        return np.hstack([np.nan_to_num(z), flags]).astype(np.float32)

    @property
    def n_inputs(self) -> int:
        return len(self.columns) + 2


class Net(nn.Module):
    def __init__(self, n_inputs: int, cfg: GRUConfig):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Dropout(cfg.input_dropout), nn.Linear(n_inputs, cfg.hidden), nn.GELU()
        )
        self.gru = nn.GRU(
            cfg.hidden,
            cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.out = nn.Linear(2 * cfg.hidden, N_CLASSES)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.inp(x), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=x.shape[1])
        return self.out(hidden)


def night_arrays(df: pd.DataFrame, prep: Preprocessor, target: str | None):
    """One ``(key, features, labels)`` item per night, in row order.

    ``df`` must hold each night's epochs in one contiguous, epoch-ordered block (as produced by
    ``data.prepare``), so concatenating the per-night predictions restores the row order.
    """
    codes = df.groupby(["subject", "night"], sort=False).ngroup().to_numpy()
    if (np.diff(codes) < 0).any() or (np.diff(codes) > 1).any():
        raise ValueError("each night's epochs must form one contiguous block")
    starts = np.flatnonzero(np.r_[True, np.diff(codes) > 0])
    stops = np.r_[starts[1:], len(df)]
    x_all = prep.transform(df)
    if target is None:
        y_all = np.full(len(df), IGNORE)
    else:
        y_all = df[target].to_numpy().astype(np.int64)
        y_all = np.where(y_all == UNKNOWN, IGNORE, y_all)
    subjects, nights = df["subject"].to_numpy(), df["night"].to_numpy()
    return [
        ((subjects[a], nights[a]), x_all[a:b], y_all[a:b])
        for a, b in zip(starts, stops, strict=True)
    ]


def _pad(items):
    lengths = torch.tensor([len(x) for _, x, _ in items])
    t = int(lengths.max())
    x = torch.zeros(len(items), t, items[0][1].shape[1])
    y = torch.full((len(items), t), IGNORE, dtype=torch.long)
    for i, (_, xi, yi) in enumerate(items):
        x[i, : len(xi)] = torch.from_numpy(xi)
        y[i, : len(yi)] = torch.from_numpy(yi)
    return x, y, lengths


@torch.no_grad()
def predict(model: Net, items, batch_size: int = 16) -> list[np.ndarray]:
    """Class probabilities ``(epochs, 5)`` for each night, in input order."""
    model.eval()
    out = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        x, _, lengths = _pad(batch)
        proba = torch.softmax(model(x, lengths), dim=-1).numpy()
        out.extend(proba[i, : int(n)] for i, n in enumerate(lengths))
    return out


def _validation_f1(model, items) -> float:
    probas = predict(model, items)
    y = np.concatenate([yi for _, _, yi in items])
    y_hat = np.concatenate([p.argmax(axis=1) for p in probas])
    keep = y != IGNORE
    return macro_f1(y[keep], y_hat[keep])


def train(train_items, val_items, n_inputs: int, cfg: GRUConfig, seed: int):
    """Train with early stopping on validation macro-F1; returns ``(model, history)``."""
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(n_inputs, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best = (-1.0, 0, copy.deepcopy(model.state_dict()))
    epochs_run = 0
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(len(train_items))
        for start in range(0, len(order), cfg.batch_size):
            x, y, lengths = _pad([train_items[i] for i in order[start : start + cfg.batch_size]])
            optimizer.zero_grad()
            loss = loss_fn(model(x, lengths).reshape(-1, N_CLASSES), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
        epochs_run = epoch + 1
        score = _validation_f1(model, val_items)
        if score > best[0]:
            best = (score, epoch + 1, copy.deepcopy(model.state_dict()))
        elif epoch + 1 - best[1] >= cfg.patience:
            break
    model.load_state_dict(best[2])
    return model, {"val_macro_f1": best[0], "best_epoch": best[1], "epochs_run": epochs_run}
