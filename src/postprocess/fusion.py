"""Feature-level fusion: nối vector tương tác của M1 và M2 để huấn luyện head MLP mới."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def build_fusion_features(
    ha1: np.ndarray, hb1: np.ndarray, ha2: np.ndarray, hb2: np.ndarray
) -> np.ndarray:
    """v = [|hA1-hB1|, hA1*hB1, |hA2-hB2|, hA2*hB2] -> 4*d chiều."""
    parts = [
        np.abs(ha1 - hb1),
        ha1 * hb1,
        np.abs(ha2 - hb2),
        ha2 * hb2,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


class FusionHead(nn.Module):
    def __init__(self, in_features: int, hidden: int = 768, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_fusion_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[FusionHead, dict[str, Any]]:
    """Huấn luyện head trên feature đã trích (encoder freeze)."""
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)

    model = FusionHead(X_train.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True
    )

    best_state = None
    best_val_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv.to(device))
            val_loss = float(nn.BCEWithLogitsLoss()(val_logits, torch.tensor(y_val, dtype=torch.float32, device=device)))
        history.append({"epoch": epoch, "train_loss": round(total / len(Xt), 6), "val_loss": round(val_loss, 6)})
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"history": history, "best_val_loss": round(best_val_loss, 6)}
