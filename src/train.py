from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .data import load_npz, ECGWindowDataset
from .model import ECGConvNet


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/mitbih_windows.npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/ecg_cnn.pt")
    args = parser.parse_args()

    set_seed(42)

    d = load_npz(args.data)
    X_train, X_val, y_train, y_val = train_test_split(
        d.X, d.y, test_size=0.2, random_state=42, stratify=d.y
    )

    train_ds = ECGWindowDataset(X_train, y_train)
    val_ds = ECGWindowDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cpu")
    model = ECGConvNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[train] device={device} epochs={args.epochs} train={len(train_ds)} val={len(val_ds)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(p)
                trues.append(y.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        acc = accuracy_score(trues, preds)
        print(f"[train] epoch={epoch} loss={np.mean(train_losses):.4f} val_acc={acc:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, out_path)
    print(f"[train] saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()