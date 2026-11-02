"""Reconstructed PyTorch training CLI for archived prepared MLPC arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import mlp_from_preset, PRESETS


def load_data(path: Path, *, scaled: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        sets = {}
        for split in ("train", "val", "test"):
            key = f"X_{split}_scaled" if scaled else f"X_{split}"
            sets[split] = (np.asarray(z[key], dtype=np.float32), np.asarray(z[f"Y_{split}"], dtype=np.float32))
        return sets


def macro_f1(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> float:
    return float(f1_score(y, (prob >= threshold).astype(int), average="macro", zero_division=0))


def train(dataset: Path, preset: str, epochs: int, output_dir: Path,
          batch_size: int = 128, lr: float = 1e-3, scaled: bool = False, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_data(dataset, scaled=scaled)
    x_train, y_train = splits["train"]
    if x_train.shape[1] != PRESETS[preset]["input_dim"]:
        raise ValueError(f"Preset {preset} expects {PRESETS[preset]['input_dim']} features; got {x_train.shape[1]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlp_from_preset(preset, output_dim=y_train.shape[1]).to(device)
    pos = y_train.sum(axis=0)
    neg = len(y_train)-pos
    weight = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 50), dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
    val_x, val_y = splits["val"]
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
                            batch_size=batch_size, shuffle=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = float("-inf")
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(x)
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in val_loader:
                probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        val_prob = np.concatenate(probs)
        score = macro_f1(val_y, val_prob)
        loss = total_loss/len(x_train)
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_f1": score})
        print(f"Epoch {epoch:03d}: train_loss={loss:.4f} val_macro_f1={score:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), output_dir / "best_mlp.pt")
    config = {"preset": preset, "scaled": scaled, "threshold": 0.5, "best_val_macro_f1": best,
              "epochs": epochs, "input_dim": int(x_train.shape[1]), "n_classes": int(y_train.shape[1]),
              "note": "New reconstruction run; not the original reported result"}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="development")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scaled", action="store_true", help="Use saved standardized features")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    print(train(args.dataset, args.preset, args.epochs, args.output_dir, args.batch_size, args.lr, args.scaled))


if __name__ == "__main__":
    main()
