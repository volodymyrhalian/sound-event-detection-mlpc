"""Reference CRNN training on reconstructed mel sequences (not historical raw-audio code)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .constants import CLASS_NAMES
from .models import AudioCRNN


class MelSequenceDataset(Dataset):
    """One recording per item to prevent leakage across its overlapping windows."""
    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        with np.load(self.files[index], allow_pickle=False) as z:
            if "annotations" not in z:
                raise ValueError(f"No annotation array for {self.files[index]}")
            mel = np.asarray(z["melspect_mean"], dtype=np.float32)
            annotations = np.asarray(z["annotations"], dtype=np.float32)
            if mel.ndim != 2 or mel.shape[1] != 128 or annotations.shape[:2] != (len(mel), len(CLASS_NAMES)):
                raise ValueError(f"Bad feature or annotation shape: {self.files[index]}")
            # Non-negative mel-energy inputs; logarithm compresses large dynamic range.
            x = torch.tensor(np.log1p(np.maximum(mel, 0.0)), dtype=torch.float32)
            y = torch.tensor(np.nanmean(annotations, axis=2) >= 0.5, dtype=torch.float32)
        return x, y


def collate_sequences(batch):
    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys_pad = pad_sequence(ys, batch_first=True, padding_value=0.0)
    mask = torch.arange(xs_pad.shape[1])[None, :] < lengths[:, None]
    return xs_pad, ys_pad, mask


def train(files: list[Path], output_dir: Path, epochs: int = 10, batch_size: int = 4,
          lr: float = 1e-3, seed: int = 42) -> dict:
    if len(files) < 5:
        raise ValueError("Need at least 5 labeled recordings")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_files, val_files = train_test_split(sorted(files), test_size=0.2, random_state=seed)
    train_loader = DataLoader(MelSequenceDataset(train_files), batch_size=batch_size,
                              shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(MelSequenceDataset(val_files), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best = -1.0
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss_values = loss_fn(logits, y)
            loss = (loss_values * mask[..., None]).sum() / (mask.sum() * y.shape[-1])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(mask.sum().item())
            count += int(mask.sum().item())
        model.eval()
        true_values, pred_values = [], []
        with torch.no_grad():
            for x, y, mask in val_loader:
                probs = torch.sigmoid(model(x.to(device))).cpu()
                true_values.append(y[mask].numpy())
                pred_values.append((probs[mask] >= 0.5).numpy())
        truth = np.concatenate(true_values)
        pred = np.concatenate(pred_values)
        score = float(f1_score(truth, pred, average="macro", zero_division=0))
        record = {"epoch": epoch, "train_loss": total / max(count, 1), "val_macro_f1": score}
        history.append(record)
        print(f"Epoch {epoch:03d}: train_loss={record['train_loss']:.4f}, val_macro_f1={score:.4f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), output_dir / "best_crnn.pt")
    config = {
        "model": "AudioCRNN", "feature_input": "reconstructed melspect_mean (128 bins)",
        "note": "New reference training run; not the original report's raw-audio CRNN or score",
        "num_train_recordings": len(train_files), "num_val_recordings": len(val_files),
        "best_val_macro_f1": best, "epochs": epochs, "seed": seed,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/crnn"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    files = sorted(args.features_dir.glob("*.npz"))
    print(train(files, args.output_dir, args.epochs, args.batch_size, args.lr))


if __name__ == "__main__":
    main()
