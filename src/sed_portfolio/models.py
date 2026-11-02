"""Reconstructed neural architectures based on surviving student code and report."""
from __future__ import annotations
import torch
from torch import nn

PRESETS: dict[str, dict] = {
    "development": {"input_dim": 32, "hidden_dims": (128, 256, 128, 32), "dropout": 0.3},
    "challenge": {"input_dim": 960, "hidden_dims": (256, 128, 64), "dropout": 0.0},
}


class AudioMLP(nn.Module):
    """Multi-label MLP that mirrors the surviving BatchNorm/ReLU/Dropout layout."""
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], output_dim: int = 15, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(previous, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)))
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp_from_preset(name: str, output_dim: int = 15) -> AudioMLP:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}")
    return AudioMLP(output_dim=output_dim, **PRESETS[name])


class AudioCRNN(nn.Module):
    """Reference CRNN (not original training code) based on report's 32/64/64 conv + GRU128.

    Accepts mel-feature sequences (B, T, 128), output (B, T, classes). Unlike the
    report's original raw-audio workflow this operates on precomputed mel features.
    """
    def __init__(self, n_mels: int = 128, n_classes: int = 15, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        if n_mels % 8 != 0:
            raise ValueError("n_mels must be divisible by 8")
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((1, 2)),
        )
        self.gru = nn.GRU(input_size=64 * (n_mels // 8), hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError("Expected (B, T, mel_bins)")
        encoded = self.encoder(mel.unsqueeze(1))
        b, c, t, f = encoded.shape
        sequence = encoded.permute(0, 2, 1, 3).reshape(b, t, c*f)
        hidden, _ = self.gru(sequence)
        return self.classifier(self.dropout(hidden))
