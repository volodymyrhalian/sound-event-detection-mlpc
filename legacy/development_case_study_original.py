import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. PATHS
# ============================================================

prepared_path = Path("prepared_dataset.npz")
model_path = Path("best_nn_model.pt")
features_dir = Path("MLPC2026_dataset_development/audio_features")

output_dir = Path("case_study_outputs")
output_dir.mkdir(exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

# IMPORTANT:
# If the neural network was trained on raw X_train, use False.
# If it was trained on X_train_scaled, use True.
#
# From your previous notebook, it looks like the model was trained on X_train,
# so False is probably correct.
USE_SCALED_FEATURES = False

THRESHOLD = 0.5


# ============================================================
# 3. LOAD PREPARED DATA
# ============================================================

prep_data = np.load(prepared_path, allow_pickle=True)

if USE_SCALED_FEATURES:
    X_test = prep_data["X_test_scaled"]
else:
    X_test = prep_data["X_test"]

Y_test = prep_data["Y_test"]
filenames_test = prep_data["filenames_test"]
class_names = prep_data["class_names"]

print("====================================")
print("Loaded prepared dataset")
print("====================================")
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
print("Number of test recordings:", len(np.unique(filenames_test)))
print("Classes:", class_names)
print()


# ============================================================
# 4. DEFINE MODEL ARCHITECTURE
# ============================================================

class AudioMLP(nn.Module):
    def __init__(
        self,
        input_dim=32,
        hidden_dims=[128, 256, 128, 32],
        output_dim=15,
        dropout=0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioMLP(
    input_dim=32,
    hidden_dims=[128, 256, 128, 32],
    output_dim=len(class_names),
    dropout=0.3
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("====================================")
print("Model loaded successfully")
print("====================================")
print()


# ============================================================
# 5. PREDICTION FUNCTION
# ============================================================

def predict(model, X, batch_size=512, threshold=0.5):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_probs = []

    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)

    probs = np.vstack(all_probs)
    preds = (probs >= threshold).astype(int)

    return preds, probs


Y_pred_test, Y_prob_test = predict(
    model,
    X_test,
    threshold=THRESHOLD
)

print("Y_pred_test:", Y_pred_test.shape)
print()


# ============================================================
# 6. RECORDING-LEVEL SCORES
# ============================================================

def recording_metrics(y_true_file, y_pred_file):
    tp = np.logical_and(y_true_file == 1, y_pred_file == 1).sum()
    fp = np.logical_and(y_true_file == 0, y_pred_file == 1).sum()
    fn = np.logical_and(y_true_file == 1, y_pred_file == 0).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    true_active = y_true_file.sum()
    pred_active = y_pred_file.sum()
    errors = fp + fn

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "errors": errors,
        "true_active": true_active,
        "pred_active": pred_active,
        "segments": len(y_true_file)
    }


def get_recording_candidates(filenames, Y_true, Y_pred):
    candidates = []

    for file_id in np.unique(filenames):
        idx = np.where(filenames == file_id)[0]

        y_true_file = Y_true[idx]
        y_pred_file = Y_pred[idx]

        metrics = recording_metrics(y_true_file, y_pred_file)

        # skip almost empty recordings because they are not interesting for case study
        if metrics["true_active"] < 10:
            continue

        candidates.append({
            "file_id": file_id,
            **metrics
        })

    return candidates


candidates = get_recording_candidates(
    filenames_test,
    Y_test,
    Y_pred_test
)


# ============================================================
# 7. CHOOSE ONE GOOD AND ONE DIFFICULT FILE
# ============================================================

# Good example: highest F1 score
good_candidates = sorted(
    candidates,
    key=lambda x: x["f1"],
    reverse=True
)

# Difficult example: many errors, but also has real activity
difficult_candidates = sorted(
    candidates,
    key=lambda x: x["errors"],
    reverse=True
)

good_file = good_candidates[0]["file_id"]
difficult_file = difficult_candidates[0]["file_id"]

selected_files = [good_file, difficult_file]

print("====================================")
print("Selected files")
print("====================================")

print("Good example:")
print(
    f"file={good_candidates[0]['file_id']}, "
    f"F1={good_candidates[0]['f1']:.3f}, "
    f"precision={good_candidates[0]['precision']:.3f}, "
    f"recall={good_candidates[0]['recall']:.3f}, "
    f"TP={good_candidates[0]['tp']}, "
    f"FP={good_candidates[0]['fp']}, "
    f"FN={good_candidates[0]['fn']}, "
    f"true_active={good_candidates[0]['true_active']}, "
    f"pred_active={good_candidates[0]['pred_active']}, "
    f"segments={good_candidates[0]['segments']}"
)

print()

print("Difficult example:")
print(
    f"file={difficult_candidates[0]['file_id']}, "
    f"F1={difficult_candidates[0]['f1']:.3f}, "
    f"precision={difficult_candidates[0]['precision']:.3f}, "
    f"recall={difficult_candidates[0]['recall']:.3f}, "
    f"TP={difficult_candidates[0]['tp']}, "
    f"FP={difficult_candidates[0]['fp']}, "
    f"FN={difficult_candidates[0]['fn']}, "
    f"true_active={difficult_candidates[0]['true_active']}, "
    f"pred_active={difficult_candidates[0]['pred_active']}, "
    f"segments={difficult_candidates[0]['segments']}"
)

print()
print("Selected files:", selected_files)
print()


# ============================================================
# 8. PLOT CASE STUDY
# ============================================================

def plot_case_study(file_id, label):
    """
    Creates one figure:
    1. Mel spectrogram
    2. Expected labels
    3. Predicted labels
    """

    idx = np.where(filenames_test == file_id)[0]

    true_labels = Y_test[idx]
    pred_labels = Y_pred_test[idx]

    npz_path = features_dir / f"{file_id}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find file: {npz_path}")

    data = dict(np.load(npz_path, allow_pickle=True))

    start_time = data["start_time"]
    end_time = data["end_time"]

    if "melspect_mean" in data:
        spectrogram = data["melspect_mean"].T
    else:
        raise KeyError("melspect_mean not found in the .npz file.")

    # Ensure everything has the same length
    T = min(
        len(start_time),
        len(end_time),
        true_labels.shape[0],
        pred_labels.shape[0],
        spectrogram.shape[1]
    )

    start_time = start_time[:T]
    end_time = end_time[:T]
    true_labels = true_labels[:T]
    pred_labels = pred_labels[:T]
    spectrogram = spectrogram[:, :T]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )

    # Spectrogram
    axes[0].imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=[start_time[0], end_time[-1], 0, spectrogram.shape[0]]
    )
    axes[0].set_title(f"{label}: Recording {file_id} - Mel Spectrogram")
    axes[0].set_ylabel("Mel bins")

    # Expected labels
    axes[1].imshow(
        true_labels.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[start_time[0], end_time[-1], 0, len(class_names)]
    )
    axes[1].set_title("Expected labels")
    axes[1].set_ylabel("Classes")

    # Predicted labels
    axes[2].imshow(
        pred_labels.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[start_time[0], end_time[-1], 0, len(class_names)]
    )
    axes[2].set_title("Predicted labels")
    axes[2].set_ylabel("Classes")
    axes[2].set_xlabel("Time (s)")

    for ax in axes[1:]:
        ax.set_yticks(np.arange(len(class_names)) + 0.5)
        ax.set_yticklabels(class_names, fontsize=7)

    plt.tight_layout()

    output_path = output_dir / f"case_study_{label.lower()}_{file_id}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", output_path)

    # Print summary for report writing
    print()
    print(f"Summary for {label} recording {file_id}:")

    for i, class_name in enumerate(class_names):
        true_count = int(true_labels[:, i].sum())
        pred_count = int(pred_labels[:, i].sum())

        if true_count > 0 or pred_count > 0:
            print(
                f"{class_name}: "
                f"true active segments = {true_count}, "
                f"predicted active segments = {pred_count}"
            )

    print()


# ============================================================
# 9. GENERATE FIGURES
# ============================================================

plot_case_study(good_file, "Good example")
plot_case_study(difficult_file, "Difficult example")


print("====================================")
print("DONE")
print("====================================")
print("Figures saved in:", output_dir)