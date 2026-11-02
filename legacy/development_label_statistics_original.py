import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FOLDER = "audio_features/"

def extract_label_characteristics():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".npz")]

    class_names_ref = None
    class_event_counts = None
    class_total_duration = None

    for filename in files:
        path = os.path.join(INPUT_FOLDER, filename)
        data = dict(np.load(path, allow_pickle=True))

        ann = data["annotations"]          # [T, C, A]
        class_names = list(data["class_names"])
        start_time = data["start_time"]
        end_time = data["end_time"]

        ann_bin = (ann > 0.5).astype(int)
        final_labels = (np.mean(ann_bin, axis=2) >= 0.5).astype(int)   # [T, C]

        if class_names_ref is None:
            class_names_ref = class_names
            C = len(class_names)
            class_event_counts = np.zeros(C, dtype=np.int64)
            class_total_duration = np.zeros(C, dtype=np.float64)

        for c in range(len(class_names)):
            y = final_labels[:, c]

            in_event = False
            event_start = None

            for t in range(len(y)):
                if y[t] == 1 and not in_event:
                    in_event = True
                    event_start = start_time[t]

                if y[t] == 0 and in_event:
                    in_event = False
                    event_end = end_time[t - 1]
                    class_event_counts[c] += 1
                    class_total_duration[c] += (event_end - event_start)

            if in_event:
                event_end = end_time[len(y) - 1]
                class_event_counts[c] += 1
                class_total_duration[c] += (event_end - event_start)

    avg_event_duration = np.divide(
        class_total_duration,
        class_event_counts,
        out=np.zeros_like(class_total_duration),
        where=class_event_counts > 0
    )

    df = pd.DataFrame({
        "class_name": class_names_ref,
        "event_count": class_event_counts,
        "total_duration_sec": class_total_duration,
        "avg_duration_sec": avg_event_duration
    })

    return df


df = extract_label_characteristics()

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Graph 1
axes[0].bar(df["class_name"], df["event_count"])
axes[0].set_ylabel("Number of events")
axes[0].set_title("Class frequency in derived labels")
axes[0].tick_params(axis="x", rotation=45)

# Graph 2
axes[1].bar(df["class_name"], df["avg_duration_sec"])
axes[1].set_ylabel("Average duration (s)")
axes[1].set_title("Average event duration by class")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("label_characteristics_combined.png", dpi=300, bbox_inches="tight")
plt.close()

print(df.sort_values("event_count", ascending=False))