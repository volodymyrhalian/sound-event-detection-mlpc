# Local dataset directory

**No university dataset, personal scene descriptions, audio recordings, annotator IDs or extracted features are included in this public-facing repository.**

Place downloaded/licensed raw audio and annotations locally, for example:

```
data/raw_audio/000001.wav
...
data/annotations.csv
```

To recreate compatible-in-shape (but not numerically identical) features:

```bash
python -m sed_portfolio.features --audio-dir data/raw_audio --annotations data/annotations.csv --output-dir data/audio_features
```

The reconstructed extractor uses 32 ms frames, 1 second summary windows, and 0.5 second hops. Original extractor implementation and some configuration details were not preserved. Thus new feature arrays should **not** be treated as numerically identical to the university-provided ones, and old checkpoints may perform poorly on them.

The archived `mlpc.zip` also contains a `prepared_outputs/prepared_dataset.npz` (not included here). Copy it to `data/prepared_dataset.npz` to train on original prepared features **without** reconstructing the original audio data.
