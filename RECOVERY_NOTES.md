# Repository reconstruction notes

This repository combines (a) files recovered from the owner's MLPC and Pattern challenge archives, and (b) **new, assistant-reconstructed code** implementing an illustrative replacement for deleted components. Do not claim that reconstructed scripts were the exact original submitted implementation.

## Preserved from the original archives
- `notebooks/*_original.ipynb` — available notebook source, with execution outputs stripped for a clean repository. Some cells depend on datasets that were not uploaded. Course-provided tutorial material may also be present: see attribution below.
- `legacy/*_original.py` — original preprocessing, case study and evaluator scripts, preserved unmodified.
- `docs/reports/*.pdf` — supplied project reports, coauthored by Oleksandr Kudriavchenko and Volodymyr Halian.
- `docs/figures/` — selected original figures.
- `checkpoints/original_*.pt` — existing MLP weight-only checkpoints from archives. Their architectures can be loaded with `sed_portfolio.models`; performance requires the exact corresponding feature generation process and data split.

## Reconstructed / newly implemented
- `src/sed_portfolio/features.py`: computes 14 feature families with four aggregations each. It preserves feature *names*, anticipated shapes and window/hop configuration, but is an approximation of the lost original feature extractor. It does not recover deleted source waveforms.
- `src/sed_portfolio/prepare.py`: rebuilds recording-disjoint train/val/test splits from NPZ feature/annotation arrays.
- `src/sed_portfolio/models.py`: MLP presets based on the notebooks and report plus a CRNN **reference implementation** informed by the report architecture. It is not the missing original CRNN code.
- `src/sed_portfolio/train_mlp.py`, `src/sed_portfolio/train_crnn.py`, `src/sed_portfolio/predict_mlp.py`: training/inference CLIs for the prepared data and features, not audited reproductions of original submission code.
- `src/sed_portfolio/temporal.py`: event-interval smoothing and segment-to-event conversion.
- `tests/*`: small synthetic unit tests; these validate program behavior, not reported model performance.

## Missing inputs / limitations
- The original raw development/challenge waveforms and the `train/validation/test/audio_features/*.npz` dataset files are not in the provided archives. The screenshot proves they existed on the user's PC previously but does not supply their data.
- `prepared_dataset.npz` is present in the uploaded `mlpc.zip` but intentionally excluded from this distributable GitHub repository. It can be used locally.
- No original CRNN training script, original BEATs Strong fine-tuning implementation or BEATs checkpoint was supplied. The **0.6000** and **0.6420** Macro-F1 scores are report-stated results, not independently reproduced by the generated code.
- Experiment notebooks may contain hard-coded paths, out-of-order cells and dependencies on missing datasets. They are preserved as historical evidence, not guaranteed single-click executable pipelines.

## Attribution
The reports identify the project as the work of **Team Unsupervised Predictors — Oleksandr Kudriavchenko and Volodymyr Halian**. Course templates/tutorial notebooks and provided evaluation code should be credited to their original maintainers where known. Seek collaborator/course permission before selecting a license or public release if required.

## Safety and privacy
The original metadata contains free-text descriptions, annotator IDs and recordings from contributors. We did **not** include raw audio, metadata or full annotation tables in this public-facing package. The `.gitignore` blocks those data formats by default. Review the contents and dataset conditions before uploading any material to GitHub.
