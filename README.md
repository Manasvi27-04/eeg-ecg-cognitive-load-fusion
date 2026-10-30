# Attention-Based Multimodal EEG-ECG Fusion for Driver Cognitive Load Assessment

[![Venue](https://img.shields.io/badge/ACM%20IKDD%20CODS%202026-Accepted%20(Poster)-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

Official repository for **"Attention-Based Multimodal EEG-ECG Fusion for Driver Cognitive Load Assessment"**, accepted as a **poster at ACM IKDD CODS 2026**.

An attention-based deep learning framework that fuses EEG and ECG signals to classify driver cognitive load. It is benchmarked against **16 baselines** (12 unimodal, 4 multimodal fusion) on the public CL-Drive dataset under one shared training and evaluation pipeline. It achieves the highest accuracy in all four settings (binary/ternary x 10-fold/LOSO), with the largest gains on the harder ternary task (up to **+7.20 percentage points** over the strongest baseline).

> **Status:** This repository documents the architecture, methodology and results. The full pipeline, trained configurations and analysis scripts will be released with the camera-ready version of the paper. Check back for updates.

<!-- TODO (camera-ready): add paper link / DOI / BibTeX -->

---

## Motivation

Driver fatigue and elevated cognitive load are major contributors to road accidents. Vision-based monitoring degrades with illumination changes, occlusion and individual facial differences. This project instead combines two complementary physiological signals: **EEG** (cortical activity) and **ECG** (autonomic activity, e.g. heart rate variability). Together they capture more of a driver's state than either alone.

## Dataset

Uses the public **CL-Drive** dataset (Angkan et al., IEEE T-ITS 2024):

- 21 participants in immersive driving simulations (nine three-minute complexity levels)
- **EEG:** Muse S headband, 4 channels (TP9, AF7, AF8, TP10), 256 Hz
- **ECG:** Shimmer device, 3 channels (LA-RA, LL-RA, Vx-RA), 512 Hz
- 3,035 labeled **10-second non-overlapping windows** (2,560 EEG samples and 5,120 ECG samples per window)
- Cognitive-load scores self-reported every 10 s on a 9-point Likert scale:
  - **Binary:** score < 5 is low, >= 5 is high. Class counts: 1,277 / 1,758
  - **Ternary:** 1-3 low, 4-6 medium, 7-9 high. Class counts: 802 / 1,587 / 646

No custom data collection was required. The dataset is not redistributed here; obtain it from the original authors.

## Architecture

```
EEG [4 x 2560 @ 256 Hz] --> Dual-kernel Conv1D (k=5,11) --> InstanceNorm+SiLU --> 9x dilated residual + CBAM --+
                                                                                                              |
                                                                  +--> pool to 128 steps, project to d=64 -----+--> + learnable positional encoding
ECG [3 x 5120 @ 512 Hz] --> Dual-kernel Conv1D (k=3,9)  --> InstanceNorm+SiLU --> 8x dilated residual + CBAM --+            |
                                                                                                                              v
                                                         Per-modality self-attention (4 heads)  -->  S_eeg, S_ecg
                                                                                                                              |
                                                         Bidirectional cross-attention  -->  C_eeg (EEG queries, ECG keys/values)
                                                                                                  C_ecg (ECG queries, EEG keys/values)
                                                                                                                              |
                                                         Z = [S_eeg || S_ecg || C_eeg || C_ecg]   (128 x 256)
                                                                                                                              v
                                                         TCN: 4 dilated residual blocks (d = 1,2,4,8; k = 3)
                                                                                                                              v
                                                         Mean-pool(TCN out) || Mean-pool(Z) --> FC(128) --> SiLU --> Dropout --> softmax
```

Key design choices:

- **Instance Normalization** in the extractors to suppress subject-specific baseline drift (following IBN-Net).
- **CBAM** (Convolutional Block Attention Module) in every dilated residual block: channel attention, then temporal attention.
- **Self-attention + bidirectional cross-attention** as the fusion mechanism; each modality attends to informative regions of the other.
- **TCN** (Temporal Convolutional Network) for temporal modeling of the fused sequence.
- About **2.17M** learnable parameters.

## Training

| Setting | Value |
|---|---|
| Loss | Class-weighted focal loss (gamma = 2.0), label smoothing 0.1, per-fold inverse-frequency class weights |
| Augmentation | Gaussian noise, channel-wise amplitude scaling (0.9 to 1.1) |
| Regularization | Mixup (50% of batches, lambda ~ Beta(0.2, 0.2)) |
| Optimizer | AdamW, lr 1e-3 (halved on validation plateau), weight decay 1e-4 |
| Batch size / precision | 128 / bfloat16 mixed precision |
| Early stopping | Patience 10, max 40 epochs |
| Normalization | Per-subject z-score per modality, statistics from the training subset of each fold |
| Seeds | 2 (results averaged) |
| Hardware | 1x NVIDIA A100 80GB, PyTorch 2.4.1, CUDA 12.1 (about 44 GPU-hours total) |

All 17 methods (16 baselines + proposed) share the same pipeline; only the model constructor differs.

## Evaluation protocol

| Protocol | Description |
|---|---|
| 10-fold CV | Stratified 10-fold cross-validation over 10-second windows (`StratifiedKFold`, fixed `random_state`) |
| LOSO | Leave-one-subject-out, 21 folds; the target driver is never seen in training (deployment case) |

Each protocol is run for binary and ternary classification. Metrics: accuracy and macro-F1. Significance: two-sided paired Wilcoxon signed-rank tests on per-fold accuracy, Holm-Bonferroni corrected within each condition.

> **Note on 10-fold CV:** folds are stratified at the window level, so adjacent windows from the same subject can appear in both training and test folds. 10-fold numbers should therefore be read as an optimistic, within-subject-style estimate. **LOSO is the subject-independent measure.**

## Results

Accuracy (%) on CL-Drive, averaged over two seeds.

| Method | Modality | #Params | Bin. 10F | Bin. LOSO | Ter. 10F | Ter. LOSO |
|---|---|---:|---:|---:|---:|---:|
| ShallowConvNet | EEG | 20,722 | 75.17 | 74.71 | 61.68 | 59.13 |
| DeepConvNet | EEG | 276,827 | 57.83 | 67.51 | 47.18 | 49.52 |
| EEGNet | EEG | 3,730 | 72.16 | 69.28 | 50.20 | 45.11 |
| EEGTCNet | EEG | 3,990 | 69.90 | 66.87 | 39.75 | 38.82 |
| ATCNet | EEG | 107,818 | 76.99 | 72.14 | 57.51 | 46.81 |
| EEG Conformer | EEG | 1,825,666 | 70.70 | 70.98 | 56.54 | 50.76 |
| ShallowConvNet | ECG | 32,722 | 59.37 | 70.27 | 46.70 | 54.86 |
| DeepConvNet | ECG | 288,602 | 56.03 | 75.59 | 47.58 | 57.16 |
| EEGNet | ECG | 6,274 | 57.84 | 63.97 | 37.60 | 40.24 |
| EEGTCNet | ECG | 3,974 | 54.46 | 56.22 | 35.29 | 33.57 |
| ATCNet | ECG | 107,786 | 77.51 | 65.77 | 58.45 | 39.30 |
| EEG Conformer | ECG | 3,564,866 | 54.02 | 54.78 | 34.53 | 41.80 |
| Early fusion (concat) | Both | 57,634 | 81.05 | 75.28 | 61.85 | 47.07 |
| Late fusion (logit avg.) | Both | 113,700 | 84.35 | 75.85 | 66.36 | 52.61 |
| GMU | Both | 121,698 | 84.78 | 76.87 | 67.63 | 54.52 |
| MulT | Both | 422,178 | 80.61 | 75.09 | 70.43 | 53.43 |
| **Proposed** | **Both** | **2,169,712** | **85.24** | **77.72** | **77.63** | **64.25** |

Margin over the best baseline in each column:

| Setting | Proposed | Best baseline | Margin |
|---|---:|---|---:|
| Binary, 10-fold | 85.24 | 84.78 (GMU) | +0.46 |
| Binary, LOSO | 77.72 | 76.87 (GMU) | +0.85 |
| Ternary, 10-fold | 77.63 | 70.43 (MulT) | +7.20 |
| Ternary, LOSO | 64.25 | 59.13 (ShallowConvNet, EEG) | +5.12 |

Gains are significant under the corrected Wilcoxon tests on the ternary task (all 16 baselines on 10-fold; 15 of 16 under LOSO). On the binary task the margins are small.

The full comparison, including **macro-F1**, per-method accuracy deltas and significance stars for all four settings, is in [`baseline_table.txt`](baseline_table.txt).

**Per-subject ternary LOSO:** accuracy ranges from 43.82% to 83.69% (mean 64.79%); the proposed model beats the mean of the best multimodal baseline (GMU) on 19 of 21 held-out subjects. Ternary LOSO macro-F1 is 45.00%.

## Ablation study

Mean accuracy (%), two seeds; delta is the change from the full model. Architectural ablations are parameter-matched to the full model within 5%.

| Configuration | Bin. 10F | Bin. LOSO | Ter. 10F | Ter. LOSO |
|---|---:|---:|---:|---:|
| **Full model** | 85.04 | 78.52 | 77.94 | 65.49 |
| EEG only | 80.63 (-4.41) | 77.63 (-0.89) | 70.33 (-7.61) | 60.22 (-5.28) |
| ECG only | 84.93 (-0.11) | 78.02 (-0.50) | 76.08 (-1.86) | 60.70 (-4.80) |
| Concat (no attention fusion) | 86.01 (+0.97) | 78.17 (-0.36) | 77.33 (-0.61) | 62.99 (-2.51) |
| Self-attention only (no cross-attn) | 85.59 (+0.55) | 78.14 (-0.38) | 76.75 (-1.19) | 64.41 (-1.08) |
| BiGRU (no TCN) | 85.45 (+0.41) | 77.77 (-0.75) | 76.34 (-1.60) | 63.62 (-1.87) |
| No mixup, no class weights | 85.22 (+0.18) | 77.99 (-0.53) | 78.43 (+0.49) | 65.42 (-0.07) |

Takeaways:

- **Both modalities contribute on the ternary task** (removing EEG costs 7.61 / 5.28 points on 10-fold / LOSO; removing ECG costs 1.86 / 4.80).
- **On the binary task, effects are small**; ECG alone is already close to the full model, and concatenation is on par with cross-attention.
- Architectural components (cross-attention, TCN) give modest gains that are significant only in some conditions.
- Removing mixup and class weights leaves accuracy essentially unchanged but lowers ternary-LOSO macro-F1 by 1.44 points (the model over-predicts the majority medium-load class).
- The full-model row differs from Table 1 by under 1.5 points because the two runs used slightly different training schedules.

## Limitations

- **Single dataset, 21 subjects, one lab simulator**; cross-hardware, cross-population and on-road generalization are untested.
- **Model size:** the proposed model (2.17M parameters) is larger than the fusion baselines (58k to 422k), so part of the gain may come from capacity rather than the fusion design alone. Parameter-matched baselines are future work.
- **Absolute ternary LOSO performance is limited** (64.25% accuracy, 45.00% macro-F1).
- **Offline 10-second windows only**; no streaming or low-latency evaluation yet.
- Two seeds limit the power of the significance tests; non-significant differences should be read with caution.

## Roadmap

- [ ] Release training / evaluation pipeline and analysis scripts
- [ ] Release trained configurations (136 runs: 128 baseline + 8 proposed) and dataset fingerprints
- [ ] Group-aware (subject / session) 10-fold splits
- [ ] Parameter-matched fusion baselines
- [ ] Validation on additional public cognitive-load datasets
- [ ] Real-time inference version
- [ ] Additional signals (galvanic skin response, eye gaze)

## Citation

```bibtex
@inproceedings{gupta2026eegecgfusion,
  title     = {Attention-Based Multimodal EEG-ECG Fusion for Driver Cognitive Load Assessment},
  author    = {Gupta, Manasvi and others},
  booktitle = {Proceedings of ACM IKDD CODS 2026},
  year      = {2026},
  note      = {Poster}
}
```

<!-- Replace author list and add pages / DOI once the camera-ready is final. -->

## License

MIT
