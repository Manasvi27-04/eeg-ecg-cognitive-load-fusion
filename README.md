# Attention-Based Multimodal EEG-ECG Fusion for Cognitive Load Assessment

An attention-based deep learning framework that fuses EEG and ECG signals to
classify driver cognitive load (mental workload), benchmarked against 16
state-of-the-art baselines on the public CL-Drive dataset — achieving the
highest accuracy across both binary and ternary classification, outperforming
the strongest baseline by up to 7.20 percentage points.

> **Note:** This repository currently documents the architecture, methodology,
> and results of the project. Code is being cleaned up for public release —
> check back for updates, or reach out if you'd like early access.

## Motivation

Driver drowsiness and mental overload are major contributors to road
accidents. Camera-based and single-signal detection methods often aren't
reliable enough on their own. This project combines two complementary
physiological signals — EEG (brain activity) and ECG (heart activity) — since
together they capture a fuller picture of cognitive state than either alone.

## Dataset

Uses the public **CL-Drive** dataset:
- 21 subjects
- EEG + ECG recorded via wearables during simulated driving
- 3,035 labeled 10-second samples
- Labels: Low / Medium / High mental workload

No custom hardware or data collection was required — evaluation is entirely
on this existing public benchmark.

## Architecture

**Pipeline:**

```
EEG Signal ──┐                                          ┌── Low
             ├─► Per-Modality Feature Extraction         │
ECG Signal ──┘   (Conv layers + CBAM Attention +          ├── Medium
                  Instance Normalization)                 │
                        │                                 └── High
                        ▼
              Self-Attention (per modality)
                        │
                        ▼
        Bidirectional Cross-Attention (EEG ↔ ECG fusion)
                        │
                        ▼
       Temporal Convolutional Network (TCN) — temporal modeling
                        │
                        ▼
                  Classification Head
```

**Key design choices:**
1. **Per-modality feature extraction** — dilated residual convolution blocks
   with CBAM attention and Instance Normalization, applied separately to EEG
   and ECG.
2. **Self-attention** — each signal attends to itself before fusion.
3. **Bidirectional cross-attention** — the core fusion mechanism; EEG and ECG
   representations attend to each other, allowing each modality to inform the
   other.
4. **Temporal Convolutional Network (TCN)** — captures patterns over time in
   the fused representation.
5. **Classification head** — outputs Low / Medium / High mental workload.

## Training

To handle class imbalance and reduce overfitting on a relatively small
subject pool:
- **Focal loss** to focus learning on harder, minority-class examples
- **Class weighting**
- **Data augmentation**
- **Mixup**

## Evaluation Protocol

Evaluated under **4 settings** to test both easier and harder generalization:

| Split | Description |
|---|---|
| 10-fold cross-validation | Within-subject, easier setting |
| Leave-One-Subject-Out (LOSO) | Tests on a completely unseen driver, harder setting |

Each protocol was run for both **binary (2-class)** and **ternary (3-class)**
classification.

## Results

Outperformed the previous best-published baseline (Park et al.) across **all
4 evaluation settings**, by **4.7 to 10.3 percentage points**, with the
largest gain on the hardest setting (ternary classification, LOSO).

Statistical significance of the improvement was confirmed using **Wilcoxon
signed-rank tests with Holm-Bonferroni correction**, ruling out chance as an
explanation for the gains.

### Ablation Study

Removed one component at a time to isolate its contribution:

| Ablation | Finding |
|---|---|
| EEG only / ECG only | Confirms multimodal fusion is the single biggest contributor to performance |
| No cross-attention | Architecture choice helps, but less than modality fusion itself |
| No TCN | Temporal modeling contributes to performance |
| No training tricks (focal loss, mixup, etc.) | Doesn't change peak accuracy much, but improves fairness across classes (better minority-class performance) |

## Limitations & Future Work

- Small subject pool (21 subjects) limits generalization claims
- Simulator-only data — not yet validated on real-world driving
- Not yet real-time — current pipeline is offline/batch evaluation

**Planned next steps:**
- Validate on additional public cognitive-load datasets
- Build a real-time inference version
- Incorporate additional signals (galvanic skin response, eye gaze)

## License

MIT
