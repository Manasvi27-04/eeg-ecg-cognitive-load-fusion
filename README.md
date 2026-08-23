# Attention-Based Multimodal EEG-ECG Fusion for Cognitive Load Assessment

An attention-based deep learning framework that fuses EEG and ECG signals to classify driver cognitive load (mental workload), benchmarked against 16 published baselines (2017–2023) on the public CL-Drive dataset — achieving the highest accuracy across all four evaluation settings (binary/ternary × 10-fold/LOSO), outperforming the strongest baseline by up to 9.73 percentage points.

Note: This repository currently documents the architecture, methodology, and results of the project. Code is being cleaned up for public release — check back for updates, or reach out if you'd like early access.

## Motivation

Driver drowsiness and mental overload are major contributors to road accidents. Camera-based and single-signal detection methods often aren't reliable enough on their own. This project combines two complementary physiological signals — EEG (brain activity) and ECG (heart activity) — since together they capture a fuller picture of cognitive state than either alone.

## Dataset

Uses the public CL-Drive dataset:

21 subjects
EEG + ECG recorded via wearables during simulated driving
3,035 labeled 10-second samples
Labels: Low / Medium / High mental workload

No custom hardware or data collection was required — evaluation is entirely on this existing public benchmark.

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
1. **Per-modality feature extraction** — dilated residual convolution blocks with CBAM attention and Instance Normalization, applied separately to EEG and ECG.
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

Evaluated against 16 published EEG/ECG baselines (2017–2023) on the CL-Drive dataset (21 subjects), under a unified protocol: identical fold seeds, per-subject normalization, augmentation, and training loop shared across every model — verified by parameter-identity tests, not just re-implementation

Setting	Proposed	Best Baseline	Margin
Binary, 10-fold	85.24%	84.78% (GMU fusion)	+0.46 pt
Binary, LOSO	77.72%	76.87% (GMU fusion)	+0.85 pt
Ternary, 10-fold	77.63%	70.43% (MulT)	+7.20 pt (p<0.05)
Ternary, LOSO	64.25%	54.52% (GMU fusion)	+9.73 pt (p<0.01)

Statistical significance confirmed via paired Wilcoxon signed-rank tests, Holm-Bonferroni corrected within each experiment.

<details> <summary>Full baseline comparison (16 methods, Ternary 10-fold)</summary>
  
Method	Year	Modality	Params	Acc %	F1 %
MulT cross-modal	2019	Both	422,307	70.43	70.37
GMU gated fusion	2020	Both	121,827	67.63	68.36
Late fusion	—	Both	113,958	66.36	67.12
ATCNet	2023	ECG	107,951	58.45	58.89
ATCNet	2023	EEG	107,983	57.51	5

(Full results across all four experiments — binary/ternary × 10-fold/LOSO — available in `results/baseline_table.txt`.)
 </details>

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
