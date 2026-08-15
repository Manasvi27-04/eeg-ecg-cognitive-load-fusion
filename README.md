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
