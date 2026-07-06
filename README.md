# Mechanistic Interpretability of Hybrid SSM-Attention Models

**Functional Specialization of Weight-Tied Attention in Zamba2-1.2B**

**Mukund Pandey** — Independent Researcher

---

## Overview

This repository contains code, results, and the full paper draft for a mechanistic interpretability study of Zamba2-1.2B — a hybrid model with 38 layers (32 Mamba-2 + 6 shared-attention hybrid at positions {5, 11, 17, 23, 29, 35}).

**Key finding:** Despite using identical QKV weights at all 6 hybrid positions, the shared attention block implements functionally distinct computations at different depths:

- **Layers 5, 11:** Induction heads (copy scores 0.669 and 0.485)
- **Layer 23:** Induction completion — largest SSMI jump (+8.9 log-prob units)
- **Layer 35:** Factual recall resolution (~96% accuracy, logp = -0.043)

Functional specialization emerges from residual stream context, not weight differences. This is the first mechanistic analysis of weight-tied attention in hybrid SSM architectures.

---

## Repository Structure

```
mech-interp-hybrid-ssm/
├── experiments/
│   ├── exp1_ssm_induction_score.ipynb    # SSM Induction Score (SSMI) across all 38 layers
│   ├── exp2_shared_attention_heads.ipynb # Copy score analysis for 6 hybrid layers
│   └── exp3_logit_lens.ipynb             # Logit-lens factual recall experiment
├── results/
│   ├── ssmi_zamba2_results.json                  # Exp 1 output: SSMI per layer
│   ├── copy_scores_zamba2_results.json           # Exp 2 output: copy scores per head
│   └── logit_lens_factual_zamba2_results.json    # Exp 3 output: factual recall logprobs
├── figures/
│   ├── ssmi_zamba2_1.2b.png/pdf          # Figure 1
│   ├── copy_scores_zamba2.png/pdf        # Figure 2
│   └── logit_lens_factual_zamba2.png/pdf # Figure 3
└── latex/
    ├── paper.tex                         # LaTeX source
    ├── references.bib                    # Bibliography (12 references)
    └── paper.pdf                         # Compiled PDF
```

---

## Experiments

All experiments run on Google Colab T4 GPU (free tier) using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) with Zamba2-1.2B adapters (PRs [#1434](https://github.com/TransformerLensOrg/TransformerLens/pull/1434) and [#1486](https://github.com/TransformerLensOrg/TransformerLens/pull/1486)).

### Experiment 1 — SSM Induction Score (SSMI)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JnzYlu16soGZPBf_ty840gruCXaiN2dT?hl=en)
Logit-lens proxy for induction-like behavior in recurrent layers. Applies `ln_final + W_U` to cached residual stream outputs, measuring log-probability of correct induction token on ABC→BCD sequences.

**Result:** SSMI peaks at Layer 23 (-2.35), up from -11.2 at Layer 22 — an 8.9 log-prob jump.

### Experiment 2 — Shared Attention Copy Scores
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ra438I9j4oy_NFKH26h8ESPfSmK0o9VT?hl=en)
Standard induction head detection across all 32 heads × 6 hybrid layers. Measures `E_t[A_h[t, t-prefix_len-1]]`.

**Result:** Strong induction heads only in layers 5 (head 19: 0.669) and 11 (head 12: 0.485). Layers 17–35 near zero.

### Experiment 3 — Logit Lens: Factual Recall
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LJvgcDSfJXwo5kUG7Qr0aeTCfSoE6Akr?hl=en)
Same logit-lens method applied to 13 factual prompts (capitals, chemical symbols). Measures log-probability of correct single-token answer at final position.

**Result:** Factual recall peaks at Layer 35 (-0.043, ~96% accuracy). Layer 23 shows moderate improvement only.

---

## Key Results

| Layer | Type | SSMI (induction) | Factual recall logp |
|-------|------|------------------|---------------------|
| 5     | hybrid | ~-19.4 | -14.334 |
| 11    | hybrid | ~-14.2 | -13.237 |
| 17    | hybrid | ~-10.6 | -11.540 |
| 23    | hybrid | **-2.35 (peak)** | -4.994 |
| 29    | hybrid | ~-2.8  | -1.845  |
| 35    | hybrid | ~-3.0  | **-0.043 (peak)** |

---

## Open-Source Infrastructure

These experiments depend on TransformerLens adapters for Zamba2-1.2B:

- **PR #1434:** `HybridModelBridge` base class for hybrid SSM-attention architectures
- **PR #1486:** `Zamba2Bridge` — complete adapter exposing `hook_in`/`hook_out` on all 38 layers and attention hooks on 6 hybrid layers

---

## arXiv

Paper submitted — arXiv ID pending (will update once assigned).

---

## Citation

```bibtex
@article{pandey2025mech,
  title={Mechanistic Interpretability of Hybrid SSM-Attention Models: 
         Induction, Factual Recall, and Weight-Tied Attention in Zamba2-1.2B},
  author={Pandey, Mukund},
  year={2026}
}
```
