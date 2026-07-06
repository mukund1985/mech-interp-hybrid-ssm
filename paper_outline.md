# Mechanistic Interpretability of Hybrid SSM-Attention Models

**Working title** (will tighten after experiments)
**Target venue:** ICLR 2027 Workshop on Mechanistic Interpretability (or Alignment Forum + arXiv)
**Authors:** Mukund Pandey

---

## Abstract (placeholder — fill after experiments)

State space models (SSMs) such as Mamba-2 offer competitive performance to Transformers at a fraction of the compute, yet their internal circuitry remains poorly understood. Hybrid architectures — models that interleave SSM layers with a small number of shared-weight attention blocks (e.g., Zamba2, NemotronH) — compound this opacity: the same attention weights are reused across multiple positions in the layer stack, raising open questions about what each component learns and how they coordinate. We adapt mechanistic interpretability methods — induction score probes, direct-path activation patching, and hook-based residual stream analysis — to this class of models. We introduce an SSM induction score proxy that measures induction-like behavior in recurrent layers without access to attention patterns, and apply it to Zamba2-1.2B and NemotronH-8B using TransformerBridge adapters we contributed to the TransformerLens library. Our findings: [TBD — fill after experiments]. Code and adapters are publicly available.

---

## 1. Introduction

### 1.1 Motivation
- Transformers are well-understood mechanistically (induction heads, name-movers, copy-suppression). SSMs are not.
- Hybrid models (Zamba2, NemotronH, Jamba, Hymba) are production-grade and widely deployed — interpretability matters.
- Shared-weight attention is a particularly underexplored design: what happens when the same QKV weights are applied at layer 6 and layer 24?

### 1.2 Key questions (research questions to answer with experiments)
- **RQ1:** Do SSM layers in hybrid models develop induction-like behavior analogous to attention induction heads?
- **RQ2:** Does the shared attention block in Zamba2/NemotronH implement the same circuits (induction, name-mover, copy-suppression) as standard attention?
- **RQ3:** How do SSM layers and shared attention layers divide representational labor? Can activation patching reveal causal handoff?

### 1.3 Contributions
1. First mechanistic analysis of hybrid Mamba-2/attention architectures using hook-based interpretability.
2. SSM Induction Score (SSMI) — a probe for induction-like behavior in recurrent layers.
3. Shared-attention circuit analysis — does weight-tying collapse or differentiate circuits at different depths?
4. Open-source adapters enabling this work (TransformerLens PRs #1434, #1486).

---

## 2. Background

### 2.1 Mechanistic Interpretability in Transformers
- Induction heads (Olsson et al., 2022)
- Circuit analysis (Wang et al., 2022 — IOI)
- Direct path patching (Goldowsky-Dill et al., 2023)

### 2.2 State Space Models and Mamba-2
- Mamba (Gu & Dao, 2023), Mamba-2 (Dao & Gu, 2024)
- Recurrent vs attention: no explicit attention pattern, hidden state compression
- Prior interpretability work on SSMs: [survey — likely thin]

### 2.3 Hybrid Architectures
- Zamba2 (Zyphra, 2024): 38 layers, 32 Mamba-2 + 6 shared-attention hybrid
- NemotronH (NVIDIA, 2025): 52 layers, heterogeneous (mamba/attention/moe/mlp)
- Jamba (AI21, 2024), Hymba (NVIDIA, 2024) — related work

### 2.4 TransformerBridge
- Our contributed adapters enabling hook-based analysis of both architectures

---

## 3. Methods

### 3.1 SSM Induction Score (SSMI)
**Motivation:** Standard induction score uses attention patterns (attn[i, i - len(prefix)] for i in seq). SSM layers have no attention matrix.

**Proxy:** Measure how much a layer's output on token B[i] depends causally on token A[i - prefix_len], where A→B is a repeated bigram. Concretely:
- Run ABC→BCD test (standard induction test)
- Activation-patch the hidden state at position (i - prefix_len) from a clean run to a corrupted run where the prefix is replaced
- SSMI(layer) = mean reduction in correct-token logit from patching

This is a *causal* proxy, not an attention-pattern proxy.

### 3.2 Shared Attention Circuit Analysis
- Use direct path patching (TransformerLens PR #1396) on hybrid layers
- Probe whether the shared attention heads implement the same QK composition patterns as GPT-2 induction heads
- Measure: QK eigenspectrum per head, copy score, source→dest direct effect

### 3.3 Residual Stream Layer Attribution
- At each token position, measure each layer's contribution to the final logit (logit lens)
- Compare Mamba layers vs shared-attention layers on a controlled prompt

### 3.4 Models
- **Baseline:** GPT-2 small (well-understood, established induction heads in layers 1-2)
- **Primary:** Zamba2-1.2B (Zyphra/Zamba2-1.2B) — small enough for T4 GPU
- **Secondary:** NemotronH-8B (nvidia/Nemotron-H-8B-Base) — requires A100

### 3.5 Datasets / Prompts
- Standard induction test: 50-token prefix + repeated 50-token suffix
- IOI task (Wang et al., 2022) for name-mover circuit probing
- [Additional: factual recall prompts for logit lens]

---

## 4. Experiments

### Experiment 1 — SSM Induction Score Across Layers
**Goal:** Show whether Mamba-2 layers in Zamba2 develop induction-like behavior, and at which layers.
**Setup:** 100 random induction sequences, measure SSMI per layer.
**Expected output:** Plot of SSMI(layer) for all 38 layers, annotated with layer type (mamba/hybrid).
**Hypothesis:** Induction-like behavior peaks in layers after the first hybrid attention block (because attention seeds the pattern, SSM amplifies).

### Experiment 2 — Shared Attention Head Analysis
**Goal:** Characterize what the 6 shared-attention blocks in Zamba2 implement.
**Setup:** Direct path patching on hybrid layers, IOI task.
**Expected output:** Head attribution heatmap; copy score per head.
**Hypothesis:** A subset of the shared heads will have high copy scores (induction-like QK composition), similar to GPT-2 heads 4.4 and 5.5.

### Experiment 3 — Logit Lens: Residual Stream Attribution
**Goal:** Show how information accretes across SSM vs attention layers.
**Setup:** Logit lens on factual recall prompts (e.g., "The Eiffel Tower is in").
**Expected output:** Layer-by-layer correct-token probability, annotated with layer type.
**Hypothesis:** The shared attention layers will show sharp jumps in logit probability; Mamba layers will show smoother accumulation.

### Experiment 4 — Cross-Architecture Comparison (NemotronH-8B)
**Goal:** Validate that findings generalize beyond Zamba2.
**Setup:** Repeat Exp 1 and 3 on NemotronH-8B.
**Note:** Requires A100 (40GB); defer until Exps 1-3 are done on Zamba2.

---

## 5. Results

[TBD — fill after experiments]

Tables and figures to generate:
- Figure 1: SSMI per layer for Zamba2-1.2B (annotated by type)
- Figure 2: Shared attention head attribution on IOI task
- Figure 3: Logit lens across SSM and hybrid layers
- Table 1: SSMI summary statistics — Zamba2 vs GPT-2 small
- Table 2: Copy scores of shared attention heads vs GPT-2 4.4/5.5

---

## 6. Discussion

### 6.1 Do SSM layers implement induction?
[Fill from Exp 1 results]

### 6.2 Shared-weight attention: one circuit or many?
[Fill from Exp 2 results]

### 6.3 Division of labor: SSM vs attention
[Fill from Exps 1+3]

### 6.4 Limitations
- Activation patching in SSMs is an approximation (hidden state is not factored like residual stream)
- Zamba2-1.2B is small — may not generalize to 7B
- SSMI proxy is causal but indirect; future work could use mechanistic probing

---

## 7. Related Work

- Olsson et al. (2022) — In-context learning and induction heads
- Wang et al. (2022) — IOI circuit
- Goldowsky-Dill et al. (2023) — Direct path patching
- Gu & Dao (2023) — Mamba
- Dao & Gu (2024) — Mamba-2
- Ali et al. (2024) — Zamba2
- NVIDIA (2025) — NemotronH
- [Any existing SSM interpretability work — survey on write-up]

---

## 8. Conclusion

[TBD — fill last]

---

## Appendix

### A. TransformerBridge Adapter Details
- How SSMBlockBridge hooks work for Mamba layers vs hybrid layers
- Why is_stateful=False is correct for Zamba2 (cache routing via past_key_values)
- NemotronH heterogeneous layer handling

### B. Compute and Reproducibility
- Zamba2-1.2B experiments: Google Colab T4, free tier
- NemotronH-8B experiments: Vast.ai A100 40GB (~$X total GPU cost)
- All code available at: [GitHub repo URL]

---

## Notes / Open Questions

- Should we include weight-tied attention QK eigenspectrum analysis? Probably yes — unique to shared-weight design.
- Is the SSMI proxy novel enough to stand alone as a contribution? Check arxiv for prior SSM interpretability probes.
- Alignment Forum submission timeline: draft should be ready for feedback ~8 weeks before ICLR workshop deadline.

---

## File structure (to build)

```
paper/
  paper_outline.md         ← this file
  experiments/
    exp1_ssm_induction_score.ipynb
    exp2_shared_attention_heads.ipynb
    exp3_logit_lens.ipynb
    exp4_nemotronh.ipynb
  figures/                 ← generated plots land here
  drafts/
    v1_draft.md
```
