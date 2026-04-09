
# SIGMA: Semantic Internalization for Generative Multimodal Alignment

> **ACL 2026** | Generative Text-to-Image Retrieval via Hierarchical Identifiers and Semantic Internalization

---

## Overview

Text-to-image retrieval has long been dominated by two paradigms: **two-tower** frameworks that align embeddings in a shared space (fast but shallow), and **one-tower** frameworks that enable deep cross-modal interaction (accurate but computationally prohibitive at scale). **Generative Retrieval (GR)** offers a promising third way — directly predicting image identifiers from a text query — yet it faces three key unsolved challenges:

1. **Semantic discriminability vs. generability trade-off** — identifiers must be expressive enough to distinguish images while remaining stable for generation.
2. **Alignment bias from "one image, multiple captions"** — rigidly mapping all captions to the same discrete ID ignores the varying degrees of semantic affinity between text-image pairs, causing gradient conflicts during training.
3. **Closed-set assumption** — existing generative methods require retraining when new images are added, making them unsuitable for dynamic, open-world scenarios.

**SIGMA** addresses all three challenges with a unified framework built on hierarchical identifiers and progressive semantic internalization.

---

## Framework

SIGMA consists of two core modules:

### Module 1: Multi-granularity Hierarchical Identifier Construction

Each image is assigned a **five-level identifier sequence** `[id1, id2, id3, id4, id5]`, constructed via a coarse-to-fine pipeline:

| Level | Name | Method | Purpose |
|-------|------|--------|---------|
| id1 | Global Semantic Anchor | K-Means clustering on visual features | Partition images into broad semantic regions |
| id2 | Dominant Semantic Components | Semi-NMF on intra-cluster features | Identify salient feature directions |
| id3 | Fine-grained Semantic Groups | Second-order Semi-NMF on residuals | Capture subtle intra-cluster variations |
| id4 | Local Residual Details | Residual Quantization across subspaces | Encode unique non-semantic visual details |
| id5 | Instance Suffix | Unique per-image terminal ID | Guarantee global uniqueness, eliminate collisions |

This hierarchical structure supports up to **1,048,576 unique identifiers** and ensures each image has a semantically traceable, collision-free representation.

### Module 2: Progressive Semantic Internalization Training

Inspired by the human cognitive strategy of "memorize the form first, then understand the semantics", SIGMA trains a multimodal large language model (Qwen2.5-VL) in three sequential stages:

**Stage 1 — Identifier Memorization**
The model learns `image → identifier` mappings via autoregressive generation. This establishes the syntactic foundation for identifier generation.

**Stage 2 — Identifier Comprehension**
The task is reversed: `identifier → description`. The model learns to associate abstract identifier tokens with concrete visual semantics, transforming them from shallow index keys into meaningful symbolic units.

**Stage 3 — Retrieval Alignment with Semantic Soft Labels (SSL)**
Given a text query, the model simultaneously: (1) generates the target identifier, and (2) predicts the query-image semantic similarity. Soft labels — pre-computed by a cross-modal teacher encoder — capture the nuanced affinity differences between multiple captions of the same image, resolving the alignment bias of rigid hard-label training.

The joint training loss is:

$$\mathcal{L}_\text{align} = \mathcal{L}_\text{gen} + \lambda_\text{ssp} \cdot \mathcal{L}_\text{ssp}$$

where $\mathcal{L}_\text{gen}$ is the autoregressive generation loss and $\mathcal{L}_\text{ssp}$ is an MSE loss on similarity predictions.

### Inference

At inference time, given a text query, SIGMA autoregressively generates the target identifier token-by-token. **Trie-based constrained decoding** ensures that every generated sequence corresponds to a valid image, with beam search restricted to paths in a prefix tree built over all training identifiers.

---

## Results

### Text-to-Image Retrieval (Flickr30K & MS-COCO)

| Type | Model | Flickr30K R@1 | Flickr30K R@5 | Flickr30K R@10 | MS-COCO R@1 | MS-COCO R@5 | MS-COCO R@10 |
|------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| Two-tower | CLIP | 58.4 | 81.5 | 88.1 | 37.8 | 62.4 | 72.7 |
| Two-tower | OpenCLIP | 63.9 | 87.3 | 93.2 | 39.4 | 65.4 | 75.6 |
| GR | GRACE-atomic | 68.4 | 88.9 | 93.7 | 41.5 | 69.1 | 79.1 |
| GR | TIGeR | 71.7 | 91.8 | 95.4 | 46.1 | 69.0 | 76.1 |
| GR | GENIUS | 74.1 | 92.0 | 94.8 | 46.1 | 74.0 | 82.7 |
| **Ours** | **SIGMA** | **77.4** | **92.5** | **97.2** | **49.3** | **72.6** | **79.5** |
| | Avg. Improvement over GR baselines | ↑11.5 | ↑7.5 | ↑8.1 | ↑9.8 | ↑9.5 | ↑5.9 |

SIGMA achieves average improvements of **+10.65% R@1, +8.50% R@5, +7.00% R@10** across both datasets over generative baselines.

### Ablation Study (Flickr30K)

| Variant | R@1 | R@5 | R@10 |
|---------|:---:|:---:|:---:|
| w/o ID Memorization | 73.2 | 87.6 | 92.0 |
| w/o ID Comprehension | 75.1 | 90.1 | 94.3 |
| w/o Semantic Soft Labels | 73.5 | 88.2 | 93.4 |
| **SIGMA (full)** | **77.4** | **92.5** | **97.2** |

All three stages are necessary. ID Memorization is the most critical (−4.2% R@1 when removed), as it establishes the visual-identifier mapping foundation.

### Efficiency: Scale-Invariant Throughput

Unlike two-tower frameworks (e.g., CLIP) whose throughput degrades ~3× from 1K to 100K candidates due to exhaustive similarity computation, SIGMA maintains **stable throughput regardless of database scale** — a direct consequence of the generative paradigm, which decouples inference complexity from dataset size. SIGMA also achieves nearly **twice the throughput** of the Flamingo-based GRACE, owing to the efficient Qwen2.5-VL backbone.

### Open-Set Retrieval (Inductive Capability)

Trained on Flickr30K, SIGMA is evaluated on 100 **unseen** MS-COCO images without any retraining. It achieves **56.4% R@5, 71.2% R@10, 84.0% R@20**, demonstrating robust inductive identifier assignment for open-world data — a capability largely absent in prior generative methods.

---

## Semantic Compression: A Key Finding

Identifiers in SIGMA are not arbitrary index keys — they function as **semantic compression codes**. When conditioned solely on an assigned identifier, the model can reconstruct accurate image captions. This confirms that visual semantics are genuinely internalized into the discrete identifier tokens through the memorization-comprehension training pipeline.

---

## Datasets

| Dataset | Images | Captions/Image | Train | Val | Test |
|---------|--------|---------------|-------|-----|------|
| Flickr30K | 31,783 | 5 | 29,783 | 1,000 | 1,000 |
| MS-COCO | 123,287 | 5 | 113,287 | 5,000 | 5,000 |

---

## Implementation Details

**Identifier Construction**
- Image encoder: `jina-embeddings-v4` (2,048-dim)
- id1: Mini-Batch K-Means (K=64)
- id2/id3: Semi-NMF with K=32 and K=16
- id4: Residual Quantization (K=32)
- Total identifier capacity: 1,048,576

**Training**
- Backbone: Qwen2.5-VL (full-parameter fine-tuning)
- Hardware: 4× NVIDIA A6000 (48GB)
- Optimizer: AdamW with cosine LR schedule

| Stage | Epochs | LR | Batch Size |
|-------|--------|----|-----------|
| ID Memorization | 5 | 5×10⁻⁵ | 64 |
| ID Comprehension | 5 | 3×10⁻⁵ | 64 |
| Retrieval Alignment | — | 2×10⁻⁵ | 10 |

---

## Ethical Considerations

SIGMA is trained on public benchmarks (Flickr30K, MS-COCO) that may contain societal biases. Retrieved results could inadvertently reflect these biases. We strictly adhere to all dataset usage licenses and use them solely for academic research. Any misuse of this framework to retrieve harmful or illegal content is explicitly condemned.

---

## Citation



---

## License

This project is for academic research purposes only. Please refer to the respective dataset licenses for Flickr30K and MS-COCO before use.
