# EmoNeXt-FER2013: Refined & Optimized

A stabilized, high-performance implementation of the **EmoNeXt** architecture, purpose-built for Facial Emotion Recognition (FER). This repository goes beyond the original implementation by introducing critical architectural bug fixes and overhauling the training pipeline to conquer the challenging FER2013 dataset.

---

## Under the Hood: What's Changed?

We identified several bottlenecks in the original implementation and applied targeted fixes across the model and training loop. 

### 1. Architectural Upgrades (`models.py`)
* **Banishing Gradient Vanishing:** The original `DotProductSelfAttention` had a flawed scale factor. We corrected this to ensure healthy gradient flow during backpropagation.
* **Preserving Spatial Context:** Instead of aggressively pooling features early on, `forward_features` now retains the `(B, C, 7, 7)` spatial dimensions.
* **Granular Facial Tracking (49 Tokens):** The forward pass has been rewritten to compute attention across 49 distinct spatial tokens. This empowers the model to map the crucial relationships between localized facial regions (e.g., how the eyes and mouth interact during a smile).
* **Fixing the Self-Attention (SA) Loss:** We corrected the SA Loss logic to compute the mean per-row. This ensures mathematical correctness for row-wise comparisons and tensor broadcasting.
* **Taming Noisy Labels:** FER2013 is notoriously noisy. We introduced **Label Smoothing (0.1)** and balanced the loss function by assigning a weight of `λ = 0.1` to the SA Loss, preventing the model from over-indexing on incorrect labels.

### 2. The Training Engine (`train.py`)
* **Smarter Learning Rate Scheduling:** Warm restarts were causing instability. We replaced them with a smoother **Linear Warmup (10 epochs)** followed by a **Cosine Decay**, resulting in a much more stable fine-tuning phase.
* **Deterministic & Fast Validation:** 
  * Boosted validation speed by bumping the `batch_size` from 1 up to **32**.
  * Swapped `RandomCrop` for `CenterCrop` during the validation phase to guarantee reproducible, deterministic metrics.
* **Modernized Optimization:** Upgraded the code to use the latest `torch.amp.GradScaler` API and extended the Early Stopping patience to **20 epochs** to allow the model to fully converge.

---

## Benchmarks on FER2013

Despite the architectural strictness applied to prevent overfitting, this implementation achieves highly competitive performance, tracking closely with the original paper's metrics.

| Model / Metric | Overall Accuracy |
| :--- | :---: |
| **Our Optimized EmoNeXt (Val - Best)** | **73.07%** |
| **Our Optimized EmoNeXt (Test - EMA)** | **72.51%** |
| *Original EmoNeXt-Tiny Paper* | *73.34%* |

**Emotion Breakdown (EMA Accuracy):**
> **Happy:** 89.7% | **Surprise:** 84.1% | **Disgust:** 71.4% | **Neutral:** 71.2% | **Angry:** 65.1% | **Sad:** 61.5% | **Fear:** 55.3%

---

## Getting Started

Follow these steps to replicate our results or train on your own data.

### Step 1: Environment Setup
Ensure you have [CUDA](https://developer.nvidia.com/cuda-downloads) configured and [PyTorch (>=1.13)](https://pytorch.org/get-started/locally/) installed. Then, grab the remaining dependencies:
```bash
pip install -r requirements.txt
