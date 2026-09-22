# Model v2 Architecture & Inference Pipelines

This document details the neural architecture, dataset curation, and runtime performance of **Model v2 (EfficientNet-B0)** alongside the **Hybrid Pipeline (Model v1 + Text Verification)** and **GenAI Fallback**.

For the multimodal Gemini route, see [GENAI.md](GENAI.md).

---

## 1. Pipeline Overview

The Document Orientation Service exposes three independent inference pipelines selectable via the `mode` parameter:

| Mode | Active Model | Dependencies | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **`pure`** (default) | `model/orientation_model_v2.onnx` | ONNX Runtime CPU | High-throughput, sub-35ms production processing; zero OCR overhead. |
| **`hybrid`** | `model/orientation_model.onnx` (v1) | ONNX Runtime + RapidOCR | Verification of ambiguous 0° vs. 180° upside-down text orientations. |
| **`genai`** | Google Gemini (`gemini-3.5-flash-lite`) | HTTP client (Google GenAI) | Zero-shot visual orientation for complex, dense, or distorted documents. |

### Configuration (`app/core/config.py`)

Key environment variables:
- `APP_INFERENCE_MODE`: Default pipeline (`pure`, `hybrid`, `genai`).
- `APP_MODEL_PATH_V2`: Path to Model v2 ONNX artifact (`model/orientation_model_v2.onnx`).
- `APP_MODEL_PATH_V1`: Path to Model v1 ONNX artifact (`model/orientation_model.onnx`).
- `APP_INFERENCE_THREADS`: ONNX Runtime intra-op thread count (default: `2`).
- `APP_HYBRID_CONFIDENCE_THRESHOLD`: Top probability cutoff triggering OCR verification (default: `0.90`).
- `APP_HYBRID_MARGIN`: Gap between 0° and 180° probabilities triggering OCR (default: `0.25`).

---

## 2. Neural Architecture Comparison

| Property | Model v1 (ResNet-18) | Model v2 (EfficientNet-B0) |
| :--- | :--- | :--- |
| **Backbone** | ResNet-18 | EfficientNet-B0 |
| **Input Resolution** | 224 × 224 | 384 × 384 |
| **Preprocessing** | Letterbox, ImageNet RGB normalization | Letterbox, ImageNet RGB normalization |
| **ONNX Artifact Size** | 42.63 MB (44,704,998 bytes) | **15.30 MB** (16,040,936 bytes) |
| **Parameters** | ~11.2M | **~4.0M** |
| **Test Accuracy** | 96.94% (pure) / 97.77% (hybrid) | **97.68% (pure) / 97.96% (hybrid)** |
| **Isolated CPU Latency (p95)** | **50.8 ms** | **70.1 ms** |

### Why EfficientNet-B0 at 384×384?
- **Text Legibility**: At 224×224, small body text (6–9 pt on dense receipts or multi-column PDFs) collapses into unresolvable pixel artifacts. At 384×384 (2.94× higher pixel area), ascenders, descenders, and paragraph baselines remain sharp.
- **Compact Model Footprint**: Inverted residual blocks with depthwise-separable convolutions reduce model weight size from 42.6 MB down to 15.3 MB, accelerating cold starts and container distribution.
- **Letterbox Preprocessing**: Full page proportions are preserved without aspect distortion by padding with white borders to a square canvas, avoiding stretched or compressed typography.

---

## 3. Training & Dataset Curation

Both models were trained on the unified multi-domain document corpus comprising **4,093 unique document pages** (2,926 train / 628 validation / 539 test):

- **DocLayNet v1.2** (2,494 unique pages): Corporate reports, scientific articles, law/regulation documents, manuals, and financial statements.
- **CORD v2** (250 receipts): Real-world camera uploads and receipt photography with natural skew and lighting variations.
- **FUNSD** (149 forms): Noisy, scanned business forms.
- **Multilingual Synthetic Corpus** (1,200 pages): Structured multilingual templates covering Cyrillic, Arabic, Indic (Devanagari), and CJK scripts.

### Training Methodology
1. **Four-Class Rotation Generation**: Every base page generates exactly 4 training samples: 0°, 90°, 180°, and 270° counterclockwise (saved with deterministic SHA-256 validation).
2. **Augmentations**: Subtle color jitter, mild affine rotation (±2°), smooth shadow rendering, Gaussian blur, and low-amplitude noise. Horizontal/vertical reflections are strictly forbidden as they corrupt text reading direction.
3. **Optimization**: AdamW optimizer, cosine annealing learning rate schedule, mixed precision (AMP), and validation macro-F1 checkpoint selection.

---

## 4. Hybrid Verification Logic

In `mode=hybrid`, the system applies a two-stage verification strategy:
1. **CNN Inference**: The neural network produces class probabilities across $[0^\circ, 90^\circ, 180^\circ, 270^\circ]$.
2. **Ambiguity Gate**:
   - If the two most probable classes are $0^\circ$ and $180^\circ$, and
   - Either the top probability is $< 0.90$ or the probability margin $|P(0^\circ) - P(180^\circ)| < 0.25$,
   - Then **RapidOCR** text-line direction detection is invoked.
3. **Decision Arbitration**:
   - If RapidOCR detects text angle $0^\circ$ or $180^\circ$, it overrides or confirms the CNN prediction (`ocr_override` or `cv_ocr_confirmed`).
   - RapidOCR cannot change the document axis (cannot turn a 0°/180° candidate into 90°/270°).
   - If OCR is unavailable or inconclusive, the CNN prediction is preserved.

---

## 5. ONNX Export & Parity Verification

To export trained checkpoints to production ONNX artifacts:

```powershell
# Export checkpoint with parity validation
python -m training.export_onnx \
  --model-path output/v2/best_model.pth \
  --output-path model/orientation_model_v2.onnx \
  --verify

# Benchmark ONNX Runtime latency on CPU
python -m training.benchmark \
  --model-path model/orientation_model_v2.onnx \
  --threads 2 \
  --runs 100 \
  --target-ms 100
```

The export script verifies numerical parity between PyTorch logits and ONNX Runtime outputs across batch sizes 1 and 2 (tolerance $\le 10^{-4}$), embedding custom metadata (`model_version`, `input_size`, `resize_mode`, `angles_ccw`) directly inside the ONNX file.
