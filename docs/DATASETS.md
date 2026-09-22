# Document Training & Evaluation Datasets

This document details the training, validation, and test datasets used for the Document Orientation Service, including source provenance, licensing, deduplication, and split preservation.

---

## 1. Unified Dataset Composition

Both **Model v1 (ResNet-18)** and **Model v2 (EfficientNet-B0)** were evaluated and trained on the universal multi-domain document corpus comprising **4,093 unique document pages** (SHA-256: `183047f7e092f1f855f1fc4821498b34f266a31d9480a731136040a5f976040e`):

| Source | Domain / Category | Train Pages | Validation Pages | Test Pages | Total Unique | License |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **[DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2)** | Corporate reports, financial filings, legal documents, patents, manuals | 1,744 | 375 | 375 | 2,494 | CDLA-Permissive-1.0 |
| **[CORD v2](https://github.com/clovaai/cord)** | Retail & restaurant receipts with camera skew and uneven lighting | 175 | 37 | 38 | 250 | CC BY 4.0 |
| **[FUNSD](https://huggingface.co/datasets/nielsr/funsd)** | Scanned, noisy administrative and business forms | 110 | 39 | 0 | 149 | Permissive research |
| **Multilingual Synthetic Corpus** | Structured administrative templates: Cyrillic, Arabic, Indic (Devanagari), CJK | 897 | 177 | 126 | 1,200 | Open / Generated |
| **Total** | | **2,926** | **628** | **539** | **4,093** | |

With 4 deterministic orthogonal rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) generated per base page:
- **Training Examples**: 11,704
- **Validation Examples**: 2,512
- **Held-out Test Examples**: 2,156

---

## 2. Dataset Hygiene & Split Integrity

1. **Official Split Preservation**: Published train/validation/test splits from DocLayNet and CORD are preserved strictly to prevent data contamination across splits.
2. **Document Family / Group Isolation**: Multi-page documents and related templates share a deterministic `group_id`. All pages belonging to the same group are pinned to the same split, ensuring that zero document templates leak into test or validation.
3. **Exact-Hash Deduplication**: SHA-256 hashes of all raw image bytes are computed and indexed. Exact duplicates within or across splits are automatically identified and excluded.
4. **Upright Verification**: All training seed documents are audited to guarantee an initial upright ($0^\circ$) orientation. Synthetically generated samples apply precise $90^\circ, 180^\circ,$ and $270^\circ$ affine transformations with deterministic seed tracking.

---

## 3. Preparation Pipeline

To prepare or reproduce the document corpus from official Hugging Face Parquet sources:

```powershell
python -u -m training.prepare_data \
  --output-dir data/v2 \
  --doclaynet 5000 \
  --cord 600 \
  --local-manifest data/local.jsonl
```

Key guarantees of the data preparation engine:
- Pinned commit revisions for remote Hugging Face datasets.
- Streaming reads of image columns only, avoiding multi-gigabyte raw PDF payload downloads.
- Storage of normalized RGB images (max dimension $\le 1024$ pixels) with JSONL manifests and SHA-256 checksums.
- Automatic resume capability: interrupted downloads resume without duplicating already processed pages.
