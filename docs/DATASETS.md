# Better document training data

Research checked 2026-09-18. Keep ResNet-18 for the first improved baseline.

| Dataset | Why it helps | Access and published terms |
|---|---|---|
| [DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) | 80,863 pages: finance, science, manuals, patents, laws and tenders. Varied layouts including color material, with page images and embedded PDFs. | Parquet, about 39.8 GB full download; streaming supported. CDLA-Permissive-1.0. |
| [CORD v2](https://github.com/clovaai/cord) | Receipt images add camera backgrounds and commercial document layouts. The public release has 1,000 images, not the full 11,000-image collection. | naver-clova-ix/cord-v2; official 800/100/100 splits. CC BY 4.0. |
| [M6Doc](https://github.com/HCIILAB/M6Doc) | 9,080 modern document images: magazines, textbooks, newspapers, exam papers, notes and books. Includes digital, scanned and photographed sources. | Full archive requires application/password. CC BY-NC-ND 4.0, non-commercial research. |
| [DocSynth-300K](https://github.com/opendatalab/DocLayout-YOLO) | Synthetic layout diversity; an optional supplement after a real-document baseline. | Follow authors' download instructions; verify dataset terms separately from the repository code license. |

**Recommendation:** DocLayNet as the main corpus, CORD as a small supplement, and your own upright color documents as the target-domain data. Include forms, invoices, certificates, letters, slides, phone photos and multipage PDFs. Reserve complete document families for testing.

The [original DocLayNet release](https://github.com/DS4SD/DocLayNet) also provides PNG images and matching single-page PDF files. Neither it nor CORD guarantees that every page is colorful. Audit sampled pages for actual color, language and layout diversity.

## Problems in the previous setup

- The original ds4sd/DocLayNet repository requires a loading script, unsupported by the installed datasets 5.x. The loader now targets the official v1.2 Parquet release.
- The [invoice repository previously selected](https://huggingface.co/datasets/devpatel18042004/receipts-invoices-bank-statements/tree/main) contains a 33.2 GB RAR archive, not the image table expected by the loader. Its dataset card leaves the license unspecified. It is no longer part of the default mix.
- Generator failures happen during iteration; the former fallback only guarded construction and therefore missed them. Source failures now stop training visibly.
- Random page splitting can put pages from one PDF in both training and evaluation. Published splits are now preserved. Local files use a deterministic file-level split before PDF rendering.
- DocVQA question counts are not unique image counts; question samples can repeat pages. These sources need deduplication before use.
- Full-resolution images were retained in memory. Training now retains only 224x224 RGB previews.

All source pages must be checked as upright. Layout annotations are not orientation labels. Remove blanks or ambiguous pages from four-class supervised training, or label them separately for an eventual abstain policy.

## Train without replacing the serving model

The Hugging Face "Resolving data files" bars only describe file discovery,
not image download completion. The loader now requests only the image column
from Parquet, shows page counts, and reports a waiting message every 15 seconds
when no new pages arrive. The first pages can take longer because remote reads
fetch data in chunks and the shuffle buffer must fill. Training loads only
train/validation pages; evaluation loads only test pages. After updating the
loader, stop an existing command with Ctrl+C and start it again to use the fix.

~~~powershell
.\.venv\Scripts\python.exe -m training.train --data-source hf_mixed --num-images 5000 --epochs 15 --seed 42 --output-dir output/color-baseline --export-onnx --onnx-output-path output/color-baseline/orientation_model.onnx
~~~

The default mix uses DocLayNet plus up to 20% CORD, capped at CORD's released split sizes. Page counts are upper bounds when a source contains fewer pages. Labels preserve the existing checkpoint convention: 0/90/180/270 degrees counterclockwise.

~~~powershell
.\.venv\Scripts\python.exe -m training.evaluate --model-path output/color-baseline/best_model.pth --data-source hf_mixed --num-images 5000 --seed 42 --output-dir output/color-baseline/evaluation
~~~

For local data select --data-source local --data-dir data/upright. Local grouping keeps all pages of one PDF together, but cannot detect related files, duplicate exports or similar templates automatically. Preserve relative paths when reproducing a local split. For a large run, pin source revisions and save a frozen page manifest.

Use a manually checked real-world test set outside training. Report accuracy by angle, source, color/grayscale, language and input type. Four rotations of one page are four training examples, not four independent documents.

## Later model improvements

ResNet-18 is already a computer-vision model. Keep it as the measured baseline. The recorded training history peaks at approximately 95.99% validation classification accuracy; this does not establish correct end-to-end API rotation or accuracy on new color documents.

A next experiment is a document-pretrained vision transformer with a four-class orientation head, such as [Microsoft DiT](https://arxiv.org/abs/2203.02378). It needs orientation fine-tuning and a comparison on the same held-out set; existing document-type classification weights do not directly solve orientation. OCR-based orientation can help uncertain text-heavy pages. A vision-language model is an optional fallback that needs latency and accuracy evaluation.

Quarter-turn orientation, small-angle deskew, perspective correction and dewarping are separate tasks. The current system handles only quarter turns.
