"""Read-only local model audit. No Gemini calls or model/data changes.

Run from repository root with: python output/project_review_2026_09_22/audit_models.py
Writes reproducible summary and prediction records alongside this script.
"""
import hashlib
import json
import logging
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.classifier import OrientationClassifier
from training.dataset import read_manifest
from training.metrics import orientation_metrics

logging.basicConfig(level=logging.ERROR)
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/v2_universal_4k/manifest.jsonl"
records = read_manifest(MANIFEST)
test = [r for r in records if r["split"] == "test"]
summary = {
    "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "onnxruntime": onnxruntime.__version__,
    "threads": 2,
    "unique_split_counts": dict(Counter(r["split"] for r in records)),
    "unique_source_splits": dict(Counter(r["source"] + "/" + r["split"] for r in records)),
    "scope": "Existing test labels, not independently audited. All four rotations. "
             "Sequential local CPU inference; latency excludes decode, rotation, PDF and HTTP. "
             "Hybrid uses identical weights to each pure baseline.",
    "models": {},
}
all_predictions = {}
for version, filename in [("v1", "orientation_model.onnx"), ("v2", "orientation_model_v2.onnx")]:
    model_path = ROOT / "model" / filename
    model = OrientationClassifier(model_path, model_version=version, threads=2)
    with Image.open(test[0]["resolved_path"]) as raw:
        warmup = raw.convert("RGB")
    for _ in range(10):
        model.predict(warmup, mode="pure")
    predictions = []
    start = time.perf_counter()
    for index, record in enumerate(test):
        with Image.open(record["resolved_path"]) as raw:
            base = ImageOps.exif_transpose(raw).convert("RGB")
        base = base.rotate(-record.get("correction_cw", 0), expand=True)
        for label in range(4):
            image = base.rotate(label * 90, expand=True)
            t = time.perf_counter()
            pure = model.predict(image, mode="pure")
            pure_ms = (time.perf_counter() - t) * 1000
            # Use production gating and OCR implementation without altering weights.
            t = time.perf_counter()
            hybrid = model.predict(image, mode="hybrid")
            hybrid_ms = (time.perf_counter() - t) * 1000
            predictions.append({
                "path": record["path"], "group_id": record["group_id"],
                "source": record["source"], "label": label,
                "pure": [0, 270, 180, 90].index(pure.predicted_orientation),
                "hybrid": [0, 270, 180, 90].index(hybrid.predicted_orientation),
                "confidence": pure.confidence, "decision": hybrid.decision_source,
                "pure_ms": pure_ms, "hybrid_ms": hybrid_ms,
            })
        if (index + 1) % 50 == 0:
            print(f"{version}: {index + 1}/{len(test)} base pages; "
                  f"{time.perf_counter() - start:.1f}s", flush=True)
    targets = [p["label"] for p in predictions]
    result = {
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "bytes": model_path.stat().st_size,
        "input_size": model.input_size, "resize_mode": model.resize_mode,
        "decisions": dict(Counter(p["decision"] for p in predictions)),
        "ocr_fixes": sum(p["pure"] != p["label"] and p["hybrid"] == p["label"] for p in predictions),
        "ocr_harms": sum(p["pure"] == p["label"] and p["hybrid"] != p["label"] for p in predictions),
        "confident_errors_at_090": sum(p["pure"] != p["label"] and p["confidence"] >= .9 for p in predictions),
        "total_seconds": time.perf_counter() - start,
    }
    for mode in ["pure", "hybrid"]:
        result[mode] = orientation_metrics(targets, [p[mode] for p in predictions])
        result[mode]["by_source"] = {
            src: orientation_metrics([p["label"] for p in predictions if p["source"] == src],
                                     [p[mode] for p in predictions if p["source"] == src])
            for src in sorted({p["source"] for p in predictions})
        }
        latencies = [p[mode + "_ms"] for p in predictions]
        result[mode]["latency_ms"] = {
            "median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)), "max": max(latencies),
        }
    result["blank_pages"] = []
    for size in [(800, 1100), (1100, 800), (1000, 1000)]:
        p = model.predict(Image.new("RGB", size, "white"), mode="pure")
        result["blank_pages"].append({"size": size, "correction_cw": p.correction_rotation,
                                      "confidence": p.confidence})
    summary["models"][version] = result
    all_predictions[version] = predictions
    (OUT / (version + "_predictions.json")).write_text(json.dumps(predictions, indent=2))
    (OUT / "model_audit.json").write_text(json.dumps(summary, indent=2))
    print(version, json.dumps({k: result[k] for k in ["decisions", "ocr_fixes", "ocr_harms"]}), flush=True)
    print(version, "pure", result["pure"]["accuracy"], "hybrid", result["hybrid"]["accuracy"], flush=True)
    del model

a, b = all_predictions["v1"], all_predictions["v2"]
summary["paired_pure"] = {
    "v1_correct_v2_wrong": sum(x["pure"] == x["label"] and y["pure"] != y["label"] for x, y in zip(a, b)),
    "v2_correct_v1_wrong": sum(x["pure"] != x["label"] and y["pure"] == y["label"] for x, y in zip(a, b)),
    "both_wrong": sum(x["pure"] != x["label"] and y["pure"] != y["label"] for x, y in zip(a, b)),
}
(OUT / "model_audit.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["paired_pure"]), flush=True)
