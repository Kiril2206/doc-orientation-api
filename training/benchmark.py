"""Measure warm ONNX CPU latency including production image preprocessing."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime
from PIL import Image

from app.services.classifier import OrientationClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--target-ms", type=float, default=100)
    parser.add_argument("--output", default="output/v2/benchmark.json")
    args = parser.parse_args()
    if args.runs < 2 or args.threads < 1 or args.warmup < 0:
        parser.error("runs >=2, threads >=1, warmup >=0 required")
    classifier = OrientationClassifier(args.model_path, threads=args.threads)
    if args.image:
        with Image.open(args.image) as raw:
            image = raw.convert("RGB")
    else:
        image = Image.new("RGB", (1240, 1754), "white")
    for _ in range(args.warmup):
        classifier.predict(image, mode="pure")
    elapsed = []
    for _ in range(args.runs):
        start = time.perf_counter()
        classifier.predict(image, mode="pure")
        elapsed.append((time.perf_counter() - start) * 1000)
    result = {"model_path": args.model_path, "input_size": classifier.input_size,
              "image_size": image.size, "threads": args.threads, "runs": args.runs,
              "median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)),
              "max_ms": max(elapsed), "target_ms": args.target_ms, "cpu": platform.processor(),
              "platform": platform.platform(), "onnxruntime": onnxruntime.__version__,
              "scope": "warm CPU, batch=1, RGB resize+normalization+ONNX; excludes file decode, PDF rendering, HTTP and OCR"}
    result["passed"] = result["p95_ms"] < args.target_ms
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
