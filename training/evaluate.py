"""Evaluate pure neural classification on the held-out manifest split."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import ManifestOrientationDataset, read_manifest
from training.metrics import orientation_metrics
from training.models import build_model, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="output/v2/evaluation")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = read_checkpoint(args.model_path)
    config = checkpoint["config"]
    if config["model_version"] != "v2":
        raise ValueError("This evaluation entry point expects a v2 checkpoint.")
    records = read_manifest(args.manifest)
    dataset = ManifestOrientationDataset(args.manifest, "test", config["input_size"], records=records)
    model = build_model(config["backbone"], pretrained=False).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size), desc="Test"):
            predictions.extend(model(images.to(args.device)).argmax(1).cpu().tolist())
            targets.extend(labels.tolist())
    metrics = orientation_metrics(targets, predictions)
    for field in ("source", "language", "category"):
        groups = {}
        for index, record in enumerate(dataset.records):
            group = str(record.get(field, "unknown"))
            groups.setdefault(group, []).extend(range(index * 4, index * 4 + 4))
        metrics["by_" + field] = {
            group: orientation_metrics([targets[i] for i in ids], [predictions[i] for i in ids])
            for group, ids in groups.items()}
    metrics["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    metrics["training_manifest_sha256"] = config.get("manifest_sha256")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d",
                xticklabels=metrics["confusion_angles_cw"], yticklabels=metrics["confusion_angles_cw"])
    plt.xlabel("Predicted angle CW"); plt.ylabel("True angle CW")
    plt.tight_layout(); plt.savefig(output / "confusion_matrix.png"); plt.close()
    print(json.dumps({key: metrics[key] for key in ("accuracy", "per_angle_cw", "confusions_0_180")}, indent=2))


if __name__ == "__main__":
    main()
