"""Train Model v2 from a prepared manifest; save resumable checkpoints."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import BalancedSourceSampler, ManifestOrientationDataset, read_manifest
from training.metrics import orientation_metrics
from training.models import BACKBONES, build_model, read_checkpoint


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp.pth")
    torch.save(value, temporary)
    temporary.replace(path)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    targets, predictions = [], []
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="Train" if training else "Validation"):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    return {"loss": loss_sum / len(targets), **orientation_metrics(targets, predictions)}


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    parser.add_argument("--input-size", type=int, choices=[224, 384, 448], default=384)
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--source-weights", help="Path to a JSON source-weight file, or inline JSON")
    parser.add_argument("--output-dir", default="output/v2")
    parser.add_argument("--resume", help="Path to last_checkpoint.pth")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-output-path")
    args = parser.parse_args()
    if not args.onnx_output_path:
        args.onnx_output_path = "model/orientation_model_v2.onnx" if args.model_version == "v2" else "model/orientation_model.onnx"
    if min(args.epochs, args.batch_size, args.patience, args.cpu_threads) <= 0 or args.workers < 0:
        parser.error("epochs, batch-size, patience and cpu-threads must be positive")
    return args


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print(f"Device: {device}; {args.backbone}, {args.input_size}px; OCR is not used.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last_checkpoint.pth").exists():
        raise ValueError("Run exists. Use --resume or choose another --output-dir.")
    records = read_manifest(args.manifest, args.allow_unreviewed)
    train = ManifestOrientationDataset(args.manifest, "train", args.input_size, records=records)
    val = ManifestOrientationDataset(args.manifest, "validation", args.input_size, records=records)
    generator = torch.Generator().manual_seed(args.seed)
    source_weights = None
    if args.source_weights:
        value = args.source_weights
        if not value.lstrip().startswith("{"):
            value = Path(value).read_text(encoding="utf-8")
        source_weights = json.loads(value)
    sampler = BalancedSourceSampler(train, source_weights, args.seed) if source_weights else None
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=args.workers, generator=generator,
                              worker_init_fn=seed_worker, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers)
    config = {**vars(args), "model_version": args.model_version, "resize_mode": "letterbox",
              "source_weights": source_weights,
              "angles_ccw": [0, 90, 180, 270],
              "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    checkpoint = read_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        for key in ("backbone", "input_size", "manifest_sha256", "epochs", "seed",
                    "source_weights", "batch_size", "lr", "weight_decay", "no_amp", "workers"):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
    model = build_model(args.backbone, pretrained=not args.no_pretrained and not checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None
    criterion = nn.CrossEntropyLoss()
    history, best, stale, start = [], -1.0, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        history, best, stale, start = checkpoint["history"], checkpoint["best"], checkpoint["stale"], checkpoint["epoch"] + 1
        torch.set_rng_state(checkpoint["rng"]["torch"])
        generator.set_state(checkpoint["rng"]["loader"])
        random.setstate(checkpoint["rng"]["python"])
        ns = checkpoint["rng"]["numpy"]
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        if device.type == "cuda" and checkpoint["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    (output / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        if sampler:
            sampler.set_epoch(epoch)
        print(f"Epoch {epoch+1}/{args.epochs}", flush=True)
        train_result = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_result = run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train": train_result, "validation": val_result})
        for angle in ("0", "180"):
            metric = val_result["per_angle_cw"][angle]
            print(f"Validation {angle} CW: precision={metric['precision']:.4f}, recall={metric['recall']:.4f}")
        improved = val_result["macro_f1"] > best
        best, stale = (val_result["macro_f1"], 0) if improved else (best, stale + 1)
        ns = np.random.get_state()
        state = {
            "config": config, "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch, "history": history, "best": best, "stale": stale,
            "rng": {"torch": torch.get_rng_state(), "loader": generator.get_state(),
                    "python": random.getstate(), "numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
                    "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
        }
        if improved:
            atomic_save({"config": config, "model_state": model.state_dict(),
                         "epoch": epoch, "metrics": val_result}, output / "best_model.pth")
        atomic_save(state, output / "last_checkpoint.pth")
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation macro F1: {best:.4f}; checkpoint: {output / 'best_model.pth'}")
    if args.export_onnx:
        from training.export_onnx import export_model
        export_model(str(output / "best_model.pth"), args.onnx_output_path, verify=True)


if __name__ == "__main__":
    main()
