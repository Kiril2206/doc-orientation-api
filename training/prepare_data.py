"""Prepare a persistent RGB corpus; training never needs to contact the Hub."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

from PIL import Image, ImageOps

from training.dataset import (
    CORD_REPO, DOCLAYNET_REPO, SPLITS, _configure_windows_tls, _loading_progress,
)

TRUSTED_REMOTE_SOURCES = {"doclaynet", "cord"}


def mark_downloaded_pages_verified(rows):
    """Treat pages from the configured official corpora as upright by contract."""
    changed = 0
    for row in rows:
        if row.get("source") in TRUSTED_REMOTE_SOURCES and row.get("upright_verified") is not True:
            row["upright_verified"] = True
            changed += 1
    return changed


def replace_with_retry(temporary, destination, attempts=20):
    """Atomically replace a file, tolerating transient Windows file locks."""
    for attempt in range(attempts):
        try:
            temporary.replace(destination)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                raise PermissionError(
                    f"Cannot replace {destination}. Close editors or other programs that have "
                    "the file open, then run the same command again. Prepared pages are preserved."
                ) from exc
            time.sleep(min(0.05 * (2 ** attempt), 0.5))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def save_manifest(path, rows):
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                         encoding="utf-8")
    replace_with_retry(temporary, path)


def save_page(image, folder, max_side):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    digest = hashlib.sha256(encoded).hexdigest()
    relative = f"images/{digest}.png"
    destination = folder / relative
    if not destination.exists():
        destination.write_bytes(encoded)
    return relative, digest


def local_pages(manifest):
    manifest = Path(manifest).resolve()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("upright_verified") is not True or not record.get("group_id"):
            raise ValueError("Local rows require group_id and upright_verified: true")
        path = (manifest.parent / record["path"]).resolve()
        if path.suffix.lower() == ".pdf":
            import fitz
            with fitz.open(path) as doc:
                if doc.needs_pass:
                    raise ValueError(f"Encrypted PDF: {path}")
                pages = [int(record["page"])] if "page" in record else range(len(doc))
                for index in pages:
                    page = doc[index]
                    scale = min(150 / 72, 1800 / max(page.rect.width, page.rect.height))
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                         colorspace=fitz.csRGB, alpha=False)
                    yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples), {
                        **record, "origin": path.name, "page": index}
        else:
            with Image.open(path) as image:
                yield ImageOps.exif_transpose(image).convert("RGB"), {
                    **record, "origin": path.name, "page": 0}


def prepare(args):
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "images").mkdir(exist_ok=True)
    manifest_path = folder / "manifest.jsonl"
    state_path = folder / "prepare_config.json"
    config = {"seed": args.seed, "doclaynet": args.doclaynet, "cord": args.cord,
              "max_side": args.max_side,
              "local_manifest_sha256": hashlib.sha256(Path(args.local_manifest).read_bytes()).hexdigest()
              if args.local_manifest else None}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["config"] != config:
            raise ValueError("Preparation parameters changed. Use another output directory.")
    else:
        state = {"config": config, "revisions": {}, "completed": []}
        save_json(state_path, state)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if manifest_path.exists() else []
    migrated = mark_downloaded_pages_verified(rows)
    if migrated:
        save_manifest(manifest_path, rows)
        print(f"Trusted upright orientation for {migrated} prepared remote pages.", flush=True)
    for name, repo, total in (
        ("doclaynet", DOCLAYNET_REPO, args.doclaynet), ("cord", CORD_REPO, args.cord)
    ):
        if not total:
            continue
        if total < 7:
            raise ValueError("Each remote corpus needs at least 7 pages for three splits")
        _configure_windows_tls()
        from datasets import Image as HFImage, load_dataset
        from huggingface_hub import HfApi
        if name not in state["revisions"]:
            state["revisions"][name] = HfApi().dataset_info(repo).sha
            save_json(state_path, state)
        budgets = [int(total * .7), int(total * .15)]
        budgets.append(total - sum(budgets))
        for split, requested in zip(SPLITS, budgets):
            key = f"{name}/{split}"
            if key in state["completed"]:
                print(f"{key}: using prepared files", flush=True)
                continue
            limit = min(requested, 800 if split == "train" else 100) if name == "cord" else requested
            existing = sum(r["source"] == name and r["split"] == split for r in rows)
            columns = ["image", "metadata"] if name == "doclaynet" else ["image"]
            with _loading_progress(key, limit) as progress:
                progress.update(existing)
                ds = load_dataset(repo, revision=state["revisions"][name], split=split,
                                  streaming=True, columns=columns, batch_size=8)
                ds = ds.cast_column("image", HFImage(decode=False))
                ds = ds.shuffle(seed=args.seed, buffer_size=64)
                for index, item in enumerate(ds):
                    if index < existing:
                        continue
                    if index >= limit:
                        break
                    raw = item["image"]
                    with Image.open(io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]) as image:
                        relative, digest = save_page(image, folder, args.max_side)
                    metadata = item.get("metadata", {})
                    docname = metadata.get("original_filename") or metadata.get("doc_name")
                    group = f"{name}:{docname}" if docname else f"{name}:{split}:{index}"
                    rows.append({
                        "path": relative, "sha256": digest, "split": split, "source": name,
                        "group_id": group, "upright_verified": True, "correction_cw": 0,
                        "category": metadata.get("doc_category", "receipt"),
                        "language": "unknown", "revision": state["revisions"][name],
                        "page": metadata.get("page_no", index),
                    })
                    save_manifest(manifest_path, rows)
                    progress.update(1)
            state["completed"].append(key)
            save_json(state_path, state)
    if args.local_manifest and "local" not in state["completed"]:
        # On retry, replace only the incomplete local portion; remote data is retained.
        rows = [r for r in rows if r["source"] != "local"]
        for image, record in local_pages(args.local_manifest):
            bucket = int(hashlib.sha256(f"{args.seed}:{record['group_id']}".encode()).hexdigest()[:8], 16) % 100
            split = record.get("split", "train" if bucket < 70 else "validation" if bucket < 85 else "test")
            if split not in SPLITS:
                raise ValueError(f"Invalid local split: {split}")
            correction = record.get("correction_cw", 0)
            if correction not in (0, 90, 180, 270):
                raise ValueError("Local correction_cw must be a quarter turn")
            relative, digest = save_page(image, folder, args.max_side)
            rows.append({**record, "path": relative, "sha256": digest, "split": split,
                         "source": "local", "group_id": "local:" + record["group_id"]})
            save_manifest(manifest_path, rows)
        state["completed"].append("local")
        save_json(state_path, state)
    if not rows:
        raise ValueError("No pages prepared. Enable a source or supply --local-manifest.")
    print(f"Prepared {len(rows)} pages: {manifest_path}")
    print("Remote pages are trusted as upright. Optional audit: "
          "python -m training.review_data --manifest " + str(manifest_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/v2")
    parser.add_argument("--doclaynet", type=int, default=5000)
    parser.add_argument("--cord", type=int, default=600)
    parser.add_argument("--local-manifest")
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_side < 448 or args.doclaynet < 0 or args.cord < 0:
        parser.error("max-side must be >=448 and source counts nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
