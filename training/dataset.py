"""RGB document orientation data with published splits and bounded memory."""
import hashlib
import io
import os
import ssl
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

DOCLAYNET_REPO = "docling-project/DocLayNet-v1.2"
CORD_REPO = "naver-clova-ix/cord-v2"
SOURCES = ("hf_mixed", "hf_doclaynet", "hf_cord", "hf_rvlcdip", "hf", "local")
SPLITS = ("train", "validation", "test")


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.05):
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


def _preview(image):
    return ImageOps.exif_transpose(image).convert("RGB").resize(
        (224, 224), Image.Resampling.BILINEAR
    )


@lru_cache(maxsize=1)
def _configure_windows_tls():
    if os.name == "nt":
        import httpx
        import huggingface_hub
        if hasattr(huggingface_hub, "set_client_factory"):
            huggingface_hub.set_client_factory(
                lambda: httpx.Client(verify=ssl.create_default_context(), follow_redirects=True)
            )


@contextmanager
def _loading_progress(description, total, report_interval=15):
    """Report slow network reads even while no new page has arrived."""
    stopped = threading.Event()
    started = time.monotonic()
    with tqdm(total=total, desc=description, unit="page") as progress:
        def report_wait():
            previous_count = progress.n
            while not stopped.wait(report_interval):
                if progress.n == previous_count:
                    tqdm.write(
                        f"{description}: waiting for data ({progress.n}/{total} pages, "
                        f"{time.monotonic() - started:.0f}s elapsed). "
                        "File resolution is complete before page downloads finish."
                    )
                previous_count = progress.n

        reporter = threading.Thread(target=report_wait, daemon=True)
        reporter.start()
        try:
            yield progress
        finally:
            stopped.set()
            reporter.join()


def _load_hf(repo, split, limit, seed=42):
    from datasets import Image as HFImage
    from datasets import load_dataset
    if limit <= 0:
        return
    _configure_windows_tls()
    description = f"{repo.rsplit('/', 1)[-1]}/{split}"
    print(f"Loading {description}: up to {limit} image pages (no PDF payloads).", flush=True)
    with _loading_progress(description, limit) as progress:
        try:
            # Project at the Parquet reader, BEFORE downloading/decoding rows.
            # select_columns() on an iterable drops columns only after reading them.
            ds = load_dataset(
                repo, split=split, streaming=True, columns=["image"], batch_size=8
            )
            ds = ds.cast_column("image", HFImage(decode=False))
            ds = ds.shuffle(seed=seed, buffer_size=min(64, limit))
            for count, item in enumerate(ds, start=1):
                raw = item["image"]
                if isinstance(raw, Image.Image):
                    preview = _preview(raw)
                else:
                    source = io.BytesIO(raw["bytes"]) if raw.get("bytes") is not None else raw["path"]
                    with Image.open(source) as image:
                        preview = _preview(image)
                progress.update(1)
                yield preview
                if count >= limit:
                    return
        except Exception as exc:
            raise RuntimeError(f"Could not load {repo}/{split}: {exc}") from exc


def _source_images(source, split, limit, seed):
    if source in ("hf", "hf_mixed"):
        cord_limit = min(max(1, limit // 5), 800 if split == "train" else 100)
        streams = [
            iter(_load_hf(DOCLAYNET_REPO, split, limit - cord_limit, seed)),
            iter(_load_hf(CORD_REPO, split, cord_limit, seed)),
        ]
        while streams:
            for stream in streams[:]:
                try:
                    yield next(stream)
                except StopIteration:
                    streams.remove(stream)
        return
    repos = {"hf_doclaynet": DOCLAYNET_REPO, "hf_cord": CORD_REPO,
             "hf_rvlcdip": "dvgodoy/rvl_cdip_mini"}
    if source not in repos:
        raise ValueError(f"Unsupported source {source!r}; choose from {SOURCES}")
    yield from _load_hf(repos[source], split, limit, seed)


def _load_local(data_dir, limit, seed=42, split=None):
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"Document directory does not exist: {root}")
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in extensions)
    random.Random(seed).shuffle(paths)
    count = 0
    for path in paths:
        # Group pages by original file, independent of listing order.
        key = f"{seed}:{path.relative_to(root).as_posix()}"
        bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
        assigned = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
        if split is not None and assigned != split:
            continue
        if path.suffix.lower() == ".pdf":
            import fitz
            with fitz.open(path) as doc:
                if doc.needs_pass:
                    raise ValueError(f"Password-protected training PDF: {path}")
                for page in doc:
                    scale = min(150 / 72, 1600 / max(page.rect.width, page.rect.height))
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                         colorspace=fitz.csRGB, alpha=False)
                    yield _preview(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                    count += 1
                    if count >= limit:
                        return
        else:
            with Image.open(path) as image:
                yield _preview(image)
            count += 1
            if count >= limit:
                return


class OrientationDataset(Dataset):
    """Labels 0/1/2/3 represent 0/90/180/270 degrees COUNTERCLOCKWISE.

    Source pages must be upright. Only 224x224 RGB previews are kept in memory.
    """
    def __init__(self, source="hf_mixed", data_dir=None, num_images=5000,
                 is_train=True, images=None, split="train", seed=42):
        if num_images <= 0:
            raise ValueError("num_images must be positive.")
        self.is_train = is_train
        if images is not None:
            self.images = [_preview(image) for image in images]
        else:
            if source == "local":
                if not data_dir:
                    raise ValueError("--data-dir is required for local data.")
                stream = _load_local(data_dir, num_images, seed, split)
            else:
                stream = _source_images(source, split, num_images, seed)
            self.images = list(stream)
            print(f"{source}/{split}: {len(self.images)} base pages, {len(self.images)*4} rotations")
        if not self.images:
            raise ValueError(f"No usable images for {source}/{split}. Add more upright documents.")
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.aug_transforms = transforms.Compose([
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
        ])
        self.noise_transform = AddGaussianNoise()

    def __len__(self):
        return len(self.images) * 4

    def __getitem__(self, idx):
        image = self.images[idx // 4]
        label = idx % 4
        image = image.rotate(label * 90, expand=True)
        if self.is_train:
            image = self.aug_transforms(image)
        tensor = self.base_transforms(image)
        if self.is_train and torch.rand(1).item() < 0.5:
            tensor = self.noise_transform(tensor)
        return tensor, label


def create_splits(source="hf_mixed", data_dir=None, num_images=5000, seed=42, splits=SPLITS):
    """Allocate a 70/15/15 page budget within disjoint published/file splits."""
    if num_images < 7:
        raise ValueError("num_images must be at least 7 for three nonempty splits.")
    budgets = [int(num_images * .70), int(num_images * .15)]
    budgets.append(num_images - sum(budgets))
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError(f"splits must contain names from {SPLITS}.")
    split_budgets = dict(zip(SPLITS, budgets))
    return tuple(
        OrientationDataset(source=source, data_dir=data_dir, num_images=split_budgets[split],
                           is_train=split == "train", split=split, seed=seed)
        for split in splits
    )


# Model v2 uses an audited, disk-backed manifest. The legacy loader above remains
# available for reproducing v1 experiments; it must never resize v2 inputs to 224.
import json
from collections import Counter

import numpy as np
from torch.utils.data import Sampler

from app.services.preprocessing import image_tensor, prepare_image


class RandomShadow:
    """Soft lighting gradient without changing character or page orientation."""
    def __call__(self, image):
        if random.random() >= .35:
            return image
        array = np.asarray(image, dtype=np.float32) / 255
        yy, xx = np.mgrid[-1:1:complex(image.height), -1:1:complex(image.width)]
        angle = random.uniform(0, 2 * np.pi)
        gradient = (np.cos(angle) * xx + np.sin(angle) * yy + 2) / 4
        darkness = random.uniform(.15, .45)
        shade = 1 - darkness * gradient
        return Image.fromarray(np.uint8(np.clip(array * shade[..., None], 0, 1) * 255))


def read_manifest(manifest_path, allow_unreviewed=False):
    path = Path(manifest_path).resolve()
    records = []
    group_splits = {}
    hashes = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("exclude", False):
            continue
        required = {"path", "split", "source", "group_id", "upright_verified"}
        if not required <= record.keys() or record["split"] not in SPLITS:
            raise ValueError(f"Invalid manifest row {line_no}")
        if record["upright_verified"] is not True and not allow_unreviewed:
            raise ValueError(f"Unreviewed uprightness at row {line_no}. Review the manifest first.")
        rotation = record.get("correction_cw", 0)
        if rotation not in (0, 90, 180, 270):
            raise ValueError(f"Invalid correction_cw at row {line_no}")
        file_path = (path.parent / record["path"]).resolve()
        if not file_path.is_file():
            raise ValueError(f"Missing image at row {line_no}: {file_path}")
        # All pages/templates with this group must stay in the same split.
        group = record["group_id"]
        if group in group_splits and group_splits[group] != record["split"]:
            raise ValueError(f"Group leaks across splits: {group}")
        group_splits[group] = record["split"]
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError(f"Image checksum changed at row {line_no}: {file_path}")
        if digest in hashes:
            if hashes[digest] != record["split"]:
                raise ValueError(f"Duplicate image leaks across splits: {file_path}")
            continue  # Identical export within one split is not independent data.
        hashes[digest] = record["split"]
        record["resolved_path"] = file_path
        record["sha256"] = digest
        records.append(record)
    if not records:
        raise ValueError("Manifest has no usable pages")
    return records


class ManifestOrientationDataset(Dataset):
    def __init__(self, manifest_path, split, input_size=384, allow_unreviewed=False,
                 records=None):
        if input_size not in (224, 384, 448):
            raise ValueError("Supported input_size: 224, 384, or 448")
        all_records = records if records is not None else read_manifest(
            manifest_path, allow_unreviewed)
        self.records = [record for record in all_records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No pages in manifest split {split}")
        self.input_size = input_size
        self.is_train = split == "train"
        self.augment = transforms.Compose([
            transforms.RandomAffine(degrees=3, translate=(.025, .025), scale=(.95, 1.05),
                                    interpolation=transforms.InterpolationMode.BILINEAR,
                                    fill=(255, 255, 255)),
            transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.2, hue=.025),
            RandomShadow(),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(.1, 1.2))], p=.2),
        ])

    def __len__(self):
        return len(self.records) * 4

    def __getitem__(self, index):
        record = self.records[index // 4]
        label = index % 4
        with Image.open(record["resolved_path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image = image.rotate(-record.get("correction_cw", 0), expand=True)
            image = image.rotate(label * 90, expand=True)  # legacy CCW class contract
            image = prepare_image(image, self.input_size, "letterbox")
        if self.is_train:
            image = self.augment(image)
            if random.random() < .25:
                array = np.asarray(image, dtype=np.float32)
                array += np.random.normal(0, random.uniform(1, 5), array.shape)
                image = Image.fromarray(np.uint8(np.clip(array, 0, 255)))
        return torch.from_numpy(image_tensor(image)), label


class BalancedSourceSampler(Sampler):
    """Sample base pages by source weight, then emit all four angles per page."""
    def __init__(self, dataset, source_weights, seed=42):
        counts = Counter(record["source"] for record in dataset.records)
        if set(source_weights) != set(counts) or any(not np.isfinite(w) or w <= 0 for w in source_weights.values()):
            raise ValueError(f"source weights must cover exactly {sorted(counts)} with positive values")
        self.weights = torch.tensor([source_weights[r["source"]] / counts[r["source"]]
                                     for r in dataset.records], dtype=torch.double)
        self.seed = seed
        self.epoch = 0
        self.base_count = len(dataset.records)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.base_count * 4

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pages = torch.multinomial(self.weights, self.base_count, replacement=True,
                                  generator=generator).tolist()
        indices = [page * 4 + angle for page in pages for angle in range(4)]
        order = torch.randperm(len(indices), generator=generator).tolist()
        return iter([indices[i] for i in order])
