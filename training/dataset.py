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
