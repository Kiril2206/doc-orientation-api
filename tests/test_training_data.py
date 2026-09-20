"""Validate source splits without downloading a corpus."""
from unittest.mock import patch
from PIL import Image
from training.dataset import create_splits, _source_images, _load_local


def test_official_splits_are_kept():
    seen = []
    def source(name, split, limit, seed):
        seen.append((name, split, limit, seed))
        for _ in range(limit):
            yield Image.new("RGB", (224, 224))
    with patch("training.dataset._source_images", side_effect=source):
        train, val, test = create_splits(num_images=20, seed=7)
    assert seen == [("hf_mixed", "train", 14, 7),
                    ("hf_mixed", "validation", 3, 7),
                    ("hf_mixed", "test", 3, 7)]
    assert (len(train), len(val), len(test)) == (56, 12, 12)
    assert train.is_train and not val.is_train and not test.is_train


def test_mixed_source_fails_visibly():
    def broken(*args, **kwargs):
        raise RuntimeError("Source unavailable")
        yield
    with patch("training.dataset._load_hf", side_effect=broken):
        import pytest
        with pytest.raises(RuntimeError, match="Source unavailable"):
            list(_source_images("hf_mixed", "train", 20, 42))


def test_local_pages_do_not_cross_splits(tmp_path):
    import fitz
    for i in range(30):
        with fitz.open() as pdf:
            for _ in range(2):
                page = pdf.new_page(width=100, height=150)
                page.draw_rect(page.rect, color=None, fill=(i/30, 0, 0))
            pdf.save(tmp_path / f"{i}.PDF")
    sets = []
    for split in ("train", "validation", "test"):
        images = list(_load_local(str(tmp_path), 100, seed=3, split=split))
        colors = [im.getpixel((100, 100)) for im in images]
        assert len(colors) == 2 * len(set(colors))
        sets.append(set(colors))
    assert all(sets)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == 30


def test_evaluation_loads_only_test_split():
    seen = []
    def source(name, split, limit, seed):
        seen.append((split, limit))
        yield Image.new("RGB", (224, 224))
    with patch("training.dataset._source_images", side_effect=source):
        (test,) = create_splits(num_images=20, splits=("test",))
    assert seen == [("test", 3)]
    assert not test.is_train


def test_parquet_reader_excludes_pdf_payloads(tmp_path):
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    import datasets
    from training.dataset import _load_hf

    encoded = io.BytesIO()
    Image.new("RGB", (40, 60), (20, 100, 200)).save(encoded, format="PNG")
    parquet = tmp_path / "pages.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"image": {"bytes": encoded.getvalue(), "path": None}, "pdf": b"unused PDF"}
        for _ in range(4)
    ]), parquet)
    real_load = datasets.load_dataset
    columns_read = []

    def local_load(repo, **kwargs):
        ds = real_load("parquet", data_files={"train": str(parquet)},
                       cache_dir=str(tmp_path / "cache"), **kwargs)
        columns_read.extend(ds.column_names)
        return ds

    with patch("datasets.load_dataset", side_effect=local_load), \
         patch("training.dataset._configure_windows_tls"):
        images = list(_load_hf("local/test", "train", 2))
    assert columns_read == ["image"]
    assert len(images) == 2
    assert all(im.size == (224, 224) and im.mode == "RGB" for im in images)
    assert images[0].getpixel((100, 100)) == (20, 100, 200)


def test_loading_wait_message_and_thread_cleanup():
    import threading
    from training.dataset import _loading_progress
    from tqdm import tqdm

    reported = threading.Event()
    with patch.object(tqdm, "write", side_effect=lambda *a, **k: reported.set()):
        with _loading_progress("slow source", 2, report_interval=.01) as progress:
            assert reported.wait(2), "Slow I/O should produce a status message."
            progress.update(1)
    assert progress.disable  # progress context has closed
