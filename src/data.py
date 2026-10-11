"""Dataset download (Roboflow), data.yaml fix-up, label parsing and EDA.

Dataset: "Sell Products" by O2O Minimart on Roboflow Universe (CC BY 4.0)
https://universe.roboflow.com/o2o-minimart/sell-products
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}  # yolo key -> roboflow folder

WORKSPACE = "o2o-minimart"
PROJECT = "sell-products"


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def get_api_key() -> str:
    """Colab secret -> Kaggle secret -> env var / .env -> prompt. Never hard-code it."""
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("ROBOFLOW_API_KEY")
        if key:
            return key
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
        if key:
            return key
    except Exception:
        pass
    if not os.environ.get("ROBOFLOW_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                os.environ["ROBOFLOW_API_KEY"] = line.split("=", 1)[1].strip()
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"]
    from getpass import getpass

    return getpass("Roboflow API key: ")


def download_dataset(version: int = 3, dest: str | Path = "datasets/sell-products") -> Path:
    """Download the YOLOv8-format export once; reuse it if already on disk."""
    dest = Path(dest)
    if (dest / "data.yaml").exists():
        print(f"dataset already at {dest}")
        return dest
    from roboflow import Roboflow

    rf = Roboflow(api_key=get_api_key())
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(version).download(
        "yolov8", location=str(dest))
    return Path(ds.location)


def fix_data_yaml(root: str | Path) -> Path:
    """Roboflow's data.yaml uses '../train/images' style paths that break when the
    folder moves. Write data_abs.yaml with an absolute root + relative splits."""
    root = Path(root).resolve()
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    out = {"path": str(root), "names": {i: n for i, n in enumerate(names)}}
    for key, folder in SPLITS.items():
        if (root / folder / "images").is_dir():
            out[key] = f"{folder}/images"
    if "test" not in out:
        print("WARNING: no test split found - evaluation will fall back to 'val'")
    path = root / "data_abs.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    return path


def load_names(data_yaml: str | Path) -> list[str]:
    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def list_images(img_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)


def label_path(img_path: Path) -> Path:
    """<split>/images/x.jpg -> <split>/labels/x.txt"""
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def read_labels(txt: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt -> [(cls, cx, cy, w, h)] normalised. Polygon rows are converted to
    their bounding box so the function works on any Roboflow export."""
    if not txt.exists():
        return []
    rows = []
    for line in txt.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        nums = list(map(float, v[1:]))
        if len(nums) == 4:
            rows.append((c, *nums))
        else:  # polygon x1 y1 x2 y2 ...
            xs, ys = nums[0::2], nums[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            rows.append((c, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return rows


def image_label(rows) -> int:
    """Image-level ground truth = class with most boxes (tie -> largest total area).
    -1 for background images (no boxes)."""
    if not rows:
        return -1
    cnt, area = Counter(), Counter()
    for c, _, _, w, h in rows:
        cnt[c] += 1
        area[c] += w * h
    return max(cnt, key=lambda c: (cnt[c], area[c]))


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def run_eda(root: str | Path, names: list[str], out_dir: str | Path = "results") -> dict:
    """Images / boxes per split, class distribution, boxes per image, box size.
    Writes CSVs + PNGs into out_dir and returns a summary dict."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    from matplotlib.ticker import MaxNLocator

    from plotstyle import SERIES

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows, class_counts, per_img, box_area = [], {}, {}, {}
    img_sizes = Counter()
    for key, folder in SPLITS.items():
        img_dir = root / folder / "images"
        if not img_dir.is_dir():
            continue
        imgs = list_images(img_dir)
        cc, nbox, areas, n_single = Counter(), [], [], 0
        for i, p in enumerate(imgs):
            rows = read_labels(label_path(p))
            nbox.append(len(rows))
            n_single += len({r[0] for r in rows}) == 1
            for c, _, _, w, h in rows:
                cc[c] += 1
                areas.append(w * h)
            if i < 200:  # sample image resolutions
                with Image.open(p) as im:
                    img_sizes[im.size] += 1
        class_counts[folder] = cc
        per_img[folder] = nbox
        box_area[folder] = areas
        split_rows.append({
            "split": folder,
            "images": len(imgs),
            "boxes": int(sum(nbox)),
            "background_images": int(sum(n == 0 for n in nbox)),
            "single_class_images_pct": round(100 * n_single / max(len(imgs), 1), 1),
            "boxes_per_image_mean": round(float(np.mean(nbox)) if nbox else 0, 2),
            "boxes_per_image_max": int(max(nbox)) if nbox else 0,
            "classes_present": len(cc),
            "box_area_median_pct": round(100 * float(np.median(areas)), 1) if areas else 0,
        })

    splits = pd.DataFrame(split_rows)
    splits.to_csv(out_dir / "eda_splits.csv", index=False)

    cls = pd.DataFrame({f: [class_counts[f].get(i, 0) for i in range(len(names))]
                        for f in class_counts}, index=names)
    cls.index.name = "class"
    cls["total"] = cls.sum(axis=1)
    cls = cls.sort_values("total", ascending=False)
    cls.to_csv(out_dir / "eda_class_counts.csv")

    # -- class distribution (train) ----------------------------------------
    col = "train" if "train" in cls else cls.columns[0]
    s = cls[col].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(12, len(s) * 0.16), 4.5))
    ax.bar(range(len(s)), s.values, color=SERIES[0], width=0.8)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=90, fontsize=6)
    ax.set_xlim(-0.6, len(s) - 0.4)
    ax.set_ylabel("bounding boxes")
    ax.grid(axis="x", visible=False)
    imb = s.max() / max(s[s > 0].min(), 1)
    ax.set_title(f"Class distribution - {col} split ({len(s)} classes, "
                 f"max/min = {imb:.1f}x)")
    fig.savefig(out_dir / "eda_class_distribution.png")
    plt.close(fig)

    # -- boxes per image + box size ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    nb = per_img.get("train", next(iter(per_img.values())))
    ar = box_area.get("train", next(iter(box_area.values())))
    axes[0].hist(nb, bins=range(0, max(nb) + 2), color=SERIES[0], rwidth=0.85, align="left")
    axes[0].set_title("Boxes per image (train)")
    axes[0].set_xlabel("boxes in image")
    axes[0].set_ylabel("images")
    axes[1].hist(np.array(ar) * 100, bins=40, color=SERIES[0], rwidth=0.9)
    axes[1].set_title("Box area as % of image (train)")
    axes[1].set_xlabel("% of image area")
    axes[1].set_ylabel("boxes")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    for a in axes:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_dir / "eda_boxes.png")
    plt.close(fig)

    summary = {
        "splits": splits.to_dict(orient="records"),
        "n_classes": len(names),
        "train_imbalance_ratio": round(float(imb), 2),
        "rarest_classes": s[s > 0].tail(5).to_dict(),
        "most_common_classes": s.head(5).to_dict(),
        "image_sizes_sample": {f"{w}x{h}": n for (w, h), n in img_sizes.most_common(5)},
    }
    return summary
