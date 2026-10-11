"""Evaluation for the YOLOv8n vs YOLOv8s comparison.

Two levels of evaluation are produced for every model:

1. Detection level (standard for object detection) - via ``model.val(split="test")``:
   Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 + per-class AP.

2. Image level (so the classification metrics the assignment asks for -
   Accuracy and ROC-AUC - can be reported):
   * ground truth of an image = the class with the most GT boxes in it
     (most images contain a single product)
   * prediction of an image   = class of the highest-confidence box, if that box
     has conf >= 0.25; otherwise "no detection" (counted as wrong)
   * score vector of an image = for every class, the max confidence of any box
     predicted as that class (0 if none) -> used for one-vs-rest ROC / AUC.

Usage (CLI, after training)::

    python src/evaluate.py --data datasets/sell-products/data_abs.yaml \
        --weights runs/detect/yolov8n/weights/best.pt runs/detect/yolov8s/weights/best.pt \
        --out results
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from plotstyle import GT_COLOR, PRED_COLOR, TEXT_2, model_color
from data import image_label, label_path, list_images, load_names, read_labels

DECISION_CONF = 0.25  # threshold for "the model detected something"
NO_DET = -1           # image-level "no detection" label


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _f1(p, r):
    return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _split_img_dir(data_yaml: Path, split: str) -> Path:
    import yaml

    cfg = yaml.safe_load(Path(data_yaml).read_text())
    rel = cfg.get(split) or cfg["val"]
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg.get("path", Path(data_yaml).parent)) / p


def model_info(weights: str | Path, imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    m = YOLO(str(weights))
    params = sum(p.numel() for p in m.model.parameters())
    gflops = None
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = round(float(get_flops(m.model, imgsz)), 1) or None
    except Exception:
        pass
    return {
        "weights_mb": round(Path(weights).stat().st_size / 2**20, 1),
        "params_m": round(params / 1e6, 2),
        "gflops": gflops,
    }


# --------------------------------------------------------------------------- #
# 1) detection-level metrics
# --------------------------------------------------------------------------- #
def detection_metrics(weights, data_yaml, names, split="test", imgsz=640,
                      project="runs/detect", name="eval") -> dict:
    """Ultralytics val on the held-out split. P/R are at IoU 0.5 and at the
    confidence that maximises mean F1 (Ultralytics default); mAP uses conf=0.001."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=16,
                  conf=0.001, iou=0.7, plots=True, project=str(project), name=name,
                  exist_ok=True, verbose=False)
    box = m.box
    p, r = float(box.mp), float(box.mr)
    idx = [int(i) for i in box.ap_class_index]
    per_class = pd.DataFrame({
        "class": [names[i] for i in idx],
        "precision": np.round(box.p, 4),
        "recall": np.round(box.r, 4),
        "ap50": np.round(box.ap50, 4),
        "ap50_95": np.round(box.ap, 4),
    })
    per_class["f1"] = np.round([_f1(a, b) for a, b in zip(box.p, box.r)], 4)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(_f1(p, r), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "val_speed_ms": {k: round(float(v), 2) for k, v in m.speed.items()},
        "save_dir": str(m.save_dir),
        "per_class": per_class,
    }


# --------------------------------------------------------------------------- #
# 2) image-level predictions
# --------------------------------------------------------------------------- #
def collect_predictions(weights, img_dir, n_classes, imgsz=640, warmup=10,
                        max_images=None):
    """Run the model image-by-image (batch=1, like a real kiosk/camera) and keep
    everything needed for image-level metrics, ROC and example grids."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = list_images(img_dir)
    if max_images:
        imgs = imgs[:max_images]
    for p in imgs[:warmup]:  # GPU warm-up, not timed
        model.predict(str(p), imgsz=imgsz, conf=0.001, verbose=False)

    records, scores = [], np.zeros((len(imgs), n_classes), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(imgs):
        gt = read_labels(label_path(p))
        r = model.predict(str(p), imgsz=imgsz, conf=0.001, max_det=100, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxyn = r.boxes.xyxyn.cpu().numpy()
        np.maximum.at(scores[i], cls, conf)  # max conf per class
        keep = np.argsort(-conf)
        keep = keep[conf[keep] >= DECISION_CONF][:5]
        top = int(cls[keep[0]]) if len(keep) else NO_DET
        records.append({
            "image": str(p),
            "y_true": image_label(gt),
            "y_pred": top,
            "pred_conf": float(conf[keep[0]]) if len(keep) else 0.0,
            "n_gt": len(gt),
            "n_gt_classes": len({g[0] for g in gt}),
            "gt_boxes": [list(map(float, g)) for g in gt],
            "pred_boxes": [[int(cls[k]), float(conf[k]), *map(float, xyxyn[k])] for k in keep],
            "infer_ms": float(r.speed["inference"]),
            "total_ms": float(sum(r.speed.values())),
        })
    print(f"  {len(imgs)} images in {time.time() - t0:.0f}s")
    return records, scores


def image_level_metrics(y_true, y_pred, scores, names) -> tuple[dict, pd.DataFrame]:
    """Accuracy, macro P/R/F1 and one-vs-rest ROC-AUC (macro + micro)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true >= 0  # drop background images (no GT box)
    y_true, y_pred, scores = y_true[mask], y_pred[mask], scores[mask]
    present = np.unique(y_true)

    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average=None, zero_division=0)
    per_class = pd.DataFrame({"class": [names[c] for c in present], "support": sup,
                              "img_precision": p.round(4), "img_recall": r.round(4),
                              "img_f1": f.round(4)})

    aucs = []
    for c in present:
        yb = (y_true == c).astype(int)
        aucs.append(roc_auc_score(yb, scores[:, c]) if 0 < yb.sum() < len(yb) else np.nan)
    per_class["auc"] = np.round(aucs, 4)

    onehot = (y_true[:, None] == present[None, :]).astype(int)
    micro_auc = roc_auc_score(onehot.ravel(), scores[:, present].ravel())

    out = {
        "n_images": int(mask.sum()),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "img_precision_macro": round(float(p.mean()), 4),
        "img_recall_macro": round(float(r.mean()), 4),
        "img_f1_macro": round(float(f.mean()), 4),
        "auc_macro_ovr": round(float(np.nanmean(aucs)), 4),
        "auc_micro": round(float(micro_auc), 4),
        "no_detection_rate": round(float((y_pred == NO_DET).mean()), 4),
    }
    return out, per_class


def roc_points(y_true, scores):
    """Micro-average ROC curve over all (image, class) pairs."""
    y_true = np.asarray(y_true)
    mask = y_true >= 0
    present = np.unique(y_true[mask])
    onehot = (y_true[mask][:, None] == present[None, :]).astype(int)
    fpr, tpr, _ = roc_curve(onehot.ravel(), scores[mask][:, present].ravel())
    return fpr, tpr


def top_confusions(y_true, y_pred, names, k=10) -> pd.DataFrame:
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred) if t >= 0 and t != p)
    total = Counter(t for t in y_true if t >= 0)
    rows = [{"true": names[t], "pred": names[p] if p >= 0 else "(no detection)",
             "count": n, "pct_of_true_class": round(100 * n / total[t], 1)}
            for (t, p), n in pairs.most_common(k)]
    return pd.DataFrame(rows, columns=["true", "pred", "count", "pct_of_true_class"])


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_roc(curves: dict, out_path):
    """curves = {model: (fpr, tpr, auc_micro, auc_macro)}; full + zoomed panel."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (name, (fpr, tpr, a_mi, a_ma)) in enumerate(curves.items()):
        for ax in axes:
            ax.plot(fpr, tpr, color=model_color(name, i),
                    label=f"{name}  micro-AUC {a_mi:.4f} | macro-AUC {a_ma:.4f}")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_2, label="random")
    axes[0].set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate",
                ylabel="True positive rate", title="ROC (one-vs-rest, micro-average)")
    axes[1].set_xscale("log")
    axes[1].set(xlim=(1e-4, 1), ylim=(0.5, 1.005), xlabel="False positive rate (log)",
                title="Zoom: low-FPR region")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusions(conf_tables: dict, out_path):
    """Small multiples: top-10 confused (true -> pred) pairs per model."""
    n = len(conf_tables)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.6), squeeze=False)
    for i, (ax, (name, df)) in enumerate(zip(axes[0], conf_tables.items())):
        df = df.iloc[::-1]
        labels = [f"{t}  ->  {p}" for t, p in zip(df["true"], df["pred"])]
        ax.barh(labels, df["count"], color=model_color(name, i), height=0.7)
        for y, (c, pct) in enumerate(zip(df["count"], df["pct_of_true_class"])):
            ax.text(c, y, f" {c} ({pct}%)", va="center", fontsize=8, color=TEXT_2)
        ax.set_title(f"{name}: top confused pairs (true -> predicted)")
        ax.set_xlabel("test images")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
        if df.empty:
            ax.text(0.5, 0.5, "no errors", ha="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_examples(records, names, correct: bool, out_path, n=8, seed=0, title=""):
    """2x4 grid. Green = ground truth, red = prediction (conf >= 0.25)."""
    from PIL import Image

    pool = [r for r in records if r["y_true"] >= 0 and (r["y_true"] == r["y_pred"]) == correct]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)] if pool else []
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rec in zip(axes.ravel(), pick):
        im = Image.open(rec["image"]).convert("RGB")
        W, H = im.size
        ax.imshow(im)
        for c, cx, cy, w, h in rec["gt_boxes"]:
            ax.add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H,
                                       fill=False, lw=2, ec=GT_COLOR))
        for c, cf, x0, y0, x1, y1 in rec["pred_boxes"][:1]:  # top-1 = image prediction
            ax.add_patch(plt.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                                       fill=False, lw=2, ec=PRED_COLOR, ls="--"))
            ax.text(x0 * W, y0 * H, f"{names[c]} {cf:.2f}", fontsize=7, color="white",
                    va="bottom", bbox=dict(fc=PRED_COLOR, ec="none", pad=1))
        pred = names[rec["y_pred"]] if rec["y_pred"] >= 0 else "(no detection)"
        ok = rec["y_true"] == rec["y_pred"]
        ax.set_title(f"GT: {names[rec['y_true']]}\nPred: {pred} ({rec['pred_conf']:.2f})",
                     fontsize=9, color=GT_COLOR if ok else PRED_COLOR)
    fig.suptitle(title + "   [green = ground truth, red dashed = top-1 prediction]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_training_curves(run_dirs: dict, out_path):
    """mAP50-95 (val) and box loss per epoch for each model, read from results.csv."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    specs = [("metrics/mAP50(B)", "mAP@0.5 (val)"), ("metrics/mAP50-95(B)", "mAP@0.5:0.95 (val)"),
             ("val/box_loss", "box loss (val)")]
    for i, (name, d) in enumerate(run_dirs.items()):
        f = Path(d) / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        x = df["epoch"] + (1 if df["epoch"].min() == 0 else 0)
        for ax, (col, t) in zip(axes, specs):
            if col in df:
                ax.plot(x, df[col], color=model_color(name, i), label=name)
                ax.set_title(t)
                ax.set_xlabel("epoch")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_comparison(table: pd.DataFrame, out_path):
    """Accuracy-type metrics (shared 0-1 axis) and speed as two separate panels."""
    metrics = ["precision", "recall", "f1", "map50", "map50_95", "accuracy", "auc_macro_ovr"]
    labels = ["Precision", "Recall", "F1", "mAP50", "mAP50-95", "Accuracy", "AUC"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [3.2, 1]})
    n = len(table)
    w = 0.8 / n
    x = np.arange(len(metrics))
    for i, (name, row) in enumerate(table.iterrows()):
        vals = [row[m] for m in metrics]
        bars = axes[0].bar(x + (i - (n - 1) / 2) * w, vals, w * 0.92,
                           color=model_color(name, i), label=name)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7, color=TEXT_2, padding=1)
        axes[1].bar(i, row["infer_ms_per_img"], 0.6, color=model_color(name, i))
        axes[1].text(i, row["infer_ms_per_img"], f"{row['infer_ms_per_img']:.1f}",
                     ha="center", va="bottom", fontsize=9, color=TEXT_2)
    lo = max(0.0, float(np.nanmin(table[metrics].values)) - 0.05)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(lo, 1.0),
                title="Test-set metrics (higher is better)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=n)
    axes[0].grid(axis="x", visible=False)
    axes[1].set(xticks=range(n), xticklabels=list(table.index),
                ylabel="ms / image (batch=1)", title="Inference time (lower is better)")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_worst_classes(per_class: dict, out_path, k=15):
    """AP50-95 of the k hardest classes (by the larger model), both models side by side."""
    names = list(per_class)
    ref = per_class[names[-1]].set_index("class")["ap50_95"].sort_values().head(k)
    fig, ax = plt.subplots(figsize=(8, 0.34 * k + 1.2))
    h = 0.8 / len(names)
    y = np.arange(len(ref))
    for i, name in enumerate(names):
        s = per_class[name].set_index("class")["ap50_95"].reindex(ref.index)
        ax.barh(y + (i - (len(names) - 1) / 2) * h, s.values, h * 0.92,
                color=model_color(name, i), label=name)
    ax.set_yticks(y)
    ax.set_yticklabels(ref.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AP@0.5:0.95 (test)")
    ax.set_title(f"{k} hardest classes")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
# Ultralytics >= 8.3.x prefixes curve plots with "Box" (BoxPR_curve.png); older
# versions don't. Copy whichever exists under one stable name.
ULTRA_PLOTS = {"confusion_matrix_normalized.png": ["confusion_matrix_normalized.png"],
               "PR_curve.png": ["BoxPR_curve.png", "PR_curve.png"],
               "F1_curve.png": ["BoxF1_curve.png", "F1_curve.png"],
               "P_curve.png": ["BoxP_curve.png", "P_curve.png"],
               "R_curve.png": ["BoxR_curve.png", "R_curve.png"]}


def evaluate_all(weights: dict, data_yaml, out_dir="results", split="test", imgsz=640,
                 project="runs/detect", train_dirs: dict | None = None, max_images=None):
    """weights = {"yolov8n": path/to/best.pt, "yolov8s": ...}. Writes every table and
    figure used in the README into out_dir and returns the comparison table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(data_yaml)
    img_dir = _split_img_dir(data_yaml, split)
    print(f"evaluating on {img_dir}")

    rows, per_class, curves, confs, preds = {}, {}, {}, {}, {}
    for name, w in weights.items():
        print(f"[{name}] detection metrics ...")
        det = detection_metrics(w, data_yaml, names, split, imgsz, project, f"{name}_{split}")
        print(f"[{name}] image-level predictions ...")
        recs, scores = collect_predictions(w, img_dir, len(names), imgsz, max_images=max_images)
        y_t = [r["y_true"] for r in recs]
        y_p = [r["y_pred"] for r in recs]
        img, img_cls = image_level_metrics(y_t, y_p, scores, names)
        info = model_info(w, imgsz)
        ms = np.array([r["infer_ms"] for r in recs])
        tot = np.array([r["total_ms"] for r in recs])

        pc = det.pop("per_class").merge(img_cls, on="class", how="left")
        pc.to_csv(out / f"per_class_{name}.csv", index=False)
        per_class[name] = pc
        cf = top_confusions(y_t, y_p, names)
        cf.to_csv(out / f"top_confusions_{name}.csv", index=False)
        confs[name] = cf
        fpr, tpr = roc_points(y_t, scores)
        curves[name] = (fpr, tpr, img["auc_micro"], img["auc_macro_ovr"])
        preds[name] = recs

        rows[name] = {**{k: v for k, v in det.items() if k not in ("save_dir", "val_speed_ms")},
                      **img,
                      "infer_ms_per_img": round(float(np.median(ms)), 2),
                      "e2e_ms_per_img": round(float(np.median(tot)), 2),
                      "fps": round(1000 / float(np.median(tot)), 1),
                      **info}
        json.dump({"model": name, **rows[name], "val_speed_ms": det["val_speed_ms"]},
                  open(out / f"metrics_{name}.json", "w"), indent=2)
        pd.DataFrame([{k: v for k, v in r.items() if k not in ("gt_boxes", "pred_boxes")}
                      for r in recs]).to_csv(out / f"predictions_{name}.csv", index=False)

        # copy the Ultralytics plots we reference in the README
        for dst, candidates in ULTRA_PLOTS.items():
            src = next((Path(det["save_dir"]) / c for c in candidates
                        if (Path(det["save_dir"]) / c).exists()), None)
            if src:
                shutil.copy(src, out / f"{name}_{dst}")
        if train_dirs and name in train_dirs and (Path(train_dirs[name]) / "results.png").exists():
            shutil.copy(Path(train_dirs[name]) / "results.png", out / f"{name}_results.png")

        plot_examples(recs, names, True, out / f"examples_correct_{name}.png",
                      title=f"{name}: correct predictions (random 8)")
        plot_examples(recs, names, False, out / f"examples_wrong_{name}.png",
                      title=f"{name}: wrong predictions (random 8)")

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(out / "comparison.csv")
    plot_comparison(table, out / "comparison_bar.png")
    plot_roc(curves, out / "roc_curve.png")
    plot_confusions(confs, out / "top_confusions.png")
    plot_worst_classes(per_class, out / "hardest_classes.png")
    if train_dirs:
        plot_training_curves(train_dirs, out / "training_curves.png")

    # where do the two models disagree?
    if len(preds) == 2:
        (a, ra), (b, rb) = preds.items()
        both = pd.DataFrame({"image": [r["image"] for r in ra], "y_true": [r["y_true"] for r in ra],
                             f"ok_{a}": [r["y_true"] == r["y_pred"] for r in ra],
                             f"ok_{b}": [r["y_true"] == r["y_pred"] for r in rb]})
        both = both[both.y_true >= 0]
        dis = {
            f"only_{a}_correct": int((both[f"ok_{a}"] & ~both[f"ok_{b}"]).sum()),
            f"only_{b}_correct": int((~both[f"ok_{a}"] & both[f"ok_{b}"]).sum()),
            "both_wrong": int((~both[f"ok_{a}"] & ~both[f"ok_{b}"]).sum()),
            "both_correct": int((both[f"ok_{a}"] & both[f"ok_{b}"]).sum()),
        }
        json.dump(dis, open(out / "disagreement.json", "w"), indent=2)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", nargs="+", required=True,
                    help="paths like runs/detect/yolov8n/weights/best.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    weights = {Path(w).parent.parent.name: w for w in a.weights}
    train_dirs = {k: Path(w).parent.parent for k, w in weights.items()}
    print(evaluate_all(weights, a.data, a.out, a.split, a.imgsz, train_dirs=train_dirs).T)


if __name__ == "__main__":
    main()
