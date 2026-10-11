"""Evaluation for the YOLOv8n vs YOLOv8s comparison.

Two levels of evaluation are produced for every model:

1. Detection level (standard for object detection) - via ``model.val(split="test")``:
   Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 + per-class AP. A second val pass at
   conf=0.25 is used only for the confusion matrix (at conf=0.001 the matrix is
   dominated by near-zero-confidence duplicate boxes and is not readable).

2. Image level (so the classification metrics the assignment asks for -
   Accuracy and ROC-AUC - can be reported):
   * ground truth of an image = the class with the most GT boxes in it
     (most images contain a single product)
   * prediction of an image   = class of the highest-confidence box, if that box
     has conf >= 0.25; otherwise "no detection" (counted as wrong)
   * score vector of an image = for every class, the max confidence of any box
     predicted as that class (0 if none) -> used for one-vs-rest ROC / AUC.
   Note: val uses multi-label NMS (max_det 300) while predict uses single-label NMS,
   so an image can count as found in mAP but as "no detection" at image level.

Box-level errors (missed GT, wrong class, poor localisation, background FP) are
counted per image at conf >= 0.25 / IoU >= 0.5 and drawn in examples_box_errors_*.

Usage (CLI, after training)::

    python src/evaluate.py --data datasets/sell-products/data_abs.yaml \
        --weights runs/detect/yolov8n/weights/best.pt runs/detect/yolov8s/weights/best.pt \
        --out results

    # re-draw the CSV-based figures only (no dataset / GPU needed)
    python src/evaluate.py --replot --out results
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from plotstyle import GT_COLOR, PRED_COLOR, TEXT_2, GRID, model_color
from data import image_label, label_path, list_images, load_names, read_labels

DECISION_CONF = 0.25  # threshold for "the model detected something"
MATCH_IOU = 0.5       # IoU for a box to count as found
NO_DET = -1           # image-level "no detection" label
K_HARDEST = 10        # hardest classes shown in figure + README table


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


def _iou(a, b):
    """IoU of two xyxy boxes (any consistent unit)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _gt_xyxy(g):
    c, cx, cy, w, h = g
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


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


def train_info(weights: str | Path, run_dir: str | Path | None = None) -> dict:
    """What actually ran: the optimizer / lr Ultralytics resolved from 'auto' (stored in
    best.pt), epochs run, best epoch and wall-clock training time (results.csv)."""
    out = {}
    try:
        import torch

        ck = torch.load(str(weights), map_location="cpu", weights_only=False)
        ta = ck.get("train_args", {}) or {}
        out.update({"optimizer": ta.get("optimizer"), "lr0": ta.get("lr0"),
                    "weight_decay": ta.get("weight_decay"),
                    "warmup_epochs": ta.get("warmup_epochs"), "amp": ta.get("amp"),
                    "ultralytics_version_trained": ck.get("version")})
    except Exception as e:  # pragma: no cover - informative only
        out["train_args_error"] = str(e)
    f = Path(run_dir) / "results.csv" if run_dir else None
    if f and f.exists():
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        out["epochs_run"] = int(len(df))
        if "metrics/mAP50-95(B)" in df:
            out["best_epoch_map50_95"] = int(df["metrics/mAP50-95(B)"].idxmax()) + 1
        if "time" in df:
            out["train_time_min"] = round(float(df["time"].iloc[-1]) / 60, 1)
    return out


def run_info() -> dict:
    """Hardware / software provenance written next to the metrics."""
    info = {"python": platform.python_version(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch
        import ultralytics

        info.update({"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                     "cuda": torch.version.cuda,
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                            text=True, check=True).stdout.strip()
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# 1) detection-level metrics
# --------------------------------------------------------------------------- #
def detection_metrics(weights, data_yaml, names, split="test", imgsz=640,
                      project="runs/detect", name="eval") -> dict:
    """Ultralytics val on the held-out split.

    * mAP / P / R: conf=0.001 (standard). P/R are at IoU 0.5 and at the confidence
      that maximises mean F1 (Ultralytics default) - that confidence is returned too.
    * confusion matrix: a second pass at conf=0.25 (the deployment threshold)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    with plt.rc_context({"axes.grid": False}):  # our global grid would cut through CM cells
        m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=16,
                      conf=0.001, iou=0.7, plots=True, project=str(project), name=name,
                      exist_ok=True, verbose=False)
        m_cm = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=16,
                         conf=DECISION_CONF, iou=0.7, plots=True, project=str(project),
                         name=f"{name}_conf{DECISION_CONF}", exist_ok=True, verbose=False)
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
    max_f1_conf = None
    try:  # same smoothing Ultralytics uses to pick the operating point
        from ultralytics.utils.metrics import smooth

        max_f1_conf = round(float(box.px[smooth(box.f1_curve.mean(0), 0.1).argmax()]), 3)
    except Exception:
        pass
    cm = None
    try:
        labels = list(names) + ["background"]
        cm = pd.DataFrame(m_cm.confusion_matrix.matrix.astype(int), index=labels, columns=labels)
        cm.index.name = "pred \\ true"
    except Exception:
        pass
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(_f1(p, r), 4),
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "max_f1_conf": max_f1_conf,
        "val_speed_ms": {k: round(float(v), 2) for k, v in m.speed.items()},
        "save_dir": str(m.save_dir),
        "cm_save_dir": str(m_cm.save_dir),
        "per_class": per_class,
        "confusion": cm,
    }


def box_confusions(cm: pd.DataFrame, k=10) -> pd.DataFrame:
    """Top-k off-diagonal (true -> predicted) box pairs from the conf=0.25 matrix,
    plus missed boxes (true -> background) and background FPs, as counts."""
    rows = []
    for pred in cm.index:
        for true in cm.columns:
            n = int(cm.loc[pred, true])
            if n and pred != true:
                rows.append({"true": true, "pred": pred, "count": n})
    df = pd.DataFrame(rows, columns=["true", "pred", "count"])
    return df.sort_values("count", ascending=False).head(k).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2) image-level predictions
# --------------------------------------------------------------------------- #
def collect_predictions(weights, img_dir, n_classes, imgsz=640, warmup=10,
                        max_images=None):
    """Run the model image-by-image (batch=1, like a real kiosk/camera) and keep
    everything needed for image-level metrics, ROC, box-level errors and example grids.
    ``image`` is stored relative to the dataset root; ``_abs`` is in-memory only."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_dir = Path(img_dir)
    root = img_dir.parent.parent
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
        order = np.argsort(-conf)
        keep = order[conf[order] >= DECISION_CONF][:20]
        low = order[:3] if not len(keep) else []  # best sub-threshold boxes, for diagnosis
        top = int(cls[keep[0]]) if len(keep) else NO_DET
        records.append({
            "image": p.relative_to(root).as_posix(),
            "_abs": str(p),
            "y_true": image_label(gt),
            "y_pred": top,
            "pred_conf": float(conf[keep[0]]) if len(keep) else 0.0,
            "n_gt": len(gt),
            "n_gt_classes": len({g[0] for g in gt}),
            "gt_boxes": [list(map(float, g)) for g in gt],
            "pred_boxes": [[int(cls[k]), float(conf[k]), *map(float, xyxyn[k])] for k in keep],
            "low_boxes": [[int(cls[k]), float(conf[k]), *map(float, xyxyn[k])] for k in low],
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
    per_class = pd.DataFrame({"class": [names[c] for c in present], "n_images": sup,
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
        "score_zero_frac": round(float((scores[:, present] == 0).mean()), 4),
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
# 3) box-level errors at the deployment threshold
# --------------------------------------------------------------------------- #
def box_errors(rec, iou_thr=MATCH_IOU) -> dict:
    """Greedy (by confidence) matching of predictions (conf >= 0.25) to GT boxes.
    Each prediction becomes TP, WRONG_CLASS (IoU >= thr with another class),
    POOR_LOC (same class, 0.1 <= IoU < thr) or BG (no overlap); unmatched GT = MISSED."""
    gts = [_gt_xyxy(g) for g in rec["gt_boxes"]]
    gcls = [int(g[0]) for g in rec["gt_boxes"]]
    used = [False] * len(gts)
    tags = []
    for c, cf, *xy in rec["pred_boxes"]:
        ious = [(_iou(xy, g), j) for j, g in enumerate(gts) if not used[j]]
        best, j = max(ious) if ious else (0.0, -1)
        same = [(_iou(xy, gts[k]), k) for k in range(len(gts)) if gcls[k] == c and not used[k]]
        best_same, js = max(same) if same else (0.0, -1)
        if best_same >= iou_thr:
            used[js] = True
            tags.append(("TP", best_same))
        elif best >= iou_thr:
            tags.append(("WRONG_CLASS", best))
        elif best_same >= 0.1:
            tags.append(("POOR_LOC", best_same))
        else:
            tags.append(("BG", best))
    cnt = Counter(t for t, _ in tags)
    return {"tp": cnt["TP"], "missed": used.count(False), "wrong_class": cnt["WRONG_CLASS"],
            "poor_loc": cnt["POOR_LOC"], "bg_fp": cnt["BG"], "pred_tags": tags,
            "gt_found": used}


def box_error_table(records) -> pd.DataFrame:
    rows = []
    for r in records:
        e = box_errors(r)
        r["_err"] = e
        rows.append({"image": r["image"], "y_true": r["y_true"], "n_gt": r["n_gt"],
                     **{k: e[k] for k in ("tp", "missed", "wrong_class", "poor_loc", "bg_fp")}})
    return pd.DataFrame(rows)


def box_error_summary(df: pd.DataFrame) -> dict:
    single, multi = df[df.n_gt == 1], df[df.n_gt > 1]
    rec = lambda d: round(float(d.tp.sum() / max(d.n_gt.sum(), 1)), 4)  # noqa: E731
    return {"gt_boxes": int(df.n_gt.sum()), "tp": int(df.tp.sum()), "missed": int(df.missed.sum()),
            "wrong_class": int(df.wrong_class.sum()), "poor_loc": int(df.poor_loc.sum()),
            "bg_fp": int(df.bg_fp.sum()),
            "box_recall_single_item_imgs": rec(single), "box_recall_multi_item_imgs": rec(multi),
            "n_multi_item_imgs": int(len(multi))}


# --------------------------------------------------------------------------- #
# plots that need the images
# --------------------------------------------------------------------------- #
def _draw_record(ax, rec, names, all_preds=True, show_tags=False, max_preds=6):
    from PIL import Image

    im = Image.open(rec["_abs"]).convert("RGB")
    W, H = im.size
    ax.imshow(im)
    found = rec.get("_err", {}).get("gt_found", [True] * len(rec["gt_boxes"]))
    for (c, cx, cy, w, h), ok in zip(rec["gt_boxes"], found):
        ax.add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False,
                                   lw=2.5, ec=GT_COLOR, ls="-" if ok or not show_tags else ":"))
        if show_tags and not ok:
            ax.text((cx - w / 2) * W, (cy + h / 2) * H, f"MISSED {names[int(c)]}", fontsize=7,
                    color="white", va="top", bbox=dict(fc=GT_COLOR, ec="none", pad=1))
    boxes = rec["pred_boxes"][:max_preds] if all_preds else rec["pred_boxes"][:1]
    tags = rec.get("_err", {}).get("pred_tags", [None] * len(rec["pred_boxes"]))
    for (c, cf, x0, y0, x1, y1), tag in zip(boxes, tags):
        ax.add_patch(plt.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                                   fill=False, lw=2, ec=PRED_COLOR, ls="--"))
        extra = f" {SHORT[tag[0]]} {tag[1]:.2f}" if show_tags and tag else ""
        ax.text(x0 * W, y0 * H, f"{names[int(c)]} {cf:.2f}{extra}", fontsize=7, color="white",
                va="bottom", bbox=dict(fc=PRED_COLOR, ec="none", pad=1))
    if not rec["pred_boxes"]:  # no detection: show what the model almost said
        for c, cf, x0, y0, x1, y1 in rec.get("low_boxes", [])[:1]:
            ax.add_patch(plt.Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                                       fill=False, lw=1.5, ec=TEXT_2, ls=":"))
            ax.text(x0 * W, y0 * H, f"best below 0.25: {names[int(c)]} {cf:.3f}", fontsize=7,
                    color="white", va="bottom", bbox=dict(fc=TEXT_2, ec="none", pad=1))
    ax.axis("off")


SHORT = {"TP": "TP", "WRONG_CLASS": "WRONG", "POOR_LOC": "LOC", "BG": "BG"}


def _grid(n):
    cols = min(4, max(n, 1))
    rows = int(np.ceil(max(n, 1) / cols))
    return rows, cols


def plot_examples(records, names, correct: bool, out_path, n=8, seed=0, title=""):
    """Grid of up to n images (sized to how many exist). Green = ground truth,
    red dashed = predictions with conf >= 0.25. Saved as JPEG (photo grid)."""
    pool = [r for r in records if r["y_true"] >= 0 and (r["y_true"] == r["y_pred"]) == correct]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))] if pool else []
    rows, cols = _grid(len(pick))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 5.0 * rows), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rec in zip(axes.ravel(), pick):
        _draw_record(ax, rec, names, all_preds=not correct)
        pred = names[rec["y_pred"]] if rec["y_pred"] >= 0 else "(no detection)"
        ok = rec["y_true"] == rec["y_pred"]
        ax.set_title(f"GT: {names[rec['y_true']]}\nPred: {pred} ({rec['pred_conf']:.2f})",
                     fontsize=9, color=GT_COLOR if ok else PRED_COLOR)
    if not pick:
        axes[0, 0].text(0.5, 0.5, "none", ha="center", transform=axes[0, 0].transAxes)
    fig.suptitle(f"{title} ({len(pick)} of {len(pool)})   [green = ground truth, red dashed = prediction]",
                 fontsize=11)
    fig.savefig(out_path, pil_kwargs={"quality": 85})
    plt.close(fig)


def plot_box_errors(records, names, out_path, n=8, title=""):
    """Images with the most box-level errors at conf >= 0.25 / IoU >= 0.5, every GT and
    every prediction drawn and tagged (TP, WRONG_CLASS, POOR_LOC, BG, MISSED)."""
    errs = [r for r in records if r.get("_err") and r["y_true"] >= 0 and
            (r["_err"]["missed"] + r["_err"]["wrong_class"] + r["_err"]["poor_loc"] + r["_err"]["bg_fp"])]
    errs.sort(key=lambda r: -(r["_err"]["missed"] + r["_err"]["wrong_class"]
                              + r["_err"]["poor_loc"] + r["_err"]["bg_fp"]))
    pick = errs[:n]
    rows, cols = _grid(len(pick))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 5.0 * rows), squeeze=False,
                             layout="constrained")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rec in zip(axes.ravel(), pick):
        _draw_record(ax, rec, names, all_preds=True, show_tags=True)
        e = rec["_err"]
        more = len(rec["pred_boxes"]) - 6
        ax.set_title(f"GT: {names[rec['y_true']]} ({rec['n_gt']} box)\n"
                     f"TP {e['tp']} · miss {e['missed']} · wrong {e['wrong_class']} · "
                     f"loc {e['poor_loc']} · bg {e['bg_fp']}" + (f" (+{more} not drawn)" if more > 0 else ""),
                     fontsize=8.5, color=PRED_COLOR)
    if not pick:
        axes[0, 0].text(0.5, 0.5, "no box-level errors", ha="center", transform=axes[0, 0].transAxes)
    fig.suptitle(f"{title} ({len(pick)} of {len(errs)} images with errors)\n"
                 "IoU after each label; TP / WRONG = other class / LOC = IoU < 0.5 / BG = no GT; "
                 "dotted green = missed GT", fontsize=10)
    fig.savefig(out_path, pil_kwargs={"quality": 85})
    plt.close(fig)


# --------------------------------------------------------------------------- #
# plots from tables only (re-drawable offline with --replot)
# --------------------------------------------------------------------------- #
DOT_METRICS = [("precision", "Box precision"), ("recall", "Box recall"), ("f1", "Box F1"),
               ("map50", "mAP@0.5"), ("map50_95", "mAP@0.5:0.95"),
               ("accuracy", "Image accuracy"), ("auc_macro_ovr", "Image AUC (OvR)")]


def plot_comparison(table: pd.DataFrame, out_path):
    """Dot plot of the 0-1 metrics on a zoomed axis (no bars, so the zoom does not
    exaggerate gaps) + latency as bars from zero in a separate panel."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
    ax = axes[0]
    y = np.arange(len(DOT_METRICS))[::-1]
    vals = table[[m for m, _ in DOT_METRICS]].astype(float)
    for yy, (m, _) in zip(y, DOT_METRICS):
        ax.plot(vals[m].values, [yy] * len(vals), color=GRID, lw=3, zorder=1)
    for i, (name, row) in enumerate(vals.iterrows()):
        ax.scatter(row.values, y, s=130 if i == 0 else 55, color=model_color(name, i), zorder=3 + i,
                   label=name, edgecolor="white", linewidth=1.5)  # first model larger: ties stay visible
    for yy, (m, _) in zip(y, DOT_METRICS):
        ax.text(1.03, yy, "  ".join(f"{v:.3f}" for v in vals[m].values), va="center",
                fontsize=8, color=TEXT_2, transform=ax.get_yaxis_transform())
    lo = np.floor((float(np.nanmin(vals.values)) - 0.005) * 100) / 100
    ax.set(yticks=y, yticklabels=[lab for _, lab in DOT_METRICS], xlim=(lo, 1.0),
           xlabel=f"score (axis zoomed to {lo:.2f}-1.00)", title="Test-set metrics (higher is better)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=len(vals), frameon=False)
    ax2 = axes[1]
    for i, (name, row) in enumerate(table.iterrows()):
        ax2.bar(i, float(row["infer_ms_per_img"]), 0.6, color=model_color(name, i))
        ax2.text(i, float(row["infer_ms_per_img"]), f"{float(row['infer_ms_per_img']):.1f}",
                 ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax2.set(xticks=range(len(table)), xticklabels=list(table.index),
            ylabel="median ms / image (batch 1, model only)", title="Inference time (lower is better)")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_speed_accuracy(table: pd.DataFrame, out_path):
    """Two-point trade-off: end-to-end latency vs mAP@0.5:0.95."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = table.astype({"e2e_ms_per_img": float, "map50_95": float, "params_m": float})
    for i, (name, row) in enumerate(t.iterrows()):
        ax.scatter(row.e2e_ms_per_img, row.map50_95, s=60 + 25 * row.params_m,
                   color=model_color(name, i), zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}\n{row.e2e_ms_per_img:.1f} ms · {row.map50_95:.3f} · {row.params_m:.1f}M",
                    (row.e2e_ms_per_img, row.map50_95), textcoords="offset points",
                    xytext=(12, 8) if i else (12, -30), fontsize=9, color=TEXT_2)
    if len(t) == 2:
        (_, a), (_, b) = t.iterrows()
        ax.annotate("", xy=(b.e2e_ms_per_img, b.map50_95), xytext=(a.e2e_ms_per_img, a.map50_95),
                    arrowprops=dict(arrowstyle="->", color=TEXT_2, lw=1.2))
    pad_x = (t.e2e_ms_per_img.max() - t.e2e_ms_per_img.min()) * 0.4 + 1
    ax.set(xlim=(0, t.e2e_ms_per_img.max() + pad_x),
           ylim=(t.map50_95.min() - 0.01, min(1.0, t.map50_95.max() + 0.01)),
           xlabel="median end-to-end ms / image (batch 1)", ylabel="mAP@0.5:0.95 (test)",
           title="Accuracy vs latency (marker size = parameters)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc(curves: dict, out_path):
    """curves = {model: (fpr, tpr, auc_micro, auc_macro)} - one panel."""
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, (name, (fpr, tpr, a_mi, a_ma)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=model_color(name, i),
                label=f"{name}  micro-AUC {a_mi:.4f} | macro-AUC {a_ma:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_2, label="random")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title="Image-level ROC (one-vs-rest, micro-average)")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusions(conf_tables: dict, out_path, title="top confused pairs (true -> predicted)",
                    min_rows=3):
    """Small multiples of top confused pairs per model. Skipped (returns False) when no
    table has at least min_rows rows - a one-bar chart says less than one sentence."""
    if not any(len(df) >= min_rows for df in conf_tables.values()):
        return False
    n = len(conf_tables)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 4.6), squeeze=False)
    for i, (ax, (name, df)) in enumerate(zip(axes[0], conf_tables.items())):
        df = df.iloc[::-1]
        labels = [f"{t}  ->  {p}" for t, p in zip(df["true"], df["pred"])]
        ax.barh(labels, df["count"], color=model_color(name, i), height=0.7)
        for yy, c in enumerate(df["count"]):
            ax.text(c, yy, f" {c}", va="center", fontsize=8, color=TEXT_2)
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel("count")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def plot_worst_classes(per_class: dict, out_path, k=K_HARDEST):
    """Dumbbell of AP@0.5:0.95 for the union of each model's k hardest classes."""
    names = list(per_class)
    ap = pd.DataFrame({m: per_class[m].set_index("class")["ap50_95"] for m in names})
    cls = sorted(set().union(*[set(ap[m].nsmallest(k).index) for m in names]),
                 key=lambda c: ap.loc[c].min())
    nimg = per_class[names[-1]].set_index("class").get("n_images")
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(cls) + 1.6))
    y = np.arange(len(cls))
    for yy, c in zip(y, cls):
        ax.plot(ap.loc[c].values, [yy] * len(names), color=GRID, lw=3, zorder=1)
    for i, m in enumerate(names):
        ax.scatter(ap.loc[cls, m].values, y, s=120 if i == 0 else 50, color=model_color(m, i),
                   zorder=3 + i, edgecolor="white", linewidth=1.5, label=m)
    labels = [f"{c} (n={int(nimg[c])})" if nimg is not None and c in nimg else c for c in cls]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    lo = max(0.0, np.floor(float(ap.loc[cls].min().min()) * 10) / 10)
    ax.set(xlim=(lo, 1.0), xlabel=f"AP@0.5:0.95 on test (axis from {lo:.1f}; n = test images)")
    ax.set_title(f"Hardest classes: union of each model's {k} lowest", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(names), frameon=False)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_per_class_scatter(per_class: dict, out_path, label_top=5):
    """Paired per-class AP@0.5:0.95: first model (x) vs second model (y), y = x line."""
    (a, pa), (b, pb) = list(per_class.items())[:2]
    d = pd.DataFrame({a: pa.set_index("class")["ap50_95"], b: pb.set_index("class")["ap50_95"]})
    diff = d[b] - d[a]
    lo = max(0.0, np.floor(float(d.min().min()) * 20) / 20)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([lo, 1], [lo, 1], color=TEXT_2, lw=1, ls="--", zorder=1)
    for mask, col, lab in [(diff > 0, model_color(b, 1), f"{b} better ({int((diff > 0).sum())})"),
                           (diff < 0, model_color(a, 0), f"{a} better ({int((diff < 0).sum())})"),
                           (diff == 0, GRID, f"equal ({int((diff == 0).sum())})")]:
        ax.scatter(d[a][mask], d[b][mask], s=28, color=col, label=lab, zorder=3,
                   edgecolor="white", linewidth=0.8)
    for c in list(diff.nlargest(label_top).index) + list(diff.nsmallest(2).index):
        ax.annotate(c, (d.loc[c, a], d.loc[c, b]), fontsize=7.5, color=TEXT_2,
                    xytext=(4, -10), textcoords="offset points")
    ax.set(xlim=(lo, 1.0), ylim=(lo, 1.0), xlabel=f"{a} AP@0.5:0.95", ylabel=f"{b} AP@0.5:0.95",
           title=f"Per-class AP@0.5:0.95 on test ({len(d)} classes)")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _read_results_csv(d):
    for f in (Path(d) / "results.csv", Path(d)):
        if f.is_file():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            return df
    return None


def plot_training_curves(run_dirs: dict, out_path, close_mosaic=10):
    """Per epoch: val mAP@0.5, val mAP@0.5:0.95 (zoomed), and train vs val box loss.
    run_dirs values may be a run folder or a copied <model>_results.csv."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    n_ep = 0
    for i, (name, d) in enumerate(run_dirs.items()):
        df = _read_results_csv(d)
        if df is None:
            continue
        x = df["epoch"] + (1 if df["epoch"].min() == 0 else 0)
        n_ep = max(n_ep, int(x.max()))
        col = model_color(name, i)
        axes[0].plot(x, df["metrics/mAP50(B)"], color=col, label=name)
        axes[1].plot(x, df["metrics/mAP50-95(B)"], color=col, label=name)
        axes[2].plot(x, df["val/box_loss"], color=col, label=f"{name} val")
        axes[2].plot(x, df["train/box_loss"], color=col, ls="--", label=f"{name} train")
    axes[0].set_title("mAP@0.5 (val)")
    axes[1].set_title("mAP@0.5:0.95 (val" + (", zoomed from epoch 3)" if n_ep >= 3 else ")"))
    axes[2].set_title("box loss: train (dashed) vs val")
    if n_ep >= 3:
        lows = [v for d in run_dirs.values() if (df := _read_results_csv(d)) is not None
                for v in df["metrics/mAP50-95(B)"].iloc[2:]]
        if lows:
            axes[1].set_ylim(min(lows) - 0.005, max(lows) + 0.005)
    for ax in axes:
        ax.set_xlabel("epoch")
        if n_ep > close_mosaic:
            ax.axvline(n_ep - close_mosaic + 0.5, color=TEXT_2, lw=1, ls=":")
    if n_ep > close_mosaic:
        axes[2].text(n_ep - close_mosaic + 0.7, axes[2].get_ylim()[1], "mosaic off", fontsize=8,
                     color=TEXT_2, va="top")
    axes[0].legend()
    axes[2].legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def replot(out_dir="results"):
    """Re-draw every figure that only needs the CSVs already in out_dir."""
    out = Path(out_dir)
    table = pd.read_csv(out / "comparison.csv", index_col="model")
    models = list(table.index)
    per_class = {m: pd.read_csv(out / f"per_class_{m}.csv") for m in models
                 if (out / f"per_class_{m}.csv").exists()}
    for pc in per_class.values():  # older runs called the image count 'support'
        pc.rename(columns={"support": "n_images"}, inplace=True)
    plot_comparison(table, out / "comparison_bar.png")
    plot_speed_accuracy(table, out / "speed_accuracy.png")
    if len(per_class) >= 2:
        plot_worst_classes(per_class, out / "hardest_classes.png")
        plot_per_class_scatter(per_class, out / "per_class_scatter.png")
    tc = {m: out / f"{m}_results.csv" for m in models if (out / f"{m}_results.csv").exists()}
    if tc:
        plot_training_curves(tc, out / "training_curves.png")
    if (out / "eda_class_counts.csv").exists():
        from data import plot_class_distribution

        plot_class_distribution(pd.read_csv(out / "eda_class_counts.csv", index_col="class"),
                                out / "eda_class_distribution.png")
    print(f"re-drew figures in {out}")


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
# Ultralytics >= 8.3.x prefixes curve plots with "Box" (BoxPR_curve.png); older
# versions don't. Copy whichever exists under one stable name.
ULTRA_PLOTS = {"PR_curve.png": ["BoxPR_curve.png", "PR_curve.png"],
               "F1_curve.png": ["BoxF1_curve.png", "F1_curve.png"]}


def _copy_first(src_dir, candidates, dst):
    src = next((Path(src_dir) / c for c in candidates if (Path(src_dir) / c).exists()), None)
    if src:
        shutil.copy(src, dst)


def evaluate_all(weights: dict, data_yaml, out_dir="results", split="test", imgsz=640,
                 project="runs/detect", train_dirs: dict | None = None, max_images=None):
    """weights = {"yolov8n": path/to/best.pt, "yolov8s": ...}. Writes every table and
    figure used in the README into out_dir and returns the comparison table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(data_yaml)
    img_dir = _split_img_dir(data_yaml, split)
    print(f"evaluating on {img_dir}")
    json.dump(run_info(), open(out / "run_info.json", "w"), indent=2)

    rows, per_class, curves, confs, bconfs, preds = {}, {}, {}, {}, {}, {}
    for name, w in weights.items():
        tdir = (train_dirs or {}).get(name)
        print(f"[{name}] detection metrics ...")
        det = detection_metrics(w, data_yaml, names, split, imgsz, project, f"{name}_{split}")
        print(f"[{name}] image-level predictions ...")
        recs, scores = collect_predictions(w, img_dir, len(names), imgsz, max_images=max_images)
        y_t = [r["y_true"] for r in recs]
        y_p = [r["y_pred"] for r in recs]
        img, img_cls = image_level_metrics(y_t, y_p, scores, names)
        berr = box_error_table(recs)
        info = model_info(w, imgsz)
        ms = np.array([r["infer_ms"] for r in recs])
        tot = np.array([r["total_ms"] for r in recs])

        nbox = Counter(int(g[0]) for r in recs for g in r["gt_boxes"])
        pc = det.pop("per_class").merge(img_cls, on="class", how="left")
        pc.insert(1, "n_boxes", [nbox.get(names.index(c), 0) for c in pc["class"]])
        pc.to_csv(out / f"per_class_{name}.csv", index=False)
        per_class[name] = pc
        cf = top_confusions(y_t, y_p, names)
        cf.to_csv(out / f"top_confusions_{name}.csv", index=False)
        confs[name] = cf
        cm = det.pop("confusion")
        if cm is not None:
            cm.to_csv(out / f"confusion_matrix_conf{DECISION_CONF}_{name}.csv")
            bc = box_confusions(cm)
            bc.to_csv(out / f"box_confusions_{name}.csv", index=False)
            bconfs[name] = bc[(bc.true != "background") & (bc.pred != "background")]
        berr.to_csv(out / f"box_errors_{name}.csv", index=False)
        fpr, tpr = roc_points(y_t, scores)
        curves[name] = (fpr, tpr, img["auc_micro"], img["auc_macro_ovr"])
        preds[name] = recs
        np.save(out / f"scores_{name}.npy", scores)
        with open(out / f"predictions_{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")

        rows[name] = {**{k: v for k, v in det.items()
                         if k not in ("save_dir", "cm_save_dir", "val_speed_ms")},
                      **img,
                      "infer_ms_per_img": round(float(np.median(ms)), 2),
                      "infer_ms_p90": round(float(np.percentile(ms, 90)), 2),
                      "e2e_ms_per_img": round(float(np.median(tot)), 2),
                      "fps": round(1000 / float(np.median(tot)), 1),
                      **info}
        extra = {"val_speed_ms": det["val_speed_ms"], "box_errors": box_error_summary(berr),
                 "train": train_info(w, tdir)}
        json.dump({"model": name, **rows[name], **extra},
                  open(out / f"metrics_{name}.json", "w"), indent=2)
        pd.DataFrame([{k: v for k, v in r.items()
                       if k not in ("gt_boxes", "pred_boxes", "low_boxes") and not k.startswith("_")}
                      for r in recs]).to_csv(out / f"predictions_{name}.csv", index=False)

        # copy the Ultralytics plots and training logs referenced in the README
        for dst, candidates in ULTRA_PLOTS.items():
            _copy_first(det["save_dir"], candidates, out / f"{name}_{dst}")
        _copy_first(det["cm_save_dir"], ["confusion_matrix_normalized.png"],
                    out / f"{name}_confusion_matrix_conf{DECISION_CONF}.png")
        if tdir:
            for src, dst in [("results.png", f"{name}_results.png"),
                             ("results.csv", f"{name}_results.csv"), ("args.yaml", f"{name}_args.yaml")]:
                if (Path(tdir) / src).exists():
                    shutil.copy(Path(tdir) / src, out / dst)

        plot_examples(recs, names, True, out / f"examples_correct_{name}.jpg",
                      title=f"{name}: correct predictions")
        plot_examples(recs, names, False, out / f"examples_wrong_{name}.jpg",
                      title=f"{name}: wrong predictions (image level)")
        plot_box_errors(recs, names, out / f"examples_box_errors_{name}.jpg",
                        title=f"{name}: box-level errors at conf >= {DECISION_CONF}, IoU >= {MATCH_IOU}")

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(out / "comparison.csv")
    plot_roc(curves, out / "roc_curve.png")
    if not plot_confusions(confs, out / "top_confusions.png"):
        (out / "top_confusions.png").unlink(missing_ok=True)
    if bconfs and not plot_confusions(bconfs, out / "box_confusions.png",
                                      title=f"box-level class swaps at conf {DECISION_CONF}"):
        (out / "box_confusions.png").unlink(missing_ok=True)
    replot(out)
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
    ap.add_argument("--data")
    ap.add_argument("--weights", nargs="+", help="paths like runs/detect/yolov8n/weights/best.pt")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--replot", action="store_true", help="only re-draw CSV-based figures")
    a = ap.parse_args()
    if a.replot:
        replot(a.out)
        return
    if not (a.data and a.weights):
        ap.error("--data and --weights are required unless --replot is given")
    weights = {Path(w).parent.parent.name: w for w in a.weights}
    train_dirs = {k: Path(w).parent.parent for k, w in weights.items()}
    print(evaluate_all(weights, a.data, a.out, a.split, a.imgsz, train_dirs=train_dirs).T)


if __name__ == "__main__":
    main()
