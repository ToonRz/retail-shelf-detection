"""Inference helpers for the prototype (app/streamlit_app.py) and a small CLI.

Kept free of Streamlit so it can be tested without a browser:

    python src/infer.py --source shelf.jpg --weights weights/yolov8n.pt
    python src/infer.py --source clip.mp4  --weights weights/yolov8n.pt --out out.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["yolov8n", "yolov8s"]
DECISION_CONF = 0.25  # same threshold as the image-level metrics in evaluate.py
NMS_IOU = 0.7         # Ultralytics default, same as in the evaluation
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DET_COLS = ["class", "conf", "x1", "y1", "x2", "y2"]
# (fourcc, extension) tried in order. VP9 and H.264 play in browsers; the pip OpenCV
# wheels ship VP9 but usually no H.264 encoder. mp4v is the fallback that every build
# can write (plays in VLC, not in every browser).
VIDEO_CODECS = [("vp09", ".webm"), ("avc1", ".mp4"), ("mp4v", ".mp4")]


def find_weights(model: str, root: str | Path = ROOT) -> Path | None:
    """weights/<model>.pt first (committed for the demo), then the training run."""
    root = Path(root)
    for p in (root / "weights" / f"{model}.pt",
              root / "runs" / "detect" / model / "weights" / "best.pt"):
        if p.is_file():
            return p
    return None


def load_model(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def _detections(result):
    import pandas as pd

    b = result.boxes
    if b is None or len(b) == 0:
        return pd.DataFrame(columns=DET_COLS)
    xyxy = b.xyxy.cpu().numpy().round(1)
    df = pd.DataFrame(xyxy, columns=["x1", "y1", "x2", "y2"])
    df.insert(0, "conf", b.conf.cpu().numpy().round(3))
    df.insert(0, "class", [result.names[int(c)] for c in b.cls.cpu().numpy()])
    return df.sort_values("conf", ascending=False).reset_index(drop=True)


def predict_image(model, image, conf=DECISION_CONF, iou=NMS_IOU, imgsz=640):
    """image: RGB array or PIL image. Returns (annotated RGB array, detections, ms).
    ms = preprocess + inference + NMS as reported by Ultralytics."""
    import numpy as np

    rgb = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics treats arrays as BGR
    r = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return r.plot()[..., ::-1], _detections(r), float(sum(r.speed.values()))


def summarize(dets):
    """Detections -> one row per class: count, best and mean confidence."""
    import pandas as pd

    if dets.empty:
        return pd.DataFrame(columns=["class", "count", "max_conf", "mean_conf"])
    g = dets.groupby("class")["conf"].agg(count="size", max_conf="max", mean_conf="mean")
    return g.round(3).sort_values(["count", "max_conf"], ascending=False).reset_index()


def _open_writer(dst: Path, fps: float, size: tuple[int, int]):
    import cv2

    for fourcc, ext in VIDEO_CODECS:
        out = dst.with_suffix(ext)
        w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w, out, fourcc
        w.release()
        out.unlink(missing_ok=True)
    raise RuntimeError("OpenCV could not open a video writer (tried vp09, avc1, mp4v)")


def annotate_video(model, src, dst, conf=DECISION_CONF, iou=NMS_IOU, imgsz=640, stride=1,
                   max_frames=None, progress=None) -> dict:
    """Run the detector on every `stride`-th frame and write those frames with boxes.
    The output keeps the original duration (fps / stride). Returns the output path
    (extension depends on the codec that was available) and per-class statistics:
    frames = processed frames in which the class was detected, max_in_frame = most
    boxes of that class in a single frame, max_conf = best confidence."""
    import cv2
    import pandas as pd

    stride = max(1, int(stride))
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot read video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer, out_path, fourcc = _open_writer(Path(dst), fps / stride, size)

    stats: dict[str, dict] = {}
    i = done = 0
    ms = 0.0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and done >= max_frames):
                break
            if i % stride == 0:
                r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
                writer.write(r.plot())
                ms += sum(r.speed.values())
                for row in summarize(_detections(r)).itertuples():
                    s = stats.setdefault(row[1], {"frames": 0, "max_in_frame": 0, "max_conf": 0.0})
                    s["frames"] += 1
                    s["max_in_frame"] = max(s["max_in_frame"], int(row.count))
                    s["max_conf"] = max(s["max_conf"], float(row.max_conf))
                done += 1
                if progress and total:
                    progress(min(1.0, (i + 1) / total))
            i += 1
    finally:
        cap.release()
        writer.release()

    table = pd.DataFrame([{"class": k, **v} for k, v in stats.items()],
                         columns=["class", "frames", "max_in_frame", "max_conf"])
    table = table.sort_values(["frames", "max_conf"], ascending=False).reset_index(drop=True)
    return {"path": out_path, "codec": fourcc, "frames_processed": done, "frames_total": total,
            "fps_in": round(fps, 2), "ms_per_frame": round(ms / max(done, 1), 1),
            "seconds": round(time.perf_counter() - t0, 1), "classes": table}


def main():
    ap = argparse.ArgumentParser(description="Run a trained detector on an image or a video")
    ap.add_argument("--source", required=True, help="image or video file")
    ap.add_argument("--weights", help="path to .pt (default: weights/<model>.pt)")
    ap.add_argument("--model", default="yolov8n", choices=MODELS)
    ap.add_argument("--conf", type=float, default=DECISION_CONF)
    ap.add_argument("--stride", type=int, default=1, help="video: use every n-th frame")
    ap.add_argument("--out", help="output file (default: <source>_pred.<ext>)")
    a = ap.parse_args()

    weights = a.weights or find_weights(a.model)
    if not weights:
        raise SystemExit(f"no weights for {a.model}: put best.pt at weights/{a.model}.pt or pass --weights")
    model = load_model(weights)
    src = Path(a.source)
    out = Path(a.out) if a.out else src.with_name(src.stem + "_pred" + src.suffix)

    if src.suffix.lower() in IMG_EXTS:
        from PIL import Image

        img, dets, ms = predict_image(model, Image.open(src), a.conf)
        Image.fromarray(img).save(out)
        print(summarize(dets).to_string(index=False) if len(dets) else "no detection")
        print(f"{ms:.1f} ms -> {out}")
    else:
        res = annotate_video(model, src, out, a.conf, stride=a.stride)
        print(res["classes"].to_string(index=False) if len(res["classes"]) else "no detection")
        print(f"{res['frames_processed']} frames, {res['ms_per_frame']} ms/frame -> {res['path']}")


if __name__ == "__main__":
    main()
