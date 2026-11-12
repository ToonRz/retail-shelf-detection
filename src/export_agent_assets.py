"""Export a trained detector for retail-shelf-agent (in-browser inference, see
docs/agent-connection/SPEC.md).

    python src/export_agent_assets.py                    # weights/yolov8n.pt, else the weights-v1 release
    python src/export_agent_assets.py --weights runs/detect/yolov8n/weights/best.pt \
        --images datasets/sell-products/test/images --n 16

Writes export/<model>/:
  model.onnx    static input images[1,3,640,640], output0[1, 4+classes, 8400], no NMS inside
  labels.json   class names in index order
  model.json    name, dataset, classes, imgsz, opset, sha256 (read by the agent's package-release.mjs)
  fixtures/     only with --images: n images (+ one non-square crop) and expected.json =
                Ultralytics' predictions with the SAME onnx (conf 0.25, iou 0.7), the
                reference for the agent's parity tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer import DECISION_CONF, IMG_EXTS, NMS_IOU, ROOT, find_weights  # noqa: E402

IMGSZ = 640


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_onnx(weights: Path, out: Path, dataset: str) -> dict:
    import onnx
    from ultralytics import YOLO

    model = YOLO(str(weights))
    onnx_path = Path(model.export(format="onnx", imgsz=IMGSZ, dynamic=False, simplify=True, nms=False))
    out.mkdir(parents=True, exist_ok=True)
    dst = out / "model.onnx"
    shutil.move(str(onnx_path), dst)

    names = [model.names[i] for i in range(len(model.names))]
    (out / "labels.json").write_text(json.dumps(names, ensure_ascii=False, indent=1) + "\n")

    g = onnx.load(str(dst)).graph
    dims = lambda v: [d.dim_value for d in v.type.tensor_type.shape.dim]  # noqa: E731
    info = {
        "name": Path(weights).stem if Path(weights).stem != "best" else Path(weights).parents[1].name,
        "dataset": dataset,
        "classes": len(names),
        "imgsz": IMGSZ,
        "opset": onnx.load(str(dst)).opset_import[0].version,
        "input": {"name": g.input[0].name, "shape": dims(g.input[0])},
        "output": {"name": g.output[0].name, "shape": dims(g.output[0])},
        "sha256": sha256(dst),
        "source_weights_sha256": sha256(Path(weights)),
    }
    assert info["output"]["shape"] == [1, 4 + len(names), 8400], info["output"]
    (out / "model.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def make_fixtures(onnx_path: Path, labels: list[str], images: Path, out: Path, n: int) -> int:
    from PIL import Image
    from ultralytics import YOLO

    files = sorted(p for p in images.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"no images in {images}")
    step = max(1, len(files) // n)
    picked = files[::step][:n]
    out.mkdir(parents=True, exist_ok=True)
    for p in picked:
        shutil.copy(p, out / p.name)
    # One non-square image so the letterbox padding path is tested too
    # (the Roboflow images are already 640x640).
    first = Image.open(picked[0]).convert("RGB")
    crop = first.crop((0, 0, int(first.width * 0.75), first.height))
    crop_name = f"crop_{picked[0].stem}.jpg"
    # 4:4:4 so the agent's test decoder (jpeg-js) and libjpeg agree to +-3 levels; with 4:2:0 their
    # chroma upsampling differs by up to ~50 levels, enough to move borderline confidences past 0.02.
    crop.save(out / crop_name, quality=95, subsampling=0)

    model = YOLO(str(onnx_path), task="detect")
    expected = []
    for name in [p.name for p in picked] + [crop_name]:
        r = model.predict(str(out / name), imgsz=IMGSZ, conf=DECISION_CONF, iou=NMS_IOU, verbose=False)[0]
        h, w = r.orig_shape
        b = r.boxes
        boxes = [{"classId": int(c), "cls": labels[int(c)], "conf": round(float(s), 4),
                  "xyxy": [round(float(v), 2) for v in xyxy]}
                 for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        expected.append({"image": name, "width": w, "height": h,
                         "boxes": sorted(boxes, key=lambda x: -x["conf"])})
    (out / "expected.json").write_text(json.dumps(
        {"conf": DECISION_CONF, "iou": NMS_IOU, "imgsz": IMGSZ, "model_sha256": sha256(onnx_path),
         "images": expected}, indent=1) + "\n")
    return len(expected)


def main():
    ap = argparse.ArgumentParser(description="Export model.onnx + labels + parity fixtures for retail-shelf-agent")
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--weights", help="path to .pt (default: weights/<model>.pt or the weights-v1 release)")
    ap.add_argument("--dataset", default="sell-products v3")
    ap.add_argument("--images", type=Path, help="folder of images for parity fixtures (e.g. the test split)")
    ap.add_argument("--n", type=int, default=16, help="number of fixture images")
    ap.add_argument("--out", type=Path, help="default: export/<model>")
    a = ap.parse_args()

    weights = Path(a.weights) if a.weights else find_weights(a.model)
    if not weights:
        raise SystemExit(f"no weights for {a.model}: put best.pt at weights/{a.model}.pt or pass --weights")
    out = a.out or ROOT / "export" / a.model
    info = export_onnx(weights, out, a.dataset)
    print(f"{out}/model.onnx  {info['classes']} classes  opset {info['opset']}  output {info['output']['shape']}")
    if a.images:
        labels = json.loads((out / "labels.json").read_text())
        k = make_fixtures(out / "model.onnx", labels, a.images, out / "fixtures", a.n)
        print(f"{out}/fixtures  {k} images + expected.json")


if __name__ == "__main__":
    main()
