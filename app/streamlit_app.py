"""Prototype: detect and classify retail products in images, videos or a camera shot.

    pip install -r requirements-app.txt
    streamlit run app/streamlit_app.py

Weights are looked up in weights/<model>.pt, then runs/detect/<model>/weights/best.pt;
otherwise a .pt file can be uploaded in the sidebar.
"""
import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer import (DECISION_CONF, MODELS, NMS_IOU, annotate_video, find_weights,  # noqa: E402
                   load_model, predict_image, summarize)

TMP = Path(tempfile.gettempdir()) / "retail-shelf-detection"
TMP.mkdir(exist_ok=True)
MODEL_NOTES = {"yolov8n": "เล็กและเร็ว (3.2M parameters)", "yolov8s": "กรอบแม่นกว่า แต่ช้ากว่า ~2 เท่า (11.2M)"}
MIME = {".mp4": "video/mp4", ".webm": "video/webm"}

st.set_page_config(page_title="Retail product detection", page_icon="🛒", layout="wide")


@st.cache_resource(show_spinner="กำลังโหลดโมเดล…")
def get_model(path: str, _mtime: float):
    return load_model(path)


def save_upload(upload, suffix: str) -> Path:
    """Write an uploaded file to disk once (name = content hash)."""
    data = upload.getvalue()
    p = TMP / (hashlib.sha1(data).hexdigest()[:16] + suffix)
    if not p.exists():
        p.write_bytes(data)
    return p


def show_result(image, model, conf, iou):
    annotated, dets, ms = predict_image(model, image, conf, iou)
    left, right = st.columns([3, 2])
    left.image(annotated, width="stretch")
    with right:
        if dets.empty:
            st.warning(f"ไม่พบสินค้าที่ confidence ≥ {conf:.2f} (ลองลด threshold ทางซ้าย)")
        else:
            top = dets.iloc[0]
            st.metric("สินค้าที่มั่นใจที่สุด (top-1)", top["class"])
            st.caption(f"confidence {top['conf']:.2f} · พบ {len(dets)} กล่อง · {dets['class'].nunique()} ชนิด")
            st.dataframe(summarize(dets).rename(columns={
                "class": "สินค้า", "count": "จำนวน", "max_conf": "conf สูงสุด", "mean_conf": "conf เฉลี่ย"}),
                hide_index=True, width="stretch")
            with st.expander("รายละเอียดทุกกล่อง (พิกัดเป็นพิกเซล)"):
                st.dataframe(dets, hide_index=True, width="stretch")
        st.caption(f"เวลา: {ms:.0f} ms ต่อรูป (preprocess + inference + NMS บนเครื่องนี้)")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("ตั้งค่า")
    name = st.radio("โมเดล", MODELS, format_func=lambda m: f"{m} – {MODEL_NOTES[m]}")
    conf = st.slider("Confidence threshold", 0.05, 0.95, DECISION_CONF, 0.05,
                     help="แสดงเฉพาะกล่องที่ confidence ไม่ต่ำกว่าค่านี้ (รายงานใช้ 0.25)")
    iou = st.slider("NMS IoU", 0.30, 0.90, NMS_IOU, 0.05,
                    help="กล่องที่ทับกันเกินค่านี้จะถูกรวมเป็นกล่องเดียว")
    weights = find_weights(name, ROOT)
    custom = st.file_uploader("ใช้ไฟล์ weights อื่น (.pt)", type=["pt"],
                              help="เช่น best.pt จาก runs/detect/<model>/weights/ ใน Google Drive")
    if custom is not None:
        weights = save_upload(custom, ".pt")

st.title("🛒 Retail product detection")
st.caption("ตรวจจับและจำแนกสินค้าร้านค้าปลีก 110 คลาส (dataset Sell Products – O2O Minimart) ด้วย YOLOv8")

if weights is None:
    st.error(f"ยังไม่มีไฟล์ weights ของ **{name}**")
    st.markdown(
        f"เลือกทางใดทางหนึ่ง\n"
        f"1. copy `best.pt` จาก `MyDrive/retail-shelf-detection/runs/detect/{name}/weights/` "
        f"มาไว้ที่ `weights/{name}.pt` แล้วกด Rerun\n"
        f"2. อัปโหลดไฟล์ `.pt` ในแถบด้านซ้าย")
    st.stop()

try:
    model = get_model(str(weights), weights.stat().st_mtime)
except Exception as e:  # corrupt / non-YOLO file
    st.error(f"โหลด weights ไม่ได้: `{weights.name}` ({type(e).__name__}: {e})")
    st.stop()

with st.sidebar:
    st.success(f"weights: `{custom.name if custom else weights.relative_to(ROOT)}`\n\n"
               f"{len(model.names)} คลาส")
    with st.expander("รายชื่อคลาสที่โมเดลรู้จัก"):
        st.write(", ".join(model.names.values()))

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ รูปภาพ", "🎞️ วิดีโอ", "📷 กล้อง"])

with tab_img:
    files = st.file_uploader("อัปโหลดรูปสินค้า (เลือกได้หลายรูป)", accept_multiple_files=True,
                             type=["jpg", "jpeg", "png", "bmp", "webp"])
    for f in files or []:
        st.subheader(f.name)
        try:
            show_result(Image.open(f), model, conf, iou)
        except Exception as e:
            st.error(f"อ่านรูปไม่ได้ ({type(e).__name__}: {e})")

with tab_vid:
    video = st.file_uploader("อัปโหลดวิดีโอ", type=["mp4", "mov", "avi", "mkv", "webm"])
    stride = st.select_slider("ประมวลผลทุก ๆ กี่เฟรม", [1, 2, 3, 5, 10], 2,
                              help="ค่ามาก = เร็วขึ้น แต่วิดีโอผลลัพธ์กระตุกขึ้น")
    if video is not None and st.button("เริ่มตรวจจับ", type="primary"):
        src = save_upload(video, Path(video.name).suffix.lower() or ".mp4")
        bar = st.progress(0.0, "กำลังประมวลผล…")
        try:
            res = annotate_video(model, src, TMP / f"{src.stem}_{name}_pred", conf, iou,
                                 stride=stride, progress=lambda p: bar.progress(p, "กำลังประมวลผล…"))
        except Exception as e:
            bar.empty()
            st.error(f"ประมวลผลวิดีโอไม่ได้ ({type(e).__name__}: {e})")
        else:
            bar.empty()
            st.session_state["video_result"] = res

    res = st.session_state.get("video_result")
    if res and Path(res["path"]).exists():
        out = Path(res["path"])
        left, right = st.columns([3, 2])
        with left:
            if res["codec"] == "mp4v":
                st.info("เบราว์เซอร์บางตัวเล่นไฟล์นี้ไม่ได้ (codec mp4v) ให้ดาวน์โหลดไปเปิดแทน")
            st.video(out.read_bytes(), format=MIME.get(out.suffix, "video/mp4"))
            st.download_button("ดาวน์โหลดวิดีโอผลลัพธ์", out.read_bytes(), "prediction" + out.suffix,
                               MIME.get(out.suffix, "video/mp4"))
        with right:
            st.caption(f"ประมวลผล {res['frames_processed']} จาก {res['frames_total']} เฟรม · "
                       f"{res['ms_per_frame']} ms ต่อเฟรม · รวม {res['seconds']} วินาที")
            if res["classes"].empty:
                st.warning("ไม่พบสินค้าในวิดีโอนี้")
            else:
                st.dataframe(res["classes"].rename(columns={
                    "class": "สินค้า", "frames": "จำนวนเฟรมที่พบ",
                    "max_in_frame": "มากสุดในเฟรมเดียว", "max_conf": "conf สูงสุด"}),
                    hide_index=True, width="stretch")
                st.caption("ไม่มีการ track วัตถุข้ามเฟรม จำนวนชิ้นจึงนับเป็นค่ามากสุดที่เห็นในเฟรมเดียว")

with tab_cam:
    shot = st.camera_input("ถ่ายภาพสินค้า")
    if shot is not None:
        show_result(Image.open(shot), model, conf, iou)
