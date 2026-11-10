# weights/

ไฟล์ weights ที่ prototype (`app/streamlit_app.py`, `src/infer.py`) ใช้:

| ไฟล์ | มาจาก | ขนาด |
|---|---|---|
| `yolov8n.pt` | `MyDrive/retail-shelf-detection/runs/detect/yolov8n/weights/best.pt` | ~6.3 MB |
| `yolov8s.pt` | `MyDrive/retail-shelf-detection/runs/detect/yolov8s/weights/best.pt` | ~21.6 MB |

วิธีได้ไฟล์มา (เลือกทางใดทางหนึ่ง)

1. ดาวน์โหลด `best.pt` ของแต่ละโมเดลจาก Google Drive แล้วเปลี่ยนชื่อตามตารางมาวางในโฟลเดอร์นี้
2. รัน notebook จนจบ: `results.zip` จะมี `weights/yolov8n.pt` และ `weights/yolov8s.pt` มาให้ แตกไฟล์ที่ root ของ repo

`.gitignore` ยกเว้น `weights/*.pt` ไว้ จึง commit สองไฟล์นี้ได้ (จำเป็นถ้าจะ deploy บน Streamlit Community Cloud)
ไฟล์ `*.pt` ที่อื่นใน repo ยังถูก ignore เหมือนเดิม
