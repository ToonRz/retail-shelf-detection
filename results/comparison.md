| Model | Box P | Box R | Box F1 | mAP50 | mAP50-95 | Img Acc | Img AUC | ms/img (model) | ms/img (e2e) | Params (M) | GFLOPs | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8n | 0.973 | 0.988 | 0.980 | 0.983 | 0.969 | 0.998 | 0.999 | **7.4** | **9.7** | **3.16** | **8.9** | **6.3** |
| yolov8s | **0.984** | 0.987 | **0.986** | 0.984 | **0.977** | 0.998 | 1.000 | 14.6 | 16.6 | 11.18 | 28.9 | 21.6 |

_Box P/R/F1: IoU 0.5 ที่ confidence ซึ่งให้ mean F1 สูงสุดของแต่ละโมเดล (yolov8n 0.816, yolov8s 0.794) · mAP: conf 0.001, ไม่ขึ้นกับ threshold · Img Acc/AUC: ระดับรูป, top-1 ที่ conf ≥ 0.25 · ms/img (model) = inference อย่างเดียว, (e2e) = รวม pre/post-processing, median ที่ batch 1 · ตัวหนา = ดีกว่าอย่างน้อย 0.5 pp (หรือดีกว่าสำหรับคอลัมน์ความเร็ว/ขนาด)_
