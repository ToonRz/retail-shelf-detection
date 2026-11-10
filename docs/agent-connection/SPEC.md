# Spec: Shelf Agent ↔ Web (รัน model ใน browser)

สถานะ: **ตกลงแล้ว** (จาก [mock/agent-connection.html](../../mock/agent-connection.html)) · งานแบ่งไว้ใน [TASKS.md](TASKS.md)

## 1. เป้าหมาย

เอา YOLOv8n ที่เทรนใน repo นี้ (110 คลาส) ไปให้คนใช้ผ่านเว็บ โดย

- **ฟรี 100%** ไม่มี server: เว็บเป็นไฟล์ static บน GitHub Pages ส่วน model รันในเครื่องผู้ใช้ด้วย onnxruntime-web
- แยก agent ออกเป็น repo ของตัวเอง ออก release เอง แล้วเว็บ **pin เวอร์ชัน** และ **bump** มาใช้ แบบเดียวกับ zcrX ↔ zcrAgentX
- รูปของผู้ใช้ไม่ถูกส่งออกจากเครื่อง

นอกขอบเขต: agent ติดตั้งที่ร้าน (edge), server inference, fleet / enroll / heartbeat, การเซ็น release (Sigstore)

## 2. สามส่วนกับสองลูป

| Repo | หน้าที่ | ผลลัพธ์ |
|---|---|---|
| [retail-shelf-detection](https://github.com/ToonRz/retail-shelf-detection) (repo นี้) | เทรน / ประเมิน / **export** | `model.onnx`, `labels.json`, `model.json` + fixture (ขึ้นเป็น release `model-yolov8n-v3` ของ agent) · release `weights-v1` (best.pt) |
| [retail-shelf-agent](https://github.com/ToonRz/retail-shelf-agent) | library JS: โหลด model, ตรวจ sha256, preprocess → run → NMS | release `vX.Y.Z` (§4) |
| [retail-shelf-web](https://github.com/ToonRz/retail-shelf-web) | หน้าเว็บ static, pin `agent.version` | GitHub Pages |

```
BUILD:   detection --export--> model release --model.lock.json--> agent --tag vX.Y.Z--> agent release
         web: make bump-agent  (ดึง + ตรวจ + แก้ agent.version + test)  --PR--> main --Pages workflow--> site
RUNTIME: browser โหลด index.html + /agent/vX.Y.Z/* จาก Pages แล้วรัน model ในเครื่อง
```

ทุก repo คุยกันผ่าน **ไฟล์ release + contract ใน §4–§5 เท่านั้น** ห้าม import โค้ดข้าม repo

## 3. Model (repo นี้)

`src/export_agent_assets.py` (ใช้ `find_weights` ของ `src/infer.py` จึงเอา `weights/yolov8n.pt` หรือ release `weights-v1` ได้เอง)

```bash
python src/export_agent_assets.py --images datasets/sell-products/test/images --n 16
```

เขียน `export/yolov8n/`:

| ไฟล์ | เนื้อหา |
|---|---|
| `model.onnx` | input `images` float32 `[1,3,640,640]` · output `output0` float32 `[1, 4+C, 8400]` (C = 110) · ไม่มี NMS ในกราฟ · opset 18 · ~12 MB |
| `labels.json` | ชื่อคลาสเรียงตาม index (ยาว C) |
| `model.json` | `name, dataset, classes, imgsz, opset, input, output, sha256, source_weights_sha256` |
| `fixtures/` | n รูปจาก test + 1 รูป crop ที่ไม่เป็นสี่เหลี่ยมจัตุรัส + `expected.json` |

`expected.json` = สิ่งที่ **Ultralytics ทายด้วย `model.onnx` ไฟล์เดียวกันนั้น** (conf 0.25, iou 0.7, imgsz 640) พิกัดเป็น pixel ของรูปต้นฉบับ
ใช้เป็นเกณฑ์ parity ของ agent (§6.4) · export ONNX ซ้ำแล้ว byte ไม่เหมือนเดิม จึงต้องสร้าง fixture คู่กับ onnx ไฟล์ที่ปล่อยจริงเสมอ

ไฟล์ทั้งสามขึ้นเป็น release ของ **retail-shelf-agent** (tag `model-yolov8n-v3`) แล้ว agent อ้างถึงด้วย `model.lock.json`:

```json
{ "source": "https://github.com/ToonRz/retail-shelf-agent/releases/download/model-yolov8n-v3/",
  "files": { "model.onnx": "<sha256>", "labels.json": "<sha256>", "model.json": "<sha256>" } }
```

`npm run fetch-model` ดาวน์โหลดและปฏิเสธถ้า sha256 ไม่ตรง · ตอนนี้ lock ชี้ไปที่ release `model-yolov8n-v3` (110 คลาส) · release `dev-coco-model` (COCO 80 คลาส) ยังเก็บไว้สำหรับ dev

## 4. Agent release (สัญญาของ release)

Tag `vX.Y.Z` บน retail-shelf-agent → GitHub Release ที่มีไฟล์แบบ **flat** (release asset ไม่มีโฟลเดอร์) สร้างด้วย `scripts/package-release.mjs`:

| ไฟล์ | |
|---|---|
| `shelf-agent.js` | ES module, export `load` (§5) |
| `model.onnx`, `labels.json` | จาก model release ตาม `model.lock.json` |
| `ort-wasm-*.wasm`, `ort-wasm-*.mjs` | onnxruntime-web ที่ pin ใน package-lock ไม่พึ่ง CDN (งาน A1 เลือกชุดที่ใช้จริง) |
| `manifest.json` | ด้านล่าง |
| `SHA256SUMS` | ทุกไฟล์รวม `manifest.json` รูปแบบ `sha256sum -c` |

```json
{
  "version": "v0.1.0",
  "contract": "shelf-agent-js/1",
  "entry": "shelf-agent.js",
  "model": { "file": "model.onnx", "labels": "labels.json", "name": "yolov8n", "dataset": "sell-products v3",
             "classes": 110, "imgsz": 640, "conf": 0.25, "iou": 0.7, "max_det": 300, "sha256": "…" },
  "files": { "labels.json": "…", "model.onnx": "…", "shelf-agent.js": "…", "ort-wasm-simd-threaded.wasm": "…" }
}
```

กฎของเวอร์ชัน:
- **patch** (`v0.1.0 → v0.1.1`) = แก้บั๊ก ไม่เปลี่ยน API หรือ model
- **minor** = เปลี่ยน model หรือเพิ่มความสามารถ แต่ API เดิมยังใช้ได้
- **major** หรือเปลี่ยน `contract` = API เปลี่ยน ต้องแก้เว็บพร้อมกัน
- `v0.0.x` = release สำหรับทดสอบ pipeline ด้วย model COCO · `v0.1.0` = release แรกที่ใช้ model จริง

## 5. JS API (contract `shelf-agent-js/1`)

ต้นฉบับอยู่ที่ `retail-shelf-agent/src/types.ts` ส่วนเว็บมีสำเนาที่ `retail-shelf-web/src/agent-contract.ts`
แก้อันใดอันหนึ่งต้องแก้ทั้งสองที่ และถ้าไม่เข้ากันกับของเดิมต้องเปลี่ยน `CONTRACT`

```ts
load({ baseUrl, expectedVersion, backend?: "webgpu" | "wasm", onProgress? }): Promise<Agent>
agent.detect(image: ImageBitmap | ImageData | HTMLImageElement | HTMLCanvasElement,
             { conf?, iou?, maxDet? }): Promise<DetectResult>
agent.backend      // ที่ใช้จริงหลัง fallback
agent.dispose()
```

`load()` ทำตามลำดับ และ throw `AgentError(code)` ทันทีที่ขั้นไหนไม่ผ่าน:

1. fetch `manifest.json` (ไม่ได้ → `fetch_failed`, ไม่ใช่ JSON / field ผิด → `manifest_invalid`)
2. `contract` ไม่ตรง → `contract_mismatch` · `version` ≠ `expectedVersion` → `version_mismatch`
3. ดาวน์โหลด model (รายงาน `onProgress`) แล้ว sha256 ≠ `manifest.model.sha256` → `model_sha256_mismatch`
4. `labels.json` ยาวไม่เท่า `classes` → `labels_mismatch`
5. สร้าง session: ใช้ `webgpu` ถ้าขอและเครื่องรองรับ ไม่งั้นใช้ `wasm` ถ้าใช้ไม่ได้ทั้งคู่ → `backend_unavailable`

`DetectResult = { version, model_sha256, backend, latency_ms, width, height, boxes: [{ cls, classId, conf, xyxy }] }`
`xyxy` เป็น pixel ของรูปต้นฉบับ เรียงตาม `conf` จากมากไปน้อย ค่าตั้งต้น conf / iou / max_det มาจาก manifest

## 6. Inference (ต้องให้ผลเท่ากับ Ultralytics)

### 6.1 Preprocess (เท่ากับ `LetterBox(auto=False, center=True)`)
```
r = min(640 / h, 640 / w)
nw, nh = round(w * r), round(h * r)          # resize bilinear
dw, dh = (640 - nw) / 2, (640 - nh) / 2
left, top = round(dw - 0.1), round(dh - 0.1) # ขอบที่เหลือเติมสี (114,114,114)
tensor = RGB / 255, NCHW float32 [1,3,640,640]
```
`round` ในที่นี้คือ `round` ของ Python ซึ่งปัดครึ่งไปหาเลขคู่ (half-to-even) **ไม่ใช่** `Math.round` ของ JS · ถ้า `nw, nh` เท่ากับขนาดเดิมไม่ต้อง resize
(ตรวจกับ `ultralytics/data/augment.py: LetterBox` และ `utils/nms.py` เวอร์ชัน 8.4.208 แล้ว)
### 6.2 Run
`session.run({ images: tensor })` → `output0` แบบ channel-major: ค่า k ของ anchor i คือ `out[k * 8400 + i]`
โดย k = 0..3 คือ cx, cy, w, h (pixel ของภาพ 640) และ k = 4..4+C-1 คือคะแนนของแต่ละคลาส (ผ่าน sigmoid แล้ว)
### 6.3 Postprocess
1. ต่อ anchor: `classId = argmax` ของคะแนนคลาส, `conf = max`; เก็บไว้เฉพาะ `conf > confThreshold`
2. cxcywh → xyxy
3. NMS แยกตามคลาส: ในคลาสเดียวกัน ถ้า IoU > `iou` ให้ตัดกล่องที่ conf ต่ำกว่าทิ้ง (เท่ากับที่ Ultralytics เลื่อนกล่องตาม class offset)
4. เก็บแค่ `max_det` กล่องแรก
5. ถอด letterbox: `x = (x - left) / r`, `y = (y - top) / r` แล้ว clip ให้อยู่ใน `[0, w] × [0, h]`
### 6.4 เกณฑ์ parity (เทียบกับ `expected.json`)
- ทุกกล่องใน expected ต้องมีกล่องที่ทายคลาสเดียวกันและ **IoU ≥ 0.9**, **|Δconf| ≤ 0.02**
- ห้ามมีกล่องเกินที่ conf ≥ 0.30 และจับคู่กับกล่องไหนไม่ได้
- (ระยะเผื่อมีไว้เพราะตัว decode JPEG และวิธี resize ใน JS ต่างจาก OpenCV เล็กน้อย)

## 7. Pin และ bump (retail-shelf-web)

- `agent.version` บรรทัดเดียว: `mock` หรือ `vX.Y.Z` ค่าอื่นทำให้ build ล้ม · `mock` = ใช้ `src/mockAgent.ts`
- `make fetch-agent-release`: ดาวน์โหลด release ตาม pin ลง `public/agent/<v>/` (ไฟล์ release **ไม่ commit**) แล้วตรวจ
  - `SHA256SUMS` ต้องตรงทุกไฟล์
  - ต้องมีไฟล์ครบตามที่ `manifest.files` บอก
  - `manifest.version` ต้องเท่ากับ pin และ `contract` ต้องเป็นตัวที่เว็บรองรับ
  - ไม่ผ่านข้อใดข้อหนึ่ง → exit ≠ 0 และลบโฟลเดอร์ทิ้ง
- `make bump-agent [AGENT_VERSION=vX.Y.Z] [ALLOW_MINOR=1]`:
  1. ถ้าไม่ระบุเวอร์ชัน ให้เลือก **patch ล่าสุด** ที่อยู่ใน minor เดียวกับ pin ปัจจุบัน (ถ้า pin เป็น `mock` ให้เลือก release ล่าสุด)
  2. ปฏิเสธเวอร์ชันที่ต่ำกว่าเดิม และปฏิเสธการข้าม minor/major ถ้าไม่ใส่ `ALLOW_MINOR=1`
  3. fetch + ตรวจ → เขียน `agent.version` → `npm test` + `npm run build`
  4. ถ้าขั้นไหนพัง ให้คืน `agent.version` เดิม · **ไม่ commit ให้** (คนเปิด PR เอง)
- CI ของเว็บ: เมื่อมี patch ใหม่กว่า pin ให้ขึ้น warning (ไม่ทำให้ CI ล้ม)

## 8. Deploy

GitHub Pages ของ retail-shelf-web (Settings → Pages → Source: GitHub Actions) · workflow ทำงานเมื่อ push `main`:
`npm ci` → `make fetch-agent-release` (ข้ามถ้า pin เป็น `mock`) → `npm run build` → deploy `dist/`
URL: `https://toonrz.github.io/retail-shelf-web/` (Vite `base` = `/retail-shelf-web/`)

## 9. การตัดสินใจ

| เรื่อง | เลือก | เหตุผล |
|---|---|---|
| agent รันที่ไหน | ใน browser (onnxruntime-web, WebGPU → WASM) | ฟรี, ไม่มี server ให้หลับตอน demo, รูปไม่ออกจากเครื่อง |
| ภาษา agent | TypeScript (+ Python สำหรับ export) | browser รันได้แค่ JS/WASM |
| model | YOLOv8n | README: v8n เร็วกว่าเกือบ 2× mAP@0.5:0.95 ต่างแค่ 0.76 pp |
| model อยู่ไหน | ใน release ของ agent (เว็บ pin ตัวเดียว) | กันโค้ด pre/post กับ model ต่างเวอร์ชัน |
| host | GitHub Pages | ฟรีสำหรับ repo public |
| ความถูกต้องของ release | SHA256SUMS + sha256 ของ model ใน browser | พอสำหรับงานส่ง · Sigstore ทีหลังได้ |
| release asset | flat ไม่มีโฟลเดอร์ | GitHub release ไม่รองรับโฟลเดอร์ (เปลี่ยนจาก `ort/` ใน mock) |
| ไฟล์ release ในเว็บ | ไม่ commit ดึงตอน build | release ใหญ่ (ort wasm) · pin อยู่ใน `agent.version` ไฟล์เดียว |

ไม่เลือก: edge (ต้องมี fleet, เกินขอบเขต) · sidecar บน HF Spaces / Render (ฟรีแต่ server หลับ ปลุกช้า)

## 10. เรื่องที่ยังเปิดอยู่

- ขนาด release: ort ให้ wasm มา 4 แบบ (14–28 MB) ถ้ารวมทุกแบบ release ใหญ่ ~104 MB → งาน A1 เลือกเฉพาะที่ใช้ แล้วแก้ filter ใน `package-release.mjs`
- License: Ultralytics YOLOv8 เป็น AGPL-3.0 และ dataset เป็น CC BY 4.0 → ใส่ที่มาในหน้า About ของเว็บ (งาน B4)
- เครื่องที่ไม่มี WebGPU จะช้า → ต้องวัดบนเครื่องที่จะใช้พรีเซนต์จริง (งาน A5)
