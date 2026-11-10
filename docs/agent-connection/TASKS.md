# Tasks: Shelf Agent ↔ Web

อ่าน [SPEC.md](SPEC.md) ก่อน · หน้าตาที่ตกลงกันไว้: [mock/agent-connection.html](../../mock/agent-connection.html)

| งาน | Repo | ใคร | ต้องรออะไร |
|---|---|---|---|
| T0 โครงทั้งหมด + contract | ทั้ง 3 | ✅ ทำแล้ว | — |
| T1–T4 model จริง, สิทธิ์, release แรก | detection / ตั้งค่า GitHub | **Toon** | best.pt จาก Drive |
| **A** inference ใน browser | retail-shelf-agent | เพื่อน A: _(ชื่อ)_ | ไม่ต้องรอ (ใช้ model COCO dev ไปก่อน) |
| **B** หน้าเว็บ | retail-shelf-web | เพื่อน B: _(ชื่อ)_ | ไม่ต้องรอ (`agent.version = mock`) |
| **C** release, bump, deploy | agent + web | เพื่อน C: _(ชื่อ)_ | ไม่ต้องรอ (ใช้ model COCO dev ไปก่อน) |

A, B, C แตะไฟล์คนละชุดกัน จึงทำพร้อมกันได้เลย ใครจะแก้ไฟล์ contract (§5 ของ spec) ต้องคุยกันก่อน

## กติกา

- ทำใน branch ของตัวเอง แล้วเปิด PR เข้า `main` · CI ต้องผ่าน (`npm test`, `npm run build`, และฝั่ง agent มี `npm run typecheck` ด้วย)
- ไฟล์ contract คือ `retail-shelf-agent/src/types.ts` กับ `retail-shelf-web/src/agent-contract.ts` · แก้แล้วต้องแก้ทั้งสองที่ใน PR คู่กัน
- ทุกงานต้องมีหลักฐานใน PR: test ที่เพิ่ม และ/หรือ screenshot / log ตามที่เขียนไว้ใน "ตรวจ" ของงานนั้น

---

## T0 — ทำแล้ว (2026-11-10)

**retail-shelf-detection**
- [x] `src/export_agent_assets.py` export `best.pt` เป็น `model.onnx` + `labels.json` + `model.json` + fixture ที่ Ultralytics ทายจาก onnx ไฟล์เดียวกัน
  - ทดสอบกับ COCO `yolov8n.pt` แล้ว: output `[1,84,8400]`, opset 18, fixture 3 รูป (แนวตั้ง / แนวนอน / crop)
- [x] SPEC.md, TASKS.md, `export/` อยู่ใน `.gitignore`

**retail-shelf-agent**
- [x] `src/types.ts` (contract), `src/manifest.ts` (`parseManifest`, `sha256Hex`)
- [x] `load()` ตรวจ manifest → pin → contract → sha256 ของ model → labels แล้วจึงเรียก `createSession` (ตอนนี้ยัง throw `not_implemented`)
- [x] `scripts/package-release.mjs` (release แบบ flat + `manifest.json` + `SHA256SUMS`) ลองกับไฟล์จริงแล้ว `shasum -c` ผ่าน
- [x] `scripts/fetch-model.mjs` + `model.lock.json` → release `dev-coco-model`
- [x] `test/fixtures/coco-dev/` · vitest 7 ข้อ · CI
- [x] ทดสอบใน browser จริง: เว็บโหลด `shelf-agent.js` → manifest → model 12 MB (sha256 ผ่าน) → labels แล้วไปหยุดที่ `createSession: task A1`

**retail-shelf-web**
- [x] Vite + TypeScript, `agent.version` (`mock` | `vX.Y.Z`, ค่าอื่น build ล้ม), `src/agent.ts` (`loadAgent`), `src/mockAgent.ts`, `src/agent-contract.ts`
- [x] หน้าเว็บแบบเรียบที่สุด (เลือกรูป → detect → วาดกล่อง + JSON) · vitest 3 ข้อ · CI

---

## งานของ Toon

- [ ] **T1** อัปโหลด weights: `gh release create weights-v1 weights/yolov8n.pt weights/yolov8s.pt -R ToonRz/retail-shelf-detection`
  (ไฟล์ best.pt จาก `MyDrive/retail-shelf-detection/runs/detect/<model>/weights/`) ทำให้ Streamlit app ใช้ได้ด้วย
  - ⛔ ติดอยู่ (2026-11-10): ไม่มี `best.pt` ในเครื่อง (ไม่มี Drive mount, ไม่อยู่ใน `~/Downloads`/`~/Documents`) ต้อง login Google Drive หรือดาวน์โหลดมาวางที่ `weights/` ก่อน
  - ⛔ ยังติด (2026-11-10, รอบ 2): เจอไฟล์ใน Drive ที่แชร์มาแล้ว (`runs/detect/yolov8n/weights/best.pt` 6.5 MB, `runs/detect/yolov8s/weights/best.pt` 22.6 MB) แต่ Google Drive connector ดาวน์โหลด binary ขนาดนี้ไม่ได้ (ผลลัพธ์ใหญ่เกิน connector ตัดการเชื่อมต่อ) · ยังไม่มี release `weights-v1` · ทางไป: ดาวน์โหลด best.pt มาวางที่ `weights/` เอง แล้วรันคำสั่งด้านล่าง
- [ ] **T2** export model จริง (บน Colab หรือเครื่องที่มี dataset):
  ```bash
  python src/export_agent_assets.py --images datasets/sell-products/test/images --n 16
  gh release create model-yolov8n-v3 export/yolov8n/model.onnx export/yolov8n/labels.json export/yolov8n/model.json -R ToonRz/retail-shelf-detection
  ```
  จากนั้นเปิด PR ใน retail-shelf-agent:
  - copy `export/yolov8n/fixtures/` ไปที่ `test/fixtures/sell-v3/`
  - แก้ `model.lock.json` ให้ชี้ไปที่ `model-yolov8n-v3` พร้อม sha256 ของทั้ง 3 ไฟล์ (ค่าอยู่ใน `model.json` หรือใช้ `shasum -a 256`)
  - ตรวจ: `npm run fetch-model` ผ่าน และ parity test ของ A ผ่านบน `sell-v3`
  - ⛔ ติดอยู่ (2026-11-10): รอ T1 (`best.pt`) และ `ROBOFLOW_API_KEY` ใน `.env` (สำหรับรูป test split ที่ใช้ทำ fixture)
  - ⛔ ยังติด (2026-11-10, รอบ 2): รอ T1 เหมือนเดิม (ไม่มี `best.pt` · ห้ามเทรนใหม่และห้ามใช้ COCO แทน) จึงยังไม่ได้ export / release / แก้ agent
  - ทำต่อเมื่อมี `weights/yolov8n.pt` และ `.env` แล้ว (ใช้ ultralytics 8.4.208):
    ```bash
    gh release create weights-v1 weights/yolov8n.pt weights/yolov8s.pt -R ToonRz/retail-shelf-detection   # T1
    python -c "import sys; sys.path.insert(0, 'src'); from data import download_dataset; download_dataset(3, 'datasets/sell-products')"
    python src/export_agent_assets.py --images datasets/sell-products/test/images --n 16   # ต้องได้ 110 คลาส, output [1,114,8400]
    gh release create model-yolov8n-v3 export/yolov8n/model.onnx export/yolov8n/labels.json export/yolov8n/model.json -R ToonRz/retail-shelf-agent
    ```
    แล้วใน agent: แก้ `model.lock.json` + copy fixture (ไม่ใช้รูปที่เห็นหน้าคน) → `npm run fetch-model && npm run typecheck && npm test`
- [x] **T3** เพิ่มเพื่อนเป็น collaborator ของทั้ง 2 repo และเปิด Pages ใน retail-shelf-web (Settings → Pages → Source: **GitHub Actions**)
  - ✅ (2026-11-10) Pages เปิดแล้ว (`build_type: workflow`) · ส่ง invite สิทธิ์ write ให้ BoonyasakR, phichitphongpond, KXNBYV ทั้งใน agent และ web แล้ว (รอเพื่อนกด accept)
- [ ] **T4** หลัง A + C1 + T2 เสร็จ: `git tag v0.1.0 && git push --tags` ใน agent → รอ release → ใน web รัน `make bump-agent ALLOW_MINOR=1` แล้วเปิด PR
  - ⛔ ติดอยู่ (2026-11-10): `src/detect.ts` ยัง throw `not_implemented` (A1–A4), ยังไม่มี `release.yml` (C1) และ `make bump-agent` (C2), และรอ T2

---

## งาน A — inference ใน browser (retail-shelf-agent)

แก้ `src/detect.ts` (และเพิ่มไฟล์ใน `src/` ได้) · ห้ามแก้ `load()` / `types.ts` โดยไม่คุยกัน · ทำตาม SPEC §6

- [ ] **A1 `createSession`**
  - ใช้ onnxruntime-web และตั้ง `ort.env.wasm.wasmPaths = baseUrl` (release เป็น flat)
  - ลอง `webgpu` ก่อน ถ้าไม่ได้ให้ใช้ `wasm` แล้วคืนค่า `backend` ที่ใช้จริง · ใช้ไม่ได้ทั้งคู่ → `AgentError("backend_unavailable")`
  - เลือกว่าจะส่งไฟล์ `ort-wasm-*` ชุดไหนจริง (ตอนนี้ copy ไปทั้ง 4 แบบ release เลยใหญ่ ~104 MB) แล้วแก้ filter ใน `scripts/package-release.mjs` ให้ test ยังผ่าน
  - ตรวจ: Chrome (WebGPU) และ Safari/Firefox (WASM) โหลด release dev ได้ และสถานะขึ้น backend ที่ถูกต้อง
- [ ] **A2 preprocess** เป็นฟังก์ชันล้วน `letterbox(rgba, w, h) → { tensor, r, left, top }` ตาม SPEC §6.1
  - ระวังการปัดเลข half-to-even และใช้ resize แบบ bilinear
  - ตรวจ: unit test ที่ขนาด 640×640 (ต้องไม่ resize), 810×1080 และ 1280×720 เทียบค่า `r/left/top` กับที่คำนวณด้วย Python
- [ ] **A3 postprocess** เป็นฟังก์ชันล้วน `decode(output, C, opts, meta) → Box[]` ตาม SPEC §6.3 (argmax, conf >, NMS แยกตามคลาส, max_det, ถอด letterbox, clip)
  - ตรวจ: unit test ด้วย output ปลอม เช่น กล่องซ้อนกันคนละคลาสต้องไม่ถูกตัด, กล่องคลาสเดียวกันที่ IoU > iou ต้องถูกตัด
- [ ] **A4 parity test** รัน `model.onnx` ใน vitest (เช่น `onnxruntime-node` + `jpeg-js` เป็น devDependency) กับ `test/fixtures/coco-dev/` เทียบกับ `expected.json` ตามเกณฑ์ SPEC §6.4
  - ตรวจ: ผ่านบน `coco-dev` ตอนนี้ และต้องผ่านบน `sell-v3` หลัง T2
- [ ] **A5 วัดความเร็ว**: หน้า bench เล็ก ๆ หรือ script ที่จับ `latency_ms` เฉลี่ยของ 20 รูปบน WebGPU และ WASM ในเครื่องที่จะใช้พรีเซนต์ แล้วใส่ตัวเลขใน PR

## งาน B — หน้าเว็บ (retail-shelf-web)

แก้ `index.html`, `src/main.ts` และเพิ่มไฟล์ UI ใน `src/` ได้ · เรียก agent ผ่าน `loadAgent()` ใน `src/agent.ts` เท่านั้น
ใช้ `agent.version = mock` ไปก่อน · หน้าตาดูจากแท็บ "ทดลองตรวจจับ" ใน mock

- [ ] **B1 input**
  - อัปโหลดรูป และลากรูปมาวางได้
  - มีปุ่ม "ใช้รูปตัวอย่าง" (ใส่รูปไว้ใน `public/samples/` ใช้รูปจาก dataset ได้เพราะเป็น CC BY 4.0 แต่ต้องใส่ที่มา)
  - ถ่ายจากกล้องด้วย `getUserMedia` แล้วตรวจเฟรมนั้น
  - มีตัวเลือก backend (WebGPU / WASM)
- [ ] **B2 ผลลัพธ์**
  - วาดกล่องบน canvas ให้แต่ละคลาสมีสีของตัวเอง
  - ตารางนับจำนวนสินค้าแต่ละชนิด
  - บอกสินค้าที่มั่นใจที่สุด และ `latency_ms` + backend
  - slider ปรับ conf (ส่งเป็น `detect(img, { conf })`)
- [ ] **B3 ระหว่างโหลดและตอนพัง**
  - progress bar ตอนโหลด model (ใช้ `onProgress`)
  - ปิดปุ่มไว้จนกว่าจะโหลดเสร็จ
  - ข้อความภาษาไทยของแต่ละ `AgentErrorCode` (ดู `src/agent-contract.ts`) เช่น `version_mismatch` = "เวอร์ชัน agent ไม่ตรงกับที่เว็บ pin ไว้"
- [ ] **B4 About + responsive**
  - หน้า About: ตัวเลข mAP / ความเร็วจาก README ของ repo นี้, ลิงก์รายงาน, ที่มาของ dataset (CC BY 4.0) และ YOLOv8 (AGPL-3.0)
  - ใช้บนมือถือได้ และรองรับ dark mode
- ตรวจ: screenshot บน desktop และมือถือแนบใน PR · `npm test` + `npm run build` ผ่าน · ลองกับ release dev จริงเมื่อ C2 เสร็จ

## งาน C — release, bump, deploy (agent + web)

ทำตาม SPEC §4, §7, §8 · ใช้ model COCO dev (`v0.0.x`) ทดสอบทั้งเส้นทาง

- [ ] **C1 release workflow** (`retail-shelf-agent/.github/workflows/release.yml`) ทำงานเมื่อ push tag `v*.*.*`
  1. `npm ci` → `npm run typecheck` → `npm test` → `npm run build` → `npm run fetch-model`
  2. `npm run package -- --version $TAG` → `gh release create $TAG release/$TAG/*`
  - tag ที่ไม่ใช่ `vX.Y.Z` ต้องไม่ทำอะไรเลย
  - ตรวจ: push `v0.0.1` แล้วได้ release ที่ดาวน์โหลดมา `sha256sum -c SHA256SUMS` ผ่าน
- [ ] **C2 bump + fetch** (`retail-shelf-web/scripts/` + `Makefile`): ทำ `make fetch-agent-release` และ `make bump-agent` ตาม SPEC §7
  - ตรวจด้วย test อัตโนมัติ (vitest หรือ bats) ที่ใช้โฟลเดอร์ release ปลอม (สร้างด้วย `package-release.mjs`) ครอบคลุม:
    - bump patch ผ่าน
    - ข้าม minor ถูกปฏิเสธถ้าไม่มี `ALLOW_MINOR=1`
    - ถอยเวอร์ชันถูกปฏิเสธ
    - SHA256SUMS ผิด / manifest ผิดเวอร์ชันถูกปฏิเสธ และ `agent.version` กลับเป็นค่าเดิม
- [ ] **C3 Pages deploy** (`retail-shelf-web/.github/workflows/pages.yml`) ตาม SPEC §8 และ CI warning เมื่อ pin ตามหลัง patch ล่าสุด
  - ตรวจ: เว็บขึ้นที่ `https://toonrz.github.io/retail-shelf-web/` ทั้งตอน pin เป็น `mock` และตอน pin `v0.0.1`

---

## ลำดับการรวมงาน

1. A1–A4 + C1 → tag `v0.0.1` (COCO) → C2 bump เว็บไป `v0.0.1` → C3 deploy → ได้เว็บที่ตรวจจับได้จริง (แต่ยังเป็นคลาส COCO)
2. T2 → lock model จริง + fixture `sell-v3` → parity ผ่าน → T4 tag `v0.1.0` → bump เว็บ → B เสร็จ → ถ่าย screenshot ลงรายงาน (README ข้อ 8)
