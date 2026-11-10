# Tasks: Shelf Agent ↔ Web

อ่าน [SPEC.md](SPEC.md) ก่อน · หน้าตาที่ตกลงกันไว้: [mock/agent-connection.html](../../mock/agent-connection.html)

| งาน | Repo | ใคร | ต้องรออะไร |
|---|---|---|---|
| T0 โครงทั้งหมด + contract | ทั้ง 3 | ✅ ทำแล้ว | — |
| T1–T4 model จริง, สิทธิ์, release แรก | detection / ตั้งค่า GitHub | **Toon** | T1–T3 เสร็จ · T4 เหลือกดออก release `v0.1.0` |
| **A** inference ใน browser | retail-shelf-agent | ✅ ทำแล้ว (2026-11-10) | — |
| **B** หน้าเว็บ | retail-shelf-web | ✅ ทำแล้ว (2026-11-10) | — |
| **C** release, bump, deploy | agent + web | ✅ ทำแล้ว (2026-11-10) | — |

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

- [x] **T1** อัปโหลด weights: release [`weights-v1`](https://github.com/ToonRz/retail-shelf-detection/releases/tag/weights-v1) ของ repo นี้ (ทำให้ Streamlit app ใช้ได้ด้วย)
  (ไฟล์ best.pt จาก `MyDrive/retail-shelf-detection/runs/detect/<model>/weights/`)
  - ✅ (2026-11-10) `yolov8n.pt` 6,559,018 B sha256 `fe2557b6b7ba6c369c85ee47ad8c5f1eabbe78c7de2597912c669efc18760f61` · `yolov8s.pt` 22,603,562 B sha256 `3684e2b58424d8a967ab7261012dd887d95d213103ad10cd240fad0844f44404`
  - ตรวจ: sha256 ของไฟล์ที่ดึงจาก Drive = digest ของ asset ใน release = ไฟล์ที่ดาวน์โหลดกลับมา (ตรงทั้ง 2 ไฟล์)
  - ทำผ่าน GitHub Actions ครั้งเดียว (`publish-weights`) เพราะ Drive connector ส่ง binary ขนาดนี้ไม่ได้ (ลบ workflow ออกแล้ว)
- [x] **T2** export model จริง → release [`model-yolov8n-v3`](https://github.com/ToonRz/retail-shelf-agent/releases/tag/model-yolov8n-v3) ของ **retail-shelf-agent**
  ```bash
  python -c "import sys; sys.path.insert(0, 'src'); from data import download_dataset; download_dataset(3, 'datasets/sell-products')"
  python src/export_agent_assets.py --images datasets/sell-products/test/images --n 16   # ultralytics 8.4.208
  ```
  - ✅ (2026-11-10) 110 คลาส · opset 18 · output `[1,114,8400]` · `model.onnx` sha256 `f970340c26c5e2354bf2c15593a5d3fe96e5cc65fd421a79cae0dd97099d420f` (จาก `weights-v1/yolov8n.pt`)
  - agent: `model.lock.json` ชี้ `model-yolov8n-v3` พร้อม sha256 ทั้ง 3 ไฟล์ · `test/fixtures/sell-v3/` = 15 รูป test + 1 crop ที่ไม่เป็นสี่เหลี่ยมจัตุรัส + `expected.json` จาก onnx ไฟล์เดียวกับที่อัปโหลด (`model_sha256` ตรง lock)
  - ไม่ใช้รูปที่เห็นคน: ตัด `IMG_5355` (มีคนนั่งอยู่ด้านหลัง) ออกจาก 16 รูปที่สคริปต์เลือก
  - ตรวจ: `npm run fetch-model` (sha256 ผ่าน), `npm run typecheck`, `npm test` (7/7) และ CI ของ agent ผ่าน · parity test บน `sell-v3` เป็นของงาน A4
- [x] **T3** เพิ่มเพื่อนเป็น collaborator ของทั้ง 2 repo และเปิด Pages ใน retail-shelf-web (Settings → Pages → Source: **GitHub Actions**)
  - ✅ (2026-11-10) Pages เปิดแล้ว (`build_type: workflow`) · ส่ง invite สิทธิ์ write ให้ BoonyasakR, phichitphongpond, KXNBYV ทั้งใน agent และ web แล้ว (รอเพื่อนกด accept)
- [ ] **T4** ออก release แรก `v0.1.0` แล้ว bump เว็บ
  - A, C1, C2, T2 เสร็จหมดแล้ว · ทดสอบ release `v0.1.0` ที่ package ในเครื่องกับเว็บใน Chromium แล้ว (WASM + WebGPU, ผลตรง `expected.json`)
  - ⛔ ค้างที่ขั้นเดียว: session ของ Claude push tag / สั่งรัน release workflow ไม่ได้ → Toon ทำเอง:
    - GitHub → retail-shelf-agent → Actions → **release** → Run workflow → version `v0.1.0` (หรือ `git tag v0.1.0 && git push origin v0.1.0`)
    - แล้วใน web: `make bump-agent ALLOW_MINOR=1` → commit `agent.version` (= `v0.1.0`) → push / PR → Pages deploy เอง
---

## งาน A — inference ใน browser (retail-shelf-agent)

แก้ `src/detect.ts` (และเพิ่มไฟล์ใน `src/` ได้) · ห้ามแก้ `load()` / `types.ts` โดยไม่คุยกัน · ทำตาม SPEC §6

- [x] **A1 `createSession`** ✅ `src/detect.ts` · release ส่งเฉพาะ `ort-wasm-simd-threaded.jsep.{wasm,mjs}` (~43 MB ทั้ง release) · ทดสอบใน Chromium แล้วทั้ง WebGPU และ WASM (ยังไม่ได้ลอง Safari/Firefox จริง)
  - ใช้ onnxruntime-web และตั้ง `ort.env.wasm.wasmPaths = baseUrl` (release เป็น flat)
  - ลอง `webgpu` ก่อน ถ้าไม่ได้ให้ใช้ `wasm` แล้วคืนค่า `backend` ที่ใช้จริง · ใช้ไม่ได้ทั้งคู่ → `AgentError("backend_unavailable")`
  - เลือกว่าจะส่งไฟล์ `ort-wasm-*` ชุดไหนจริง (ตอนนี้ copy ไปทั้ง 4 แบบ release เลยใหญ่ ~104 MB) แล้วแก้ filter ใน `scripts/package-release.mjs` ให้ test ยังผ่าน
  - ตรวจ: Chrome (WebGPU) และ Safari/Firefox (WASM) โหลด release dev ได้ และสถานะขึ้น backend ที่ถูกต้อง
- [x] **A2 preprocess** ✅ `src/preprocess.ts` (resize แบบ fixed-point เท่า cv2) · test เทียบค่าจาก Python ใน `test/preprocess.test.ts` เป็นฟังก์ชันล้วน `letterbox(rgba, w, h) → { tensor, r, left, top }` ตาม SPEC §6.1
  - ระวังการปัดเลข half-to-even และใช้ resize แบบ bilinear
  - ตรวจ: unit test ที่ขนาด 640×640 (ต้องไม่ resize), 810×1080 และ 1280×720 เทียบค่า `r/left/top` กับที่คำนวณด้วย Python
- [x] **A3 postprocess** ✅ `src/postprocess.ts` · ถอด letterbox แยก gain x/y แบบ `scale_boxes` ของ 8.4.208 · `test/postprocess.test.ts` เป็นฟังก์ชันล้วน `decode(output, C, opts, meta) → Box[]` ตาม SPEC §6.3 (argmax, conf >, NMS แยกตามคลาส, max_det, ถอด letterbox, clip)
  - ตรวจ: unit test ด้วย output ปลอม เช่น กล่องซ้อนกันคนละคลาสต้องไม่ถูกตัด, กล่องคลาสเดียวกันที่ IoU > iou ต้องถูกตัด
- [x] **A4 parity test** ✅ `test/parity.test.ts` (onnxruntime-web wasm ใน node + jpeg-js) ผ่านบน `sell-v3` 16 รูป และ `coco-dev` 3 รูป · CI ผ่าน รัน `model.onnx` ใน vitest (เช่น `onnxruntime-node` + `jpeg-js` เป็น devDependency) กับ `test/fixtures/coco-dev/` เทียบกับ `expected.json` ตามเกณฑ์ SPEC §6.4
  - ตรวจ: ผ่านบน `coco-dev` ตอนนี้ และต้องผ่านบน `sell-v3` หลัง T2
- [x] **A5 วัดความเร็ว** ✅ หน้า About ของเว็บ → "วัดความเร็วเครื่องนี้" (20 รูปต่อ backend) · ใน container ที่ทดสอบ: WASM เฉลี่ย 384 ms · ⚠️ ยังต้องกดวัดบนเครื่องที่จะใช้พรีเซนต์แล้วจดตัวเลข: หน้า bench เล็ก ๆ หรือ script ที่จับ `latency_ms` เฉลี่ยของ 20 รูปบน WebGPU และ WASM ในเครื่องที่จะใช้พรีเซนต์ แล้วใส่ตัวเลขใน PR

## งาน B — หน้าเว็บ (retail-shelf-web)

แก้ `index.html`, `src/main.ts` และเพิ่มไฟล์ UI ใน `src/` ได้ · เรียก agent ผ่าน `loadAgent()` ใน `src/agent.ts` เท่านั้น
ใช้ `agent.version = mock` ไปก่อน · หน้าตาดูจากแท็บ "ทดลองตรวจจับ" ใน mock

- [x] **B1 input** ✅
  - อัปโหลดรูป และลากรูปมาวางได้
  - มีปุ่ม "ใช้รูปตัวอย่าง" (ใส่รูปไว้ใน `public/samples/` ใช้รูปจาก dataset ได้เพราะเป็น CC BY 4.0 แต่ต้องใส่ที่มา)
  - ถ่ายจากกล้องด้วย `getUserMedia` แล้วตรวจเฟรมนั้น
  - มีตัวเลือก backend (WebGPU / WASM)
- [x] **B2 ผลลัพธ์** ✅
  - วาดกล่องบน canvas ให้แต่ละคลาสมีสีของตัวเอง
  - ตารางนับจำนวนสินค้าแต่ละชนิด
  - บอกสินค้าที่มั่นใจที่สุด และ `latency_ms` + backend
  - slider ปรับ conf (ส่งเป็น `detect(img, { conf })`)
- [x] **B3 ระหว่างโหลดและตอนพัง** ✅ (`src/ui/logic.ts` `ERROR_TH`)
  - progress bar ตอนโหลด model (ใช้ `onProgress`)
  - ปิดปุ่มไว้จนกว่าจะโหลดเสร็จ
  - ข้อความภาษาไทยของแต่ละ `AgentErrorCode` (ดู `src/agent-contract.ts`) เช่น `version_mismatch` = "เวอร์ชัน agent ไม่ตรงกับที่เว็บ pin ไว้"
- [x] **B4 About + responsive** ✅
  - หน้า About: ตัวเลข mAP / ความเร็วจาก README ของ repo นี้, ลิงก์รายงาน, ที่มาของ dataset (CC BY 4.0) และ YOLOv8 (AGPL-3.0)
  - ใช้บนมือถือได้ และรองรับ dark mode
- ตรวจ: screenshot อยู่ที่ `docs/screenshots/` ของ retail-shelf-web (desktop สว่าง, มือถือมืด, About) ถ่ายกับ release `v0.1.0` ที่ package ในเครื่อง · `npm test` + `npm run build` ผ่าน

## งาน C — release, bump, deploy (agent + web)

ทำตาม SPEC §4, §7, §8 · ใช้ model COCO dev (`v0.0.x`) ทดสอบทั้งเส้นทาง

- [x] **C1 release workflow** ✅ `release.yml` (push tag หรือ Run workflow ใส่เวอร์ชัน) · ข้าม `v0.0.1` เพราะ lock เป็น model จริงแล้ว ใช้ `v0.1.0` (T4) เป็นรอบตรวจ (`retail-shelf-agent/.github/workflows/release.yml`) ทำงานเมื่อ push tag `v*.*.*`
  1. `npm ci` → `npm run typecheck` → `npm test` → `npm run build` → `npm run fetch-model`
  2. `npm run package -- --version $TAG` → `gh release create $TAG release/$TAG/*`
  - tag ที่ไม่ใช่ `vX.Y.Z` ต้องไม่ทำอะไรเลย
  - ตรวจ: push `v0.0.1` แล้วได้ release ที่ดาวน์โหลดมา `sha256sum -c SHA256SUMS` ผ่าน
- [x] **C2 bump + fetch** ✅ `scripts/agent-release.mjs` + `Makefile` · `test/agent-release.test.ts` ครบทุกข้อด้านล่าง · ลอง `make fetch-agent-release` กับ release v0.1.0 ในเครื่องแล้ว (`retail-shelf-web/scripts/` + `Makefile`): ทำ `make fetch-agent-release` และ `make bump-agent` ตาม SPEC §7
  - ตรวจด้วย test อัตโนมัติ (vitest หรือ bats) ที่ใช้โฟลเดอร์ release ปลอม (สร้างด้วย `package-release.mjs`) ครอบคลุม:
    - bump patch ผ่าน
    - ข้าม minor ถูกปฏิเสธถ้าไม่มี `ALLOW_MINOR=1`
    - ถอยเวอร์ชันถูกปฏิเสธ
    - SHA256SUMS ผิด / manifest ผิดเวอร์ชันถูกปฏิเสธ และ `agent.version` กลับเป็นค่าเดิม
- [x] **C3 Pages deploy** ✅ `pages.yml` deploy แล้วตอน pin `mock` · CI ขึ้น warning เมื่อมี patch ใหม่กว่า pin (`make agent-lag`) (`retail-shelf-web/.github/workflows/pages.yml`) ตาม SPEC §8 และ CI warning เมื่อ pin ตามหลัง patch ล่าสุด
  - ตรวจ: เว็บขึ้นที่ `https://toonrz.github.io/retail-shelf-web/` ทั้งตอน pin เป็น `mock` และตอน pin `v0.0.1`

---

## ลำดับการรวมงาน

1. A1–A4 + C1 → tag `v0.0.1` (COCO) → C2 bump เว็บไป `v0.0.1` → C3 deploy → ได้เว็บที่ตรวจจับได้จริง (แต่ยังเป็นคลาส COCO)
2. T2 → lock model จริง + fixture `sell-v3` → parity ผ่าน → T4 tag `v0.1.0` → bump เว็บ → B เสร็จ → ถ่าย screenshot ลงรายงาน (README ข้อ 8)
