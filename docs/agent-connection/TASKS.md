# Tasks: Shelf Agent ↔ Web

อ่าน [SPEC.md](SPEC.md) ก่อน · หน้าตาที่ตกลงกันไว้: [mock/agent-connection.html](../../mock/agent-connection.html)

| งาน | Repo | ใคร | ต้องรออะไร |
|---|---|---|---|
| T0 โครงทั้งหมด + contract | ทั้ง 3 | ✅ ทำแล้ว | — |
| T1–T4 model จริง, สิทธิ์, release แรก | detection / ตั้งค่า GitHub | **Toon** | ✅ เสร็จทั้งหมด |
| **A** inference ใน browser | retail-shelf-agent | ✅ ทำแล้ว (2026-11-10) | — |
| **B** หน้าเว็บ | retail-shelf-web | ✅ ทำแล้ว (2026-11-10) | — |
| **C** release, bump, deploy | agent + web | ✅ ทำแล้ว (2026-11-10) | — |
| **S** หน้าเฝ้าชั้นวาง (แจ้งเตือนของหมด): model 1 คลาส, agent `v0.2.0`, หน้า `#watch` | ทั้ง 3 | ✅ ทำแล้ว (2026-11-12) · README [ข้อ 8.3](../../README.md#83-หน้าเฝ้าชั้นวาง-แจ้งเตือนของหมด) | — |
| **M** รวม 2 model ใน release เดียว (agent `v0.3.x`, `manifest.shelfModel`): หน้าตรวจจับได้ชื่อสินค้า 110 คลาส + กล่อง `product` ของที่ไม่รู้จัก | agent + web | ✅ ทำแล้ว (2026-11-12) · agent PR #1 → `v0.3.0`, `v0.3.1` (แก้แถบโหลด) · เว็บ pin `agent.version` = `v0.3.1` | — |
| **F1–F3, I1–I3** ทดสอบจริง, ความเร็ว, รายงาน + นำเสนอ, โค้ดเพิ่มคนละชิ้น | detection (README ข้อ 2, 8) + agent + web | เพื่อน 3 คน → [งานที่เหลือ](#งานที่เหลือ-เพื่อน-3-คน) | ไม่ต้องรอ |

A, B, C แตะไฟล์คนละชุดกัน จึงทำพร้อมกันได้เลย ใครจะแก้ไฟล์ contract (§5 ของ spec) ต้องคุยกันก่อน

## กติกา

- ทำใน branch ของตัวเอง แล้วเปิด PR เข้า `main` · CI ต้องผ่าน (`npm test`, `npm run build`, และฝั่ง agent มี `npm run typecheck` ด้วย)
- ไฟล์ contract คือ `retail-shelf-agent/src/types.ts` กับ `retail-shelf-web/src/agent-contract.ts` · แก้แล้วต้องแก้ทั้งสองที่ใน PR คู่กัน
- ทุกงานต้องมีหลักฐานใน PR: test ที่เพิ่ม และ/หรือ screenshot / log ตามที่เขียนไว้ใน "ตรวจ" ของงานนั้น

---

## งานที่เหลือ (เพื่อน 3 คน)

โค้ดทุกส่วนเสร็จแล้วและเว็บขึ้นแล้วที่ <https://toonrz.github.io/retail-shelf-web/> (หน้าตรวจจับ agent `v0.3.1` = model 110 คลาส + shelf model · หน้าเฝ้าชั้นวาง agent `v0.3.1` ตั้งแต่ 2026-11-13, เดิม `v0.2.1`)
ที่เหลือคืองานที่ต้องใช้คน / เครื่องจริง, รายงาน + นำเสนอ (F) และงานเขียนโค้ดเพิ่มคนละชิ้น (I) · ช่อง _(เพื่อน N กรอก)_ อยู่ใน [README ข้อ 8](../../README.md#ข้อ-8--ต้นแบบ-prototype)
ตกลงกันเองว่าใครเป็นเพื่อน 1 / 2 / 3 แล้วใส่ชื่อในวงเล็บ · แก้ README ผ่าน PR เข้า `main` ได้เลย (ไม่ต้องรอกัน)

### เพื่อน 1 — ทดสอบกับเครื่องและสินค้าจริง: _(ชื่อ)_

- [ ] **F1.1 ทดสอบหลาย browser** เปิดเว็บแล้วไล่ทำตามตารางนี้ทีละเครื่อง (✅ / ❌ + หมายเหตุ)

  | | Chrome (คอม) | Safari (Mac) | Firefox | Safari (iPhone) | Chrome (Android) |
  |---|---|---|---|---|---|
  | โหลดจนขึ้น "พร้อมใช้" + backend ที่ได้ (WebGPU/WASM) | | | | | |
  | รูปตัวอย่าง → มีกรอบ + ตารางนับ | | | | | |
  | อัปโหลดรูปเอง / ลากรูปมาวาง | | | | | |
  | กล้อง → ถ่ายและตรวจ | | | | | |
  | slider ความมั่นใจ → ผลเปลี่ยน | | | | | |
  | หน้าเกี่ยวกับ + dark mode | | | | | |

  - เจอปัญหา: เปิด issue ใน `retail-shelf-web` (หรือ `retail-shelf-agent` ถ้าเป็นเรื่องโหลด model) พร้อม browser, เครื่อง, ข้อความ error และ screenshot
  - ใส่สรุปลง README ข้อ 8.1.2 แถว "อุปกรณ์จริง" และข้อ 8.1.5 บรรทัด Safari / Firefox
- [ ] **F1.2 ทดสอบกับสินค้าจริงนอก dataset** ถ่ายสินค้าที่อยู่ใน 110 คลาส (ดูรายชื่อใน `labels.json` ของ release) จากร้านจริงด้วยกล้องของเว็บ อย่างน้อย 20 รูป
  - ให้มีหลายแบบ: ชิ้นเดียว / หลายชิ้นในรูป / บนชั้นวาง / แสงน้อย / เอียง
  - จดลงตาราง README ข้อ 8.1.4 (top-1 ถูก, ไม่เจอ, ผิดชนิด) และข้อสังเกต
  - เลือกรูปที่ดีและที่พลาด 2–4 รูปไว้ใน `results/field_*.jpg` · **ห้ามมีหน้าคนในรูป**
- [ ] **F1.3 เตรียมอธิบายตอนนำเสนอ:** agent ทำงานยังไง (letterbox → onnxruntime-web → NMS) และ parity test (README 8.1.1–8.1.2, `retail-shelf-agent/src/`)

### เพื่อน 2 — ความเร็ว + แอป Streamlit: _(ชื่อ)_

- [ ] **F2.1 วัดความเร็ว (A5)** บนโน้ตบุ๊กที่จะใช้นำเสนอ และมือถือ 1 เครื่อง: หน้าเกี่ยวกับ → *วัดความเร็วเครื่องนี้*
  - จดเฉลี่ย WebGPU / WASM และเวลาโหลด model ครั้งแรก ลงตาราง README ข้อ 8.1.3 (ใส่รุ่นเครื่อง + browser)
  - ถ้า WebGPU ขึ้นว่า "ไม่มี" บน Chrome ให้ลองเปิด `chrome://gpu` ดูว่า WebGPU ถูกปิดหรือไม่ แล้วจดไว้
- [ ] **F2.2 แอป Streamlit กับ `best.pt` จริง** (`pip install -r requirements-app.txt && streamlit run app/streamlit_app.py`)
  - ทดสอบโหมดรูป / วิดีโอ / กล้อง ทั้ง YOLOv8n และ YOLOv8s (แอปโหลด `weights-v1` เอง)
  - ถ่ายภาพหน้าจอใหม่ด้วยรูปสินค้าจริงแทน `results/prototype_screenshot.png` (ตอนนี้เป็นภาพ COCO)
  - อัปเดตบรรทัด "ยังไม่ได้ทดสอบ" ใน README ข้อ 8.2.2
- [ ] **F2.3 (ถ้ามีเวลา) deploy Streamlit Community Cloud** ตามขั้นตอนใน README 8.2.2 แล้วใส่ลิงก์ `.streamlit.app` ในตารางต้นข้อ 8
- [ ] **F2.4 เตรียมอธิบายตอนนำเสนอ:** หน้าเว็บ (`retail-shelf-web/src/main.ts`) และเทียบเว็บ browser กับ Streamlit (ตารางต้นข้อ 8)

### เพื่อน 3 — รายงาน + การนำเสนอ: _(ชื่อ)_

- [ ] **F3.1 README ข้อ 2** กรอกชื่อ รหัส และบทบาทของทุกคนให้ตรงกับที่ทำจริง (รวมงานเว็บ / agent / ทดสอบ / นำเสนอ)
- [ ] **F3.2 เกลาร่าง README ข้อ 8** อ่านทั้งข้อแล้วแก้ภาษาให้อ่านง่าย · เมื่อเพื่อน 1 และ 2 กรอกครบ ให้ลบ placeholder และกล่อง "ร่าง" ด้านบนข้อ 8
  - เช็กว่าข้อ 9 (สรุป) ยังตรงกับข้อ 8 ใหม่ เช่น เพิ่มเว็บ browser ในข้อ 9.4 แนวทางพัฒนาต่อถ้าจำเป็น
- [ ] **F3.3 สไลด์** ปัญหา → ข้อมูล → model → ผล (ข้อ 7) → ต้นแบบ (แผนภาพ 3 repo จาก 8.1.1) → demo → สรุป (ข้อ 9)
- [ ] **F3.4 บทสาธิต + แผนสำรอง**
  - ลำดับ demo: เปิดเว็บ → รูปตัวอย่าง → ถ่ายสินค้าจริงด้วยกล้อง → ปรับ slider → หน้าเกี่ยวกับ
  - เปิดเว็บทิ้งไว้ก่อนเริ่มนำเสนอ (model ~43 MB จะได้อยู่ใน cache) · เตรียมคลิปอัดหน้าจอไว้เผื่อเน็ตหรือกล้องมีปัญหา
- [ ] **F3.5 เตรียมอธิบายตอนนำเสนอ:** release / pin / bump (`release.yml`, `make bump-agent`) และเหตุผลที่เลือกรันใน browser (SPEC §9)

### งานเขียนโค้ด (คนละชิ้น)

แต่ละชิ้นแก้ไฟล์คนละชุด จึงทำพร้อมกันได้ · ทำใน branch ของตัวเอง → PR เข้า `main` → CI ต้องผ่าน · **ห้ามแก้ contract** (`types.ts` / `agent-contract.ts`)
ถ้าจำเป็นต้องแก้ไฟล์ร่วม เช่น `src/main.ts` หรือ `index.html` ให้แก้แค่จุดที่ต่อเข้ากับไฟล์ใหม่ของตัวเอง (ไม่กี่บรรทัด)
เสร็จแล้วเพิ่มย่อหน้าสั้น ๆ ใน README ข้อ 8.1 ว่าทำอะไร + ภาพหรือตัวเลขประกอบ

#### I1 (เพื่อน 1) — test ใน browser จริงใน CI ของ agent · repo `retail-shelf-agent`

ตอนนี้ parity test รันบน Node อย่างเดียว ส่วนการทดสอบใน Chromium ทำด้วยมือ ให้ CI ทดสอบ release ใน browser ทุกครั้ง

- [ ] เพิ่ม `@playwright/test` เป็น devDependency และ `test/e2e/` (แยกจาก vitest)
- [ ] test: `npm run build` → `npm run package -- --version v0.0.0` → เปิด static server ที่ serve `release/v0.0.0/` + `test/fixtures/sell-v3/`
  → เปิด Chromium → `load({ baseUrl, expectedVersion: "v0.0.0", backend: "wasm" })` → `detect()` ทุกรูปใน `sell-v3` (เป็น `HTMLImageElement`)
  → เทียบกับ `expected.json` ด้วยเกณฑ์เดียวกับ `parityProblems` ใน `test/parity.test.ts`
- [ ] test เพิ่ม: `expectedVersion` ผิด → `version_mismatch` · แก้ไบต์ใน `model.onnx` (ผ่าน `page.route`) → `model_sha256_mismatch`
- [ ] เพิ่ม job ใน `.github/workflows/ci.yml` (`npx playwright install --with-deps chromium`) และ `npm run test:e2e`
- ตรวจ: CI เขียว · ลองแก้ `iou` ใน `src/postprocess.ts` ชั่วคราวแล้ว e2e ต้องแดง (แนบ log ทั้งสองแบบใน PR แล้วค่อยคืนค่า)

#### I2 (เพื่อน 2) — ตรวจต่อเนื่องจากกล้อง (real-time) · repo `retail-shelf-web`

ตอนนี้กล้องถ่ายได้ทีละภาพ ให้มีโหมดที่ตรวจทุกเฟรมแล้ววาดกรอบทับวิดีโอสด

- [ ] ไฟล์ใหม่ `src/live.ts`: ปุ่ม "ตรวจต่อเนื่อง" / "หยุด" · วนลูป: จับเฟรม (`createImageBitmap(video)`) → `agent.detect` → วาดกรอบบน canvas ที่ซ้อนอยู่บน `<video>` → เฟรมถัดไป
  - ห้ามส่งเฟรมใหม่ระหว่างที่เฟรมก่อนยังตรวจไม่เสร็จ (ข้ามเฟรมแทนการต่อคิว)
  - แสดง FPS และ ms ต่อเฟรม · หยุดเองเมื่อสลับแท็บ (`document.visibilityState`) · ปิดกล้องแล้วต้องคืนกล้องจริง (`track.stop()`)
  - ตารางนับสินค้าอัปเดตตามเฟรมล่าสุด (ใช้ `countByClass` เดิมใน `src/ui/logic.ts`)
- [ ] ส่วนที่เป็นตรรกะล้วน (ตัวนับ FPS แบบค่าเฉลี่ยเคลื่อนที่, ตัวตัดสินว่าจะข้ามเฟรมไหม) แยกไว้ใน `src/ui/live-logic.ts` + unit test ใน `test/live.test.ts`
- [ ] ถ้า UI กระตุกตอนใช้ WASM (~400 ms ต่อเฟรม) ลองย้าย `loadAgent` + `detect` ไปทำใน Web Worker (ไม่บังคับ)
- ตรวจ: `npm test` + `npm run build` ผ่าน · แนบคลิปสั้นหรือ GIF จากมือถือจริงใน PR · จด FPS ที่ได้บน WebGPU / WASM ลง README 8.1.3

#### I3 (เพื่อน 3) — ใช้ offline ได้ + ตรวจหลายรูปแล้วดาวน์โหลด CSV · repo `retail-shelf-web`

- [ ] **offline:** service worker (`public/sw.js` หรือ `src/sw.ts`) cache หน้าเว็บ + `assets/` + `agent/<เวอร์ชัน>/*`
  - ชื่อ cache ต้องมีเวอร์ชัน agent (`AGENT_VERSION`) และลบ cache เวอร์ชันเก่าเมื่อ bump · โหลดครั้งที่สองไม่ต้องดาวน์โหลด model 43 MB ใหม่
  - ขึ้นป้าย "ใช้ offline ได้แล้ว" เมื่อ cache ครบ · ตรรกะเลือก/ลบ cache แยกเป็นฟังก์ชันล้วน + unit test
- [ ] **ตรวจหลายรูป:** ไฟล์ใหม่ `src/batch.ts`: เลือกหลายรูปพร้อมกัน → ตรวจทีละรูป (มีแถบความคืบหน้า) → ตาราง ชื่อไฟล์ / top-1 / จำนวนแต่ละสินค้า
  - ปุ่ม "ดาวน์โหลด CSV" (คอลัมน์: file, product, count, max_conf) · ฟังก์ชันสร้าง CSV แยกเป็นฟังก์ชันล้วน + unit test (ระวังชื่อไฟล์ที่มี `,` หรือ `"`)
- ตรวจ: `npm test` + `npm run build` ผ่าน · โหลดเว็บครั้งหนึ่ง → DevTools → Network → Offline → รีโหลด → ยังตรวจรูปได้ (แนบ screenshot) · แนบ CSV ตัวอย่างใน PR

#### เพิ่มเติม (ถ้ามีเวลา, ใครก็ได้)

- export YOLOv8s เป็น ONNX แล้วออก agent `v0.2.0` ให้เลือก model ในเว็บได้ (เปลี่ยน model = minor → `make bump-agent ALLOW_MINOR=1`) · ต้องคุยกันก่อนเพราะอาจต้องแก้ contract
- ทำตาม issue ที่เจอจาก F1.1
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
- [x] **T4** release แรก [`v0.1.0`](https://github.com/ToonRz/retail-shelf-agent/releases/tag/v0.1.0) แล้ว bump เว็บ
  - ✅ (2026-11-10) `release.yml` (Run workflow, version `v0.1.0`) สร้าง release ไฟล์ flat 7 ไฟล์ ~43 MB · `sha256sum -c SHA256SUMS` ผ่านใน workflow
  - web: `make bump-agent ALLOW_MINOR=1` (mock → v0.1.0) ผ่าน test + build · Pages deploy พร้อม release แล้ว: https://toonrz.github.io/retail-shelf-web/
  - ที่เหลือ: กดวัดความเร็ว (A5) บนเครื่องที่จะพรีเซนต์ แล้วถ่าย screenshot ลงรายงาน (README ข้อ 8)
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
