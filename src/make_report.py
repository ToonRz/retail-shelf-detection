"""Fill the auto-generated sections of README.md from the files in results/.

Run after evaluate.py (the notebook does this for you):

    python src/make_report.py --results results --readme README.md

Only the text between the ``<!-- AUTO:<name>:START -->`` / ``<!-- AUTO:<name>:END -->``
markers is replaced, so the hand-written analysis stays untouched. Differences of 0-1
metrics are written in percentage points (pp): 0.9693 -> 0.9769 = +0.76 pp.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

K_HARDEST = 10

COLS = [("precision", "Box P"), ("recall", "Box R"), ("f1", "Box F1"),
        ("map50", "mAP50"), ("map50_95", "mAP50-95"), ("accuracy", "Img Acc"),
        ("auc_macro_ovr", "Img AUC"), ("infer_ms_per_img", "ms/img (model)"),
        ("e2e_ms_per_img", "ms/img (e2e)"), ("params_m", "Params (M)"), ("gflops", "GFLOPs"),
        ("weights_mb", "Size (MB)")]

LOWER_IS_BETTER = {"infer_ms_per_img", "e2e_ms_per_img", "params_m", "gflops", "weights_mb"}
UNIT_METRICS = {"precision", "recall", "f1", "map50", "map50_95", "accuracy", "auc_macro_ovr"}
MIN_GAP = 0.005  # a 0-1 metric is only bolded when the models differ by >= 0.5 pp


def _fmt(v, col):
    if pd.isna(v):
        return "-"
    if col in UNIT_METRICS:
        return f"{v:.3f}"
    return f"{v:.2f}" if col == "params_m" else f"{v:.1f}"


def md_table(df: pd.DataFrame, cols) -> str:
    df = df.reindex(columns=list(dict.fromkeys([*df.columns, *[c for c, _ in cols]])))
    head = "| Model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    body = []
    for name, row in df.iterrows():
        cells = []
        for c, _ in cols:
            col = df[c].astype(float)
            best = col.min() if c in LOWER_IS_BETTER else col.max()
            gap = col.max() - col.min()
            meaningful = gap >= MIN_GAP if c in UNIT_METRICS else _fmt(col.max(), c) != _fmt(col.min(), c)
            s = _fmt(row[c], c)
            bold = len(df) > 1 and meaningful and s == _fmt(best, c)
            cells.append(f"**{s}**" if bold else s)
        body.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def table_notes(df: pd.DataFrame) -> str:
    conf = ""
    if "max_f1_conf" in df and df["max_f1_conf"].notna().all():
        conf = " (" + ", ".join(f"{m} {v:.3f}" for m, v in df["max_f1_conf"].items()) + ")"
    return ("\n\n_Box P/R/F1: IoU 0.5 ที่ confidence ซึ่งให้ mean F1 สูงสุดของแต่ละโมเดล" + conf +
            " · mAP: conf 0.001, ไม่ขึ้นกับ threshold · Img Acc/AUC: ระดับรูป, top-1 ที่ conf ≥ 0.25 · "
            "ms/img (model) = inference อย่างเดียว, (e2e) = รวม pre/post-processing, median ที่ batch 1 · "
            f"ตัวหนา = ดีกว่าอย่างน้อย {MIN_GAP * 100:.1f} pp (หรือดีกว่าสำหรับคอลัมน์ความเร็ว/ขนาด)_")


def simple_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _per_class(res: Path, m: str) -> pd.DataFrame | None:
    f = res / f"per_class_{m}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f).rename(columns={"support": "n_images"}).set_index("class")


def paired_tests(res: Path, a: str, b: str) -> dict:
    """Paired per-class comparison of b vs a (same test set, same checkpoints)."""
    from scipy import stats

    pa, pb = _per_class(res, a), _per_class(res, b)
    if pa is None or pb is None:
        return {}
    out = {}
    for col in ("ap50_95", "ap50"):
        d = (pb[col] - pa[col]).dropna()
        better, worse = int((d > 0).sum()), int((d < 0).sum())
        rng = np.random.default_rng(0)
        boots = [d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        out[col] = {"better": better, "worse": worse, "equal": int((d == 0).sum()),
                    "n": int(len(d)), "mean_pp": 100 * d.mean(),
                    "ci_pp": tuple(100 * np.percentile(boots, [2.5, 97.5])),
                    "wilcoxon_p": float(stats.wilcoxon(pb[col], pa[col]).pvalue) if (d != 0).any() else 1.0,
                    "sign_p": float(stats.binomtest(better, better + worse).pvalue) if better + worse else 1.0}
    return out


def findings(res: Path, df: pd.DataFrame) -> str:
    lines = []
    if len(df) >= 2:
        small, big = df.sort_values("params_m").index[[0, -1]]
        a, b = df.loc[small], df.loc[big]
        pp = lambda col: 100 * (b[col] - a[col])  # noqa: E731
        slow = b.infer_ms_per_img / a.infer_ms_per_img
        lines += [
            f"- **{big}** เทียบกับ **{small}** (pp = จุดเปอร์เซ็นต์ บนสเกล 0-100): mAP50-95 {pp('map50_95'):+.2f} pp, "
            f"mAP50 {pp('map50'):+.2f} pp, box precision {pp('precision'):+.2f} pp, "
            f"box recall {pp('recall'):+.2f} pp, accuracy ระดับรูป {pp('accuracy'):+.2f} pp",
            f"- ต้นทุน: {big} ใช้เวลา inference (median) **{slow:.2f} เท่า** "
            f"({a.infer_ms_per_img:.1f} -> {b.infer_ms_per_img:.1f} ms/รูป), end-to-end "
            f"{b.e2e_ms_per_img / a.e2e_ms_per_img:.2f} เท่า, parameters "
            f"{b.params_m / a.params_m:.1f} เท่า, ไฟล์ใหญ่ {b.weights_mb / a.weights_mb:.1f} เท่า",
        ]
        p90 = {m: np.percentile(pd.read_csv(res / f"predictions_{m}.csv").infer_ms, 90)
               for m in (small, big) if (res / f"predictions_{m}.csv").exists()}
        if len(p90) == 2:
            lines.append(f"- latency p90 (model only): {small} {p90[small]:.1f} ms, {big} {p90[big]:.1f} ms "
                         f"= {p90[big] / p90[small]:.2f} เท่า (ที่ median {slow:.2f} เท่า)")
        t = paired_tests(res, small, big)
        if t:
            r, r50 = t["ap50_95"], t["ap50"]
            verdict = "ไม่ต่างกัน" if r50["wilcoxon_p"] >= 0.05 else (
                f"{big} ดีกว่า" if r50["mean_pp"] > 0 else f"{small} ดีกว่า")
            lines.append(
                f"- ทดสอบแบบจับคู่รายคลาส ({r['n']} คลาส, test set และ checkpoint ชุดนี้): AP50-95 {big} ดีกว่า "
                f"{r['better']} / แย่กว่า {r['worse']} / เท่ากัน {r['equal']} คลาส, Wilcoxon p = {r['wilcoxon_p']:.4f}, "
                f"sign test p = {r['sign_p']:.4f}, bootstrap 95% CI ของค่าเฉลี่ย {r['ci_pp'][0]:+.2f} ถึง "
                f"{r['ci_pp'][1]:+.2f} pp · AP50: {r50['better']}/{r50['worse']}/{r50['equal']}, "
                f"Wilcoxon p = {r50['wilcoxon_p']:.2f} ({verdict})")
        lines.append(f"- รูปที่ไม่มีกล่องไหน conf >= 0.25 (no detection): {small} "
                     f"{100 * a.no_detection_rate:.2f}%, {big} {100 * b.no_detection_rate:.2f}%")
    dis = res / "disagreement.json"
    if dis.exists():
        d = json.loads(dis.read_text())
        lab = {"both_correct": "ถูกทั้งคู่", "both_wrong": "ผิดทั้งคู่"}
        disc = sum(v for k, v in d.items() if k.startswith("only_"))
        lines.append("- รูปใน test set (ระดับรูป): " + ", ".join(
            f"{lab.get(k, k.replace('only_', 'ถูกเฉพาะ ').replace('_correct', ''))} = {v}"
            for k, v in d.items()) + (" → ไม่มีรูปที่สองโมเดลให้ผลต่างกัน (McNemar ไม่มีข้อมูลให้ทดสอบ)"
                                       if disc == 0 else ""))
    return "\n".join(lines)


def confusion_section(res: Path, models) -> str:
    tabs = {m: pd.read_csv(res / f"top_confusions_{m}.csv") for m in models
            if (res / f"top_confusions_{m}.csv").exists()}
    out = []
    if tabs:
        swaps = {m: t[t["pred"] != "(no detection)"] for m, t in tabs.items()}
        if all(len(t) == 0 for t in swaps.values()):
            nodet = {m: int(t["count"].sum()) for m, t in tabs.items()}
            out.append("**ระดับรูป (top-1 ที่ conf ≥ 0.25):** ไม่มีการทายสลับคลาสเลย ความผิดพลาดมีแค่ "
                       "no detection: " + ", ".join(f"{m} {n} รูป" for m, n in nodet.items()))
        else:
            for m, t in tabs.items():
                out.append(f"**{m}** – คู่ (คลาสจริง -> คลาสที่ทาย) ที่ผิดบ่อยที่สุด ระดับรูป "
                           f"(พบ {len(t)} คู่)\n\n" + simple_table(t))
    for m in models:
        f = res / f"box_confusions_{m}.csv"
        if f.exists():
            t = pd.read_csv(f)
            out.append(f"**{m}** – class swap ระดับกล่องที่ conf ≥ 0.25 (จาก confusion matrix ครบ 110 คลาส, "
                       "ไม่รวมกล่องที่หาไม่เจอ/กล่องเกิน ซึ่งอยู่ในตารางความผิดพลาดระดับกล่อง)\n\n"
                       + (simple_table(t) if len(t) else "_ไม่มี_"))
    return "\n\n".join(out)


def hardest_section(res: Path, models, k=K_HARDEST) -> str:
    if not models:
        return ""
    ref = models[-1]
    t = _per_class(res, ref)
    if t is None:
        return ""
    if "n_boxes" not in t and (res / "eda_class_counts.csv").exists():
        t["n_boxes"] = pd.read_csv(res / "eda_class_counts.csv", index_col="class")["test"]
    t = t.sort_values("ap50_95").head(k).reset_index()
    view = pd.DataFrame({"class": t["class"], "test img": t.get("n_images"), "test box": t.get("n_boxes"),
                         "box P": t["precision"], "box R": t["recall"], "AP50": t["ap50"],
                         "AP50-95": t["ap50_95"]})
    for m in models[:-1]:
        o = _per_class(res, m)
        if o is not None:
            view[f"AP50-95 ({m})"] = t["class"].map(o["ap50_95"])
    if "img_recall" in t:
        view["img recall"] = t["img_recall"]
    return (f"{k} คลาสที่ AP50-95 ต่ำที่สุดของ **{ref}** (test set). box P/R = ระดับกล่องที่ IoU 0.5 "
            f"ณ conf ที่ให้ mean F1 สูงสุด, img recall = ระดับรูป:\n\n" + simple_table(view.round(3)))


def box_error_section(res: Path, models) -> str:
    rows = []
    for m in models:
        f = res / f"metrics_{m}.json"
        if f.exists():
            e = json.loads(f.read_text()).get("box_errors")
            if e:
                rows.append({"model": m, **e})
    if not rows:
        return ("_ยังไม่มีข้อมูล: ต้องรัน notebook ใหม่ (เฉพาะส่วนประเมิน) ด้วย src/evaluate.py เวอร์ชันนี้ "
                "จึงจะได้ results/box_errors_<model>.csv และ examples_box_errors_<model>.jpg_")
    return ("ความผิดพลาดระดับกล่องที่ conf ≥ 0.25 และ IoU ≥ 0.5 (missed = GT ที่ไม่มีกล่องคลาสเดียวกันครอบ, "
            "wrong_class = กล่องทับ GT แต่ผิดคลาส, duplicate = กล่องซ้ำบนสินค้าที่เจอแล้ว, "
            "poor_loc = ทับ GT แต่ IoU 0.1–0.5, bg_fp = กล่องบนพื้นหลัง):\n\n"
            + simple_table(pd.DataFrame(rows)))


def train_section(res: Path, models) -> str:
    rows = []
    for m in models:
        f = res / f"metrics_{m}.json"
        if f.exists():
            tr = json.loads(f.read_text()).get("train")
            if tr:
                rows.append({"model": m, **{k: v for k, v in tr.items() if not k.endswith("_error")}})
    info = res / "run_info.json"
    txt = ""
    if rows:
        txt += simple_table(pd.DataFrame(rows))
    if info.exists():
        ri = json.loads(info.read_text())
        txt += "\n\nสภาพแวดล้อมตอนประเมิน: " + ", ".join(f"{k} {v}" for k, v in ri.items())
    return txt or ("_ยังไม่มีข้อมูล: optimizer/lr ที่ใช้จริง เวลาเทรน และชื่อ GPU จะถูกบันทึกเมื่อรัน "
                   "notebook ใหม่ (ค่าที่อนุมานได้ตอนนี้อยู่ในตาราง 5.5)_")


def eda_section(res: Path) -> str:
    f = res / "eda_splits.csv"
    if not f.exists():
        return ""
    return simple_table(pd.read_csv(f))


def replace_block(text: str, key: str, content: str) -> str:
    pat = re.compile(rf"(<!-- AUTO:{key}:START -->)(.*?)(<!-- AUTO:{key}:END -->)", re.S)
    if not pat.search(text):
        print(f"marker AUTO:{key} not found in README - skipped")
        return text
    return pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)


def main(results="results", readme="README.md"):
    res = Path(results)
    text = Path(readme).read_text(encoding="utf-8")
    text = replace_block(text, "eda", eda_section(res))
    comp = res / "comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp, index_col="model")
        models = list(df.sort_values("params_m").index)
        md = md_table(df.loc[models], COLS)
        (res / "comparison.md").write_text(md + table_notes(df.loc[models]) + "\n", encoding="utf-8")
        text = replace_block(text, "table", md + table_notes(df.loc[models]))
        text = replace_block(text, "findings", findings(res, df))
        text = replace_block(text, "confusions", confusion_section(res, models))
        text = replace_block(text, "hardest", hardest_section(res, models))
        text = replace_block(text, "boxerrors", box_error_section(res, models))
        text = replace_block(text, "train", train_section(res, models))
    Path(readme).write_text(text, encoding="utf-8")
    print(f"updated {readme}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    main(a.results, a.readme)
