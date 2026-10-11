"""Fill the auto-generated sections of README.md from the files in results/.

Run after evaluate.py (the notebook does this for you):

    python src/make_report.py --results results --readme README.md

Only the text between the ``<!-- AUTO:<name>:START -->`` / ``<!-- AUTO:<name>:END -->``
markers is replaced, so the hand-written analysis stays untouched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

COLS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("map50", "mAP50"), ("map50_95", "mAP50-95"), ("accuracy", "Accuracy"),
        ("auc_macro_ovr", "AUC (macro OvR)"), ("infer_ms_per_img", "ms/img"),
        ("fps", "FPS (e2e)"), ("params_m", "Params (M)"), ("gflops", "GFLOPs"), ("weights_mb", "Size (MB)")]

LOWER_IS_BETTER = {"infer_ms_per_img", "params_m", "gflops", "weights_mb"}


def _fmt(v, col):
    if pd.isna(v):
        return "-"
    if col in ("infer_ms_per_img", "fps", "weights_mb", "params_m", "gflops"):
        return f"{v:.1f}" if col != "params_m" else f"{v:.2f}"
    return f"{v:.3f}"


def md_table(df: pd.DataFrame, cols) -> str:
    head = "| Model | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|" + "|".join("---:" for _ in cols) + "|"
    body = []
    for name, row in df.iterrows():
        cells = []
        for c, _ in cols:
            if c not in df:
                df[c] = float("nan")
            best = df[c].max() if c not in LOWER_IS_BETTER else df[c].min()
            s = _fmt(row[c], c)
            cells.append(f"**{s}**" if len(df) > 1 and row[c] == best else s)
        body.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body])


def simple_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def findings(res: Path, df: pd.DataFrame) -> str:
    lines = []
    if len(df) >= 2:
        small, big = df.sort_values("params_m").index[[0, -1]]
        a, b = df.loc[small], df.loc[big]
        d_map = 100 * (b.map50_95 - a.map50_95)
        d_m50 = 100 * (b.map50 - a.map50)
        d_acc = 100 * (b.accuracy - a.accuracy)
        slow = b.infer_ms_per_img / a.infer_ms_per_img
        lines += [
            f"- **{big}** เทียบกับ **{small}**: mAP50-95 {d_map:+.2f} จุด, mAP50 {d_m50:+.2f} จุด, "
            f"accuracy ระดับรูป {d_acc:+.2f} จุด",
            f"- ต้นทุน: {big} ใช้เวลา inference **{slow:.2f} เท่า** "
            f"({a.infer_ms_per_img:.1f} -> {b.infer_ms_per_img:.1f} ms/รูป), parameters "
            f"{b.params_m / a.params_m:.1f} เท่า, ไฟล์ใหญ่ {b.weights_mb / a.weights_mb:.1f} เท่า",
            f"- mAP50-95 ที่ได้เพิ่มต่อเวลาที่ช้าลง 1 ms: "
            f"{d_map / max(b.infer_ms_per_img - a.infer_ms_per_img, 1e-6):.2f} จุด/ms",
            f"- รูปที่ไม่มีกล่องไหน conf >= 0.25 (no detection): {small} {100 * a.no_detection_rate:.2f}%, "
            f"{big} {100 * b.no_detection_rate:.2f}%",
        ]
    dis = res / "disagreement.json"
    if dis.exists():
        d = json.loads(dis.read_text())
        lab = {"both_correct": "ถูกทั้งคู่", "both_wrong": "ผิดทั้งคู่"}
        lines.append("- รูปใน test set (ระดับรูป): " + ", ".join(
            f"{lab.get(k, k.replace('only_', 'ถูกเฉพาะ ').replace('_correct', ''))} = {v}"
            for k, v in d.items()))
    return "\n".join(lines)


def confusion_section(res: Path, models) -> str:
    out = []
    for m in models:
        f = res / f"top_confusions_{m}.csv"
        if f.exists():
            t = pd.read_csv(f)
            out.append(f"**{m}** – 10 คู่ (คลาสจริง -> คลาสที่ทาย) ที่ผิดบ่อยที่สุด ระดับรูป\n\n"
                       + (simple_table(t) if len(t) else "_ไม่มีรูปที่ทายผิด_"))
    return "\n\n".join(out)


def hardest_section(res: Path, models, k=10) -> str:
    if not models:
        return ""
    ref = models[-1]
    f = res / f"per_class_{ref}.csv"
    if not f.exists():
        return ""
    t = pd.read_csv(f).sort_values("ap50_95").head(k)
    cols = ["class", "precision", "recall", "ap50", "ap50_95", "support", "img_recall", "auc"]
    t = t[[c for c in cols if c in t]]
    for m in models[:-1]:
        g = res / f"per_class_{m}.csv"
        if g.exists():
            o = pd.read_csv(g).set_index("class")["ap50_95"]
            t[f"ap50_95 ({m})"] = t["class"].map(o)
    return f"10 คลาสที่ AP50-95 ต่ำที่สุดของ **{ref}** (test set) เทียบกับโมเดลเล็ก:\n\n" + simple_table(t.round(3))


def eda_section(res: Path) -> str:
    f = res / "eda_splits.csv"
    if not f.exists():
        return ""
    t = pd.read_csv(f)
    return simple_table(t)


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
        (res / "comparison.md").write_text(md + "\n", encoding="utf-8")
        text = replace_block(text, "table", md)
        text = replace_block(text, "findings", findings(res, df))
        text = replace_block(text, "confusions", confusion_section(res, models))
        text = replace_block(text, "hardest", hardest_section(res, models))
    Path(readme).write_text(text, encoding="utf-8")
    print(f"updated {readme}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--readme", default="README.md")
    a = ap.parse_args()
    main(a.results, a.readme)
