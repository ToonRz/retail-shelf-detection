"""Shared matplotlib style so every figure in results/ looks consistent."""
import matplotlib

matplotlib.use("Agg")  # headless (Colab / CI); notebooks still display saved PNGs
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order: slot 1 = first model, slot 2 = second model.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
GT_COLOR = "#008300"    # ground-truth boxes
PRED_COLOR = "#e34948"  # predicted boxes

MODEL_COLORS = {"yolov8n": SERIES[0], "yolov8s": SERIES[1]}


def model_color(name, i=0):
    return MODEL_COLORS.get(name, SERIES[i % len(SERIES)])


def apply():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


apply()
