#!/usr/bin/env python3

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )

from pathlib import Path
from decimal import Decimal
import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
PANELS = [
    ("Ridge", "available"),
    ("Ridge", "unavailable"),
    ("HGB", "available"),
    ("HGB", "unavailable"),
]
BLOCKS = list("DBMPCAK")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )


def plot2(f2):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharex=True, sharey=True)
    for ax, ((model, hist), p) in zip(axes.flat, zip(PANELS, f2.values())):
        x = np.arange(31)
        dev = np.array(p["development"], float)
        nested = np.array(p["nested"], float)
        temporal = np.array(p["temporal"], float)
        ax.plot(
            x,
            dev,
            color="#777777",
            linestyle=":",
            lw=1.6,
            marker="o",
            ms=4,
            markevery=5,
            mfc="white",
            label="Development OOF",
        )
        ax.plot(
            x,
            nested,
            color="#505050",
            linestyle="--",
            lw=1.6,
            marker="s",
            ms=4,
            markevery=5,
            mfc="white",
            label="Nested validation",
        )
        ax.plot(
            x,
            temporal,
            color="#111111",
            linestyle="-",
            lw=1.75,
            marker="^",
            ms=4,
            markevery=5,
            label="Temporal evaluation",
        )
        ax.axhline(
            float(p["benchmark"]["value"]),
            color="#3c3c3c",
            linestyle="-.",
            lw=1.3,
            label="Temporal benchmark",
        )
        ax.set_title(f"{model} / History {hist}")
        ax.set_xlim(0, 30)
        ax.set_ylim(0.60, 0.77)
        ax.set_xticks(np.arange(0, 31, 5))
        ax.set_yticks(np.arange(0.60, 0.761, 0.02))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", color="#dddddd", lw=0.7)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Log loss (lower is better)")
    for ax in axes[1, :]:
        ax.set_xlabel("Additional original-item budget", labelpad=7)
    hs, ls = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        hs,
        ls,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.042),
        ncol=2,
        frameon=False,
        columnspacing=2,
    )
    fig.text(
        0.5,
        0.014,
        "Temporal benchmark: P0 when history is available; NONE when history is unavailable.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.955, bottom=0.18, wspace=0.10, hspace=0.22
    )
    fig.savefig(OUT / "figure2.png", dpi=320)
    fig.savefig(OUT / "figure2.svg")
    plt.close(fig)


def plot3(f3):
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.0), sharex=True, sharey=True)
    for ax, ((key, p)) in zip(axes.flat, f3.items()):
        y = np.arange(7)
        mid = np.array(p["temporal"], float)
        lo = np.array(p["lower"], float)
        hi = np.array(p["upper"], float)
        ax.plot(
            np.array(p["development"], float),
            y - 0.13,
            linestyle="none",
            marker="s",
            mfc="white",
            mec="#555555",
            mew=1.1,
            ms=5,
        )
        ax.errorbar(
            mid,
            y + 0.13,
            xerr=np.vstack([mid - lo, hi - mid]),
            fmt="o",
            color="#111111",
            markersize=4.2,
            capsize=2.6,
            elinewidth=1.15,
        )
        ax.axvline(0, color="#777777", lw=0.9, linestyle="--")
        ax.grid(axis="x", color="#dddddd", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{p['model']} / H {'fixed' if p['history']=='included' else 'excluded'}"
        )
        ax.set_yticks(y, BLOCKS)
        ax.set_ylim(6.55, -0.55)
        ax.set_xlim(-0.08, 0.055)
        ax.set_xticks(np.arange(-0.08, 0.041, 0.02))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:.2f}".replace("-", "−"))
        )
    for ax in axes[1, :]:
        ax.set_xlabel("Average marginal contribution\n(log-loss reduction)", labelpad=7)
    hs = [
        Line2D([], [], linestyle="none", marker="s", mfc="white", mec="#555555", ms=5),
        Line2D([], [], color="#111111", marker="o", ms=4.2, lw=1.15),
    ]
    fig.legend(
        hs,
        ["Development OOF (descriptive)", "Temporal / simultaneous 95% interval"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        frameon=False,
        columnspacing=2,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.955, bottom=0.20, wspace=0.17, hspace=0.22
    )
    fig.savefig(OUT / "figure3.png", dpi=320)
    fig.savefig(OUT / "figure3.svg")
    plt.close(fig)


def validate(data):
    assert list(data["figure2"]) == [f"{m}|{h}" for m, h in PANELS]
    assert list(data["figure3"]) == [
        "Ridge|included",
        "Ridge|excluded",
        "HGB|included",
        "HGB|excluded",
    ]
    for panel in data["figure2"].values():
        assert panel["budgets"] == list(range(31))
        for key in ["development", "nested", "temporal"]:
            assert len(panel[key]) == 31
            assert all(Decimal(v).is_finite() for v in panel[key])
        assert panel["benchmark"]["label"] == (
            "P0" if panel["history"] == "available" else "NONE"
        )
    for panel in data["figure3"].values():
        assert panel["blocks"] == BLOCKS
        for key in ["development", "temporal", "lower", "upper"]:
            assert len(panel[key]) == 7
            assert all(Decimal(v).is_finite() for v in panel[key])
        for lo, mid, hi in zip(panel["lower"], panel["temporal"], panel["upper"]):
            assert Decimal(lo) <= Decimal(mid) <= Decimal(hi)


def main():
    global OUT
    ap = argparse.ArgumentParser(
        description="Regenerate the revised figures from aggregate JSON, without the source DOCX or microdata."
    )
    ap.add_argument("--data", type=Path, default=ROOT / "data/plot_data.json")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.data.is_file():
        ap.error(f"Aggregate plotting data not found: {args.data.resolve()}")
    try:
        data = json.loads(args.data.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        ap.error(f"Cannot read aggregate plotting data {args.data}: {exc}")
    validate(data)
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot2(data["figure2"])
    plot3(data["figure3"])
    print(
        "Rendered figure2.png/.svg and figure3.png/.svg from validated full-precision aggregate values."
    )


if __name__ == "__main__":
    main()
