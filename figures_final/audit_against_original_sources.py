#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from decimal import Decimal
from docx import Document
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_figures"
PANELS = [
    ("Ridge", "available"),
    ("Ridge", "unavailable"),
    ("HGB", "available"),
    ("HGB", "unavailable"),
]
BLOCKS = list("DBMPCAK")


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_label(p):
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def clean(s):
    return s.replace("−", "-").replace("\u00a0", " ").strip()


def num(s):
    return Decimal(clean(s))


def stringify(v):
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dict):
        return {k: stringify(x) for k, x in v.items()}
    if isinstance(v, list):
        return [stringify(x) for x in v]
    return v


def location(doc, table, row, col, value):
    return {
        "document": doc,
        "table": table,
        "row": row,
        "column": col,
        "original_cell_text": value,
    }


def model_label(value):
    label = clean(value).lower()
    if "릿지" in label or "ridge" in label:
        return "Ridge"
    if "hgb" in label:
        return "HGB"
    raise ValueError(f"Unrecognized model label: {value}")


def history_label(value):
    label = clean(value).lower()
    if label in {"미가용", "unavailable"}:
        return "unavailable"
    if label in {"가용", "available"}:
        return "available"
    raise ValueError(f"Unrecognized history availability label: {value}")


def fixed_history_label(value):
    label = clean(value).lower()
    if "제외" in label or "excluded" in label:
        return "excluded"
    if "포함" in label or "fixed" in label or "included" in label:
        return "included"
    raise ValueError(f"Unrecognized fixed-history label: {value}")


def extract(main_path, supp_path):
    main = Document(main_path)
    supp = Document(supp_path)
    main_tables = [table for table in main.tables if len(table.rows) > 1]
    supp_tables = [table for table in supp.tables if len(table.rows) > 1]
    if len(main_tables) < 5 or len(supp_tables) < 12:
        raise ValueError(
            "The source documents must contain main Tables 3 and 5 and supplementary Tables S7–S12."
        )
    expected = [(main_tables[2], 21), (main_tables[4], 13), (supp_tables[11], 29)]
    expected.extend((supp_tables[index], 32) for index in range(6, 10))
    if any(len(table.rows) != rows for table, rows in expected):
        raise ValueError(
            "Source table dimensions do not match the published Figures 2 and 3 inputs."
        )
    f2 = {}
    f3 = {}
    audit = []

    for idx, (model, hist) in zip([6, 7, 8, 9], PANELS):
        key = f"{model}|{hist}"
        f2[key] = {
            "model": model,
            "history": hist,
            "nested": [],
            "temporal": [],
            "development": [None] * 31,
            "budgets": list(range(31)),
            "selected_combination": [],
            "development_evidence": [None] * 31,
            "nested_evidence": [],
            "temporal_evidence": [],
        }
        for ri, row in enumerate(supp_tables[idx].rows[1:], 1):
            c = [clean(x.text) for x in row.cells]
            b = int(c[0])
            assert b == ri - 1
            f2[key]["selected_combination"].append(c[1])
            for series, ci in [("nested", 2), ("temporal", 3)]:
                f2[key][series].append(num(c[ci]))
                f2[key][series + "_evidence"].append(
                    location(source_label(supp_path), f"S{idx+1}", ri, ci + 1, c[ci])
                )

        bi = 6 if hist == "available" else 7
        bcell = clean(supp_tables[bi].rows[1].cells[3].text)
        f2[key]["benchmark"] = {
            "label": "P0" if hist == "available" else "NONE",
            "value": num(bcell),
            "evidence": location(source_label(supp_path), f"S{bi+1}", 1, 4, bcell),
        }
    for ri, row in enumerate(main_tables[2].rows[1:], 1):
        c = [clean(x.text) for x in row.cells]
        key = f"{model_label(c[1])}|{history_label(c[0])}"
        b = int(c[2])
        f2[key]["development"][b] = num(c[5])
        f2[key]["development_evidence"][b] = location(
            source_label(main_path), "3", ri, 6, c[5]
        )
        for ci, series in [(6, "nested"), (7, "temporal")]:
            assert num(c[ci]) == f2[key][series][b], (
                key,
                b,
                series,
                c[ci],
                f2[key][series][b],
            )
            audit.append(
                {
                    "check": "main Table 3 / supplementary Tables S7–S10 agreement",
                    "panel": key,
                    "budget": b,
                    "series": series,
                    "value": str(num(c[ci])),
                    "pass": True,
                }
            )
    for model, hist in PANELS:
        condition = "included" if hist == "available" else "excluded"
        key = f"{model}|{condition}"
        f3[key] = {
            "model": model,
            "history": condition,
            "blocks": BLOCKS,
            "development": [None] * 7,
            "temporal": [],
            "lower": [],
            "upper": [],
            "development_evidence": [None] * 7,
            "temporal_evidence": [],
            "interval_evidence": [],
        }
    for ri, row in enumerate(supp_tables[11].rows[1:], 1):
        c = [clean(x.text) for x in row.cells]
        model = model_label(c[0])
        hist = fixed_history_label(c[0])
        key = f"{model}|{hist}"
        bi = len(f3[key]["temporal"])
        assert c[1] == BLOCKS[bi]
        lo, hi = map(num, re.findall(r"[+\-]?\d+\.\d+", c[3]))
        mid = num(c[2])
        assert lo <= mid <= hi
        f3[key]["temporal"].append(mid)
        f3[key]["lower"].append(lo)
        f3[key]["upper"].append(hi)
        f3[key]["temporal_evidence"].append(
            location(source_label(supp_path), "S12", ri, 3, c[2])
        )
        f3[key]["interval_evidence"].append(
            location(source_label(supp_path), "S12", ri, 4, c[3])
        )
    for ri, row in enumerate(main_tables[4].rows[1:], 1):
        c = [clean(x.text) for x in row.cells]
        model = model_label(c[0])
        hist = fixed_history_label(c[1])
        key = f"{model}|{hist}"
        bi = BLOCKS.index(c[2][0])
        f3[key]["development"][bi] = num(c[3])
        f3[key]["development_evidence"][bi] = location(
            source_label(main_path), "5", ri, 4, c[3]
        )
        lo, hi = map(num, re.findall(r"[+\-]?\d+\.\d+", c[5]))
        assert num(c[4]) == f3[key]["temporal"][bi]
        assert lo == f3[key]["lower"][bi] and hi == f3[key]["upper"][bi]
        audit.append(
            {
                "check": "main Table 5 / supplementary Table S12 agreement",
                "panel": key,
                "block": BLOCKS[bi],
                "temporal": c[4],
                "lower": str(lo),
                "upper": str(hi),
                "pass": True,
            }
        )
    return f2, f3, audit


def integrate_complete(f2, f3, path):

    ext = json.loads(Path(path).read_text(encoding="utf-8"))
    for fig, current, n in [("figure2", f2, 31), ("figure3", f3, 7)]:
        for key, p in current.items():
            rec = ext[fig][key]
            assert len(rec["development"]) == n
            assert rec.get("source_file") and rec.get(
                "source_column"
            ), f"Missing provenance: {fig} / {key}"
            for i, v in enumerate(rec["development"]):
                value = Decimal(str(v))
                old = p["development"][i]
                if old is not None:
                    assert abs(value - old) <= Decimal(
                        "0.00005"
                    ), f"Development value disagrees with DOCX: {fig}/{key}/{i}"
                p["development"][i] = value
                p["development_evidence"][i] = {
                    "source_file": rec["source_file"],
                    "source_column": rec["source_column"],
                    "source_row": rec.get("source_rows", [None] * n)[i],
                    "supplied_value": str(value),
                    "docx_cross_check": str(old) if old is not None else None,
                }


def integrate_aggregates(f2, f3, base, checks):

    files = [
        "development_budget_policies.csv",
        "nested_budget_summary.csv",
        "temporal_budget_policy_metrics.csv",
        "block_attribution_development_descriptive.csv",
        "block_attribution_evaluation.csv",
    ]
    tables = {}
    hashes = {}
    for name in files:
        path = base / name
        with path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        tables[name] = [(i + 2, r) for i, r in enumerate(rows)]
        hashes[source_label(path)] = sha256(path)

    def select(name, filters):
        found = [
            (line, r)
            for line, r in tables[name]
            if all(r[k] == str(v) for k, v in filters.items())
        ]
        assert len(found) == 1, (name, filters, len(found))
        return found[0]

    def assign(panel, series, index, name, filters, column, kind):
        line, r = select(name, filters)
        value = Decimal(r[column])
        old = panel[series][index]
        if old is not None:

            diff = abs(value - old)
            assert diff <= Decimal("0.00005"), (
                kind,
                series,
                index,
                str(value),
                str(old),
            )
            checks.append(
                {
                    "check": "full-precision CSV / rounded DOCX agreement",
                    "figure": kind,
                    "panel": panel["model"] + "|" + panel["history"],
                    "series": series,
                    "index": index,
                    "csv_value": str(value),
                    "docx_value": str(old),
                    "absolute_difference": str(diff),
                    "pass": True,
                }
            )
        panel[series][index] = value
        ev = {
            "source_file": source_label(base / name),
            "source_csv_line": line,
            "source_column": column,
            "row_filter": filters,
            "exact_csv_string": r[column],
            "reported_docx_value": str(old) if old is not None else None,
        }
        if series in ["development", "nested", "temporal"]:
            panel[series + "_evidence"][index] = ev
        else:
            panel.setdefault(series + "_evidence", [None] * 7)[index] = ev
        return r

    for key, panel in f2.items():
        for b in range(31):
            filt = {
                "algorithm": panel["model"].lower(),
                "availability": "history_" + panel["history"],
                "budget": b,
            }
            rd = assign(
                panel,
                "development",
                b,
                files[0],
                filt,
                "development_oof_log_loss",
                "figure2",
            )
            rn = assign(
                panel, "nested", b, files[1], filt, "nested_log_loss_mean", "figure2"
            )
            rt = assign(
                panel,
                "temporal",
                b,
                files[2],
                {**filt, "evaluation_domain": "all"},
                "log_loss",
                "figure2",
            )
            assert (
                rd["selected_subset"] == rn["selected_subset"] == rt["selected_subset"]
            )
            assert (
                Decimal(rd["development_oof_log_loss"])
                == Decimal(rn["development_oof_log_loss"])
                == Decimal(rt["development_oof_log_loss"])
            )
            assert rt["n"] == "1520"
        bf = {
            "algorithm": "ridge",
            "availability": "history_" + panel["history"],
            "budget": 0,
            "evaluation_domain": "all",
        }
        line, br = select(files[2], bf)
        label = "P0" if panel["history"] == "available" else "NONE"
        assert (
            br["source_model"] == ("P0" if label == "P0" else "ridge::NONE")
            and Decimal(br["delta_baseline_log_loss"]) == 0
        )
        bv = Decimal(br["log_loss"])
        assert abs(bv - panel["benchmark"]["value"]) <= Decimal("0.00005")
        panel["benchmark"] = {
            "label": label,
            "value": bv,
            "evidence": {
                "source_file": source_label(base / files[2]),
                "source_csv_line": line,
                "source_column": "log_loss",
                "row_filter": bf,
                "exact_csv_string": br["log_loss"],
            },
        }
        for b in range(31):
            _, tr = select(
                files[2],
                {
                    "algorithm": panel["model"].lower(),
                    "availability": "history_" + panel["history"],
                    "budget": b,
                    "evaluation_domain": "all",
                },
            )
            assert tr["baseline_model"] == ("P0" if label == "P0" else "prevalence")
            assert abs(
                Decimal(tr["log_loss"]) - Decimal(tr["delta_baseline_log_loss"]) - bv
            ) < Decimal("1e-12")
    for key, panel in f3.items():
        for i, block in enumerate(BLOCKS):
            filt = {
                "algorithm": panel["model"].lower(),
                "has_history": "True" if panel["history"] == "included" else "False",
                "block": block,
            }
            assign(
                panel,
                "development",
                i,
                files[3],
                filt,
                "shapley_log_loss_reduction",
                "figure3",
            )
            ef = {**filt, "domain": "whole"}
            row = assign(
                panel,
                "temporal",
                i,
                files[4],
                ef,
                "shapley_log_loss_reduction",
                "figure3",
            )
            assign(panel, "lower", i, files[4], ef, "joint_basic95_low", "figure3")
            assign(panel, "upper", i, files[4], ef, "joint_basic95_high", "figure3")
            assert panel["lower"][i] <= panel["temporal"][i] <= panel["upper"][i]
            assert (
                row["n"] == "1520"
                and row["family_size"] == "56"
                and row["bootstrap_reps"] == "2000"
            )
            assert (
                Decimal(row["development_oof_attribution"]) == panel["development"][i]
            )
            half = Decimal(row["joint_critical_absolute_log_loss"])
            assert abs(panel["upper"][i] - panel["temporal"][i] - half) < Decimal(
                "1e-12"
            )
            assert abs(panel["temporal"][i] - panel["lower"][i] - half) < Decimal(
                "1e-12"
            )

        checks.append(
            {
                "check": "temporal interval order and common half-width; development cross-file identity",
                "figure": "figure3",
                "panel": key,
                "n": 1520,
                "family_size": 56,
                "bootstrap_reps": 2000,
                "pass": True,
            }
        )
    return hashes


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


def main():
    global OUT
    ap = argparse.ArgumentParser(
        description="Audit Figures 2 and 3 against source DOCX tables and full-precision aggregate CSV files."
    )
    ap.add_argument("--main-docx", type=Path, default=ROOT / "source/main.docx")
    ap.add_argument("--supp-docx", type=Path, default=ROOT / "source/supp.docx")
    ap.add_argument(
        "--aggregate-dir",
        type=Path,
        default=ROOT.parent / "01_required_prior_analysis/revision_20260907/outputs",
    )
    ap.add_argument("--complete-data", type=Path)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    for label, path in [
        ("Main manuscript", args.main_docx),
        ("Supplementary material", args.supp_docx),
    ]:
        if not path.is_file():
            ap.error(
                f"{label} DOCX not found: {path.resolve()}. Supply the corresponding --main-docx or --supp-docx argument."
            )
    base = args.aggregate_dir
    required = [
        "development_budget_policies.csv",
        "nested_budget_summary.csv",
        "temporal_budget_policy_metrics.csv",
        "block_attribution_development_descriptive.csv",
        "block_attribution_evaluation.csv",
    ]
    use_aggregates = base.is_dir()
    if use_aggregates:
        absent = [
            str((base / name).resolve())
            for name in required
            if not (base / name).is_file()
        ]
        if absent:
            ap.error("Required aggregate CSV files are missing: " + ", ".join(absent))
    elif args.complete_data is None or not args.complete_data.is_file():
        ap.error(
            f"Aggregate directory not found: {base.resolve()}. Supply --aggregate-dir or an existing --complete-data file."
        )
    f2, f3, checks = extract(args.main_docx, args.supp_docx)
    aggregate_hashes = {}
    if use_aggregates:
        aggregate_hashes = integrate_aggregates(f2, f3, base, checks)
    else:
        integrate_complete(f2, f3, args.complete_data)
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    missing = {
        "figure2": {
            k: [i for i, v in enumerate(p["development"]) if v is None]
            for k, p in f2.items()
        },
        "figure3": {
            k: [BLOCKS[i] for i, v in enumerate(p["development"]) if v is None]
            for k, p in f3.items()
        },
    }
    count = sum(len(x) for f in missing.values() for x in f.values())
    evidence = {
        "sources": {
            source_label(p): sha256(p) for p in [args.main_docx, args.supp_docx]
        },
        "aggregate_sources": aggregate_hashes,
        "precision": "Plots use full-precision CSV values. Every reported DOCX numeric cell is cross-checked at four-decimal precision. No image digitization or interpolation.",
        "figure2": f2,
        "figure3": f3,
    }
    (OUT / "evidence_mapping.json").write_text(
        json.dumps(stringify(evidence), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "all_available_cross_checks_passed": True,
        "checks": checks,
        "missing_development_values": missing,
        "missing_count": count,
        "plot_generation": "blocked" if count else "complete",
    }
    (OUT / "numerical_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    schema = {
        "figure2": {
            k: {
                "development": ["NUMBER"] * 31,
                "source_file": "exact aggregate CSV filename",
                "source_column": "development OOF log-loss column",
                "source_rows": list(range(31)),
            }
            for k in f2
        },
        "figure3": {
            k: {
                "development": ["NUMBER"] * 7,
                "source_file": "exact aggregate CSV filename",
                "source_column": "development OOF average marginal contribution column",
                "source_rows": BLOCKS,
            }
            for k in f3
        },
    }
    (OUT / "required_complete_data_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    if count:
        print(
            f"BLOCKED: {count} exact development values are not reported in the supplied DOCX tables. Evidence and validation written; no approximate plots created."
        )
        return 2
    style()
    plot2(f2)
    plot3(f3)
    print(
        "Created exact-data figures 2 and 3 (PNG/SVG), evidence_mapping.json, numerical_validation.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
