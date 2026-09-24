#!/usr/bin/env python3


import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import run_exhaustive as core


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--disjoint", action="store_true")
    group.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    out = Path("analysis_outputs")
    test = pd.read_csv("analysis_private/evaluation.csv")
    pp = np.load("analysis_private/evaluation_predictions_private.npz")
    prefix = ""
    if args.disjoint or args.shared:
        dev = pd.read_csv("analysis_private/development.csv")
        mask = test.id.isin(set(dev.id)).to_numpy()
        mask = ~mask if args.disjoint else mask
        test = test.loc[mask].copy()
        pp = {k: pp[k][mask] for k in pp.files}
        prefix = "disjoint_" if args.disjoint else "shared_"
    y = test.outcome_any_use.to_numpy()
    specs = core.configurations()
    rows = []
    for algo in ["ridge", "ridge_fixed", "hgb"]:
        for hist in [True, False]:
            for domain in ["extended_K", "original_noK"]:
                ss = [
                    s
                    for s in specs
                    if s["has_history"] == hist
                    and (domain == "extended_K" or "K" not in s["blocks"])
                ]
                full = algo + "::" + max(ss, key=lambda s: s["raw_items"])["subset"]
                benchmark = "P0" if hist else "prevalence"
                contrasts = []
                ds = []
                for s in ss:
                    key = algo + "::" + s["subset"]
                    for typ, ref in [("full", full), ("benchmark", benchmark)]:
                        if key == ref:
                            continue
                        contrasts.append((key, typ, ref))
                        ds.append(
                            core.loss_vector(y, pp[key]) - core.loss_vector(y, pp[ref])
                        )
                d = np.column_stack(ds)
                mean = d.mean(0)
                rng = np.random.default_rng(core.SEED + 87)
                b = []
                for k in range(20):
                    b.append(
                        rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=100)
                        @ d
                        / len(y)
                    )
                b = np.vstack(b)
                q = np.quantile(np.max(mean[None, :] - b, axis=1), 0.95)
                ci = {}
                for j, (key, typ, ref) in enumerate(contrasts):
                    ci[(key, typ)] = (mean[j], mean[j] + q)
                for s in ss:
                    key = algo + "::" + s["subset"]
                    df, uf = ci.get((key, "full"), (0.0, 0.0))
                    db, ub = ci[(key, "benchmark")]
                    rows.append(
                        {
                            "algorithm": algo,
                            "has_history": hist,
                            "domain": domain,
                            "model": key,
                            "subset": s["subset"],
                            "collection_items": (
                                s["additional_items"] if hist else s["raw_items"]
                            ),
                            "log_loss": float(core.loss_vector(y, pp[key]).mean()),
                            "full_model": full,
                            "benchmark": benchmark,
                            "delta_loss_vs_full": df,
                            "upper95_vs_full_joint": uf,
                            "delta_loss_vs_benchmark": db,
                            "upper95_vs_benchmark_joint": ub,
                            "joint_family_contrasts": len(contrasts),
                            "simultaneous_noninferior_full_margin_01": bool(uf < 0.01),
                            "simultaneous_superior_benchmark": bool(ub < 0),
                            "exploratory_dual_criterion_pass": bool(
                                uf < 0.01 and ub < 0
                            ),
                            "status": "post-evaluation hypothesis-generating; independent future validation required",
                        }
                    )
    df = pd.DataFrame(rows)
    df["evaluation_subgroup"] = prefix.strip("_") or "all"
    df["evaluation_n"] = len(test)
    df["model_variants_passing_same_subset"] = 0
    df["learner_families_passing_same_subset"] = 0
    for _, ix in df.groupby(["has_history", "domain", "subset"]).groups.items():
        count = int(df.loc[ix, "exploratory_dual_criterion_pass"].sum())
        df.loc[ix, "model_variants_passing_same_subset"] = count
        passing = df.loc[ix][df.loc[ix, "exploratory_dual_criterion_pass"]]
        df.loc[ix, "learner_families_passing_same_subset"] = passing.algorithm.replace(
            {"ridge_fixed": "ridge"}
        ).nunique()
    df.sort_values(
        ["domain", "has_history", "algorithm", "collection_items", "log_loss"]
    ).to_csv(out / (prefix + "all_subset_dual_reference_exploratory.csv"), index=False)
    shortlist = df[df.exploratory_dual_criterion_pass].sort_values(
        ["domain", "has_history", "collection_items", "log_loss"]
    )
    shortlist.to_csv(
        out / (prefix + "post_evaluation_exploratory_shortlist.csv"), index=False
    )
    print("PASS COUNTS")
    print(
        df.groupby(["algorithm", "has_history", "domain"])
        .exploratory_dual_criterion_pass.sum()
        .to_string()
    )
    print(shortlist.to_string(index=False))


if __name__ == "__main__":
    main()
