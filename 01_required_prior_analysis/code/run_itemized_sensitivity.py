#!/usr/bin/env python3


import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import run_exhaustive as core

OUT = Path("analysis_outputs")
PRIV = Path("analysis_private")


def main():
    dev = pd.read_csv(PRIV / "development.csv")
    ydev = dev.outcome_any_use.to_numpy()
    frozen = json.loads((OUT / "frozen_development_selection.json").read_text())
    original_p_numeric = core.BLOCKS["P"]["numeric"]
    original_k_numeric = core.BLOCKS["K"]["numeric"]
    try:
        core.BLOCKS["P"]["numeric"] = [f"safeguard_{k}" for k in range(1, 12)]
        core.BLOCKS["K"]["numeric"] = [
            f"digital_{k}" for k in [1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 20]
        ]
        folds = list(
            StratifiedKFold(5, shuffle=True, random_state=core.SEED).split(dev, ydev)
        )
        allr = []
        for algo in ["ridge", "hgb"]:
            for has_h in [True, False]:
                for domain in ["extended_K", "original_noK"]:
                    blocks = (["H"] if has_h else []) + [
                        b for b in core.OPTIONAL if domain == "extended_K" or b != "K"
                    ]
                    spec = {
                        "subset": core.name(blocks),
                        "blocks": blocks,
                        "has_history": has_h,
                        "domain": domain,
                    }
                    r, _, _ = core.cv_task(dev, folds, spec, algo)
                    allr.append(r)
        core.dump_json(
            OUT / "frozen_itemized_full_sensitivity.json",
            {
                "status": "post-hoc exploratory representation sensitivity",
                "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "development_sha256": core.sha(PRIV / "development.csv"),
                "models": allr,
            },
        )

        test = pd.read_csv(PRIV / "evaluation.csv")
        y = test.outcome_any_use.to_numpy()
        primary = np.load(PRIV / "evaluation_predictions_private.npz")
        predictions = {}
        rows = []
        comparisons = []
        diffs = []
        for r in allr:
            p, n = core.fit_predict(
                dev, test, r["blocks"], r["algorithm"], r["best_config"]
            )
            key = r["algorithm"] + "::itemized_" + r["subset"]
            predictions[key] = p
            rows.append(
                {
                    "model": key,
                    "algorithm": r["algorithm"],
                    "has_history": r["has_history"],
                    "domain": r["domain"],
                    "best_config": r["best_config"],
                    "development_oof_log_loss": r["oof_log_loss"],
                    "encoded_features": n,
                    "raw_items": sum(core.BLOCKS[b]["raw_items"] for b in r["blocks"]),
                    **core.metrics(y, p),
                }
            )
            chosen = next(
                s
                for s in frozen["selections"]
                if s["rule"] == "one_se"
                and s["algorithm"] == r["algorithm"]
                and s["has_history"] == r["has_history"]
                and s["domain"] == r["domain"]
            )
            ck = r["algorithm"] + "::" + chosen["chosen_subset"]
            comparisons.append(
                {
                    "candidate": ck,
                    "reference": key,
                    "algorithm": r["algorithm"],
                    "has_history": r["has_history"],
                    "domain": r["domain"],
                }
            )
            diffs.append(core.loss_vector(y, primary[ck]) - core.loss_vector(y, p))
        d = np.column_stack(diffs)
        means = d.mean(axis=0)
        rng = np.random.default_rng(core.SEED + 62)
        bs = []
        for i in range(20):
            bs.append(
                rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=100)
                @ d
                / len(y)
            )
        bs = np.vstack(bs)
        q = np.quantile(np.max(means[None, :] - bs, axis=1), 0.95)
        for j, c in enumerate(comparisons):
            c.update(
                delta_log_loss=means[j],
                paired_percentile95_low=np.quantile(bs[:, j], 0.025),
                paired_percentile95_high=np.quantile(bs[:, j], 0.975),
                simultaneous_upper95_basic=means[j] + q,
                family_contrasts=len(comparisons),
                bootstrap_reps=len(bs),
            )
        pd.DataFrame(rows).to_csv(
            OUT / "itemized_full_model_sensitivity.csv", index=False
        )
        pd.DataFrame(comparisons).to_csv(
            OUT / "selected_vs_itemized_full_sensitivity.csv", index=False
        )
        np.savez_compressed(PRIV / "itemized_predictions_private.npz", **predictions)

    finally:
        core.BLOCKS["P"]["numeric"] = original_p_numeric
        core.BLOCKS["K"]["numeric"] = original_k_numeric
    chain = {
        "P0_H": ["H"],
        "M1": ["H", "D"],
        "M2": ["H", "D", "B"],
        "M3": ["H", "D", "B", "M"],
        "M4": ["H", "D", "B", "M", "P", "C"],
        "M5": ["H", "D", "B", "M", "P", "C", "A"],
        "M6": ["H", "D", "B", "M", "P", "C", "A", "K"],
    }
    cr = []
    for label, blocks in chain.items():
        for c in core.RIDGE_GRID:
            p, n = core.fit_predict(dev, test, blocks, "ridge", c)
            cr.append(
                {
                    "legacy_model": label,
                    "subset": core.name(blocks),
                    "C": c,
                    "encoded_features": n,
                    "delta_log_loss_vs_P0": float(
                        (
                            core.loss_vector(y, p) - core.loss_vector(y, primary["P0"])
                        ).mean()
                    ),
                    **core.metrics(y, p),
                }
            )
    pd.DataFrame(cr).to_csv(OUT / "legacy_chain_common_C_sensitivity.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
