#!/usr/bin/env python3


import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_exhaustive as core


def main():
    out = Path("analysis_outputs")
    fr = json.loads((out / "frozen_development_selection.json").read_text())
    test = pd.read_csv("analysis_private/evaluation.csv")
    y = test.outcome_any_use.to_numpy()
    pred = np.load("analysis_private/evaluation_predictions_private.npz")
    refit = pd.read_csv(out / "union_participant_refit_bootstrap_replicates.csv")
    rows = []
    for domain in ["extended_K", "original_noK"]:
        ss = [
            s
            for s in fr["selections"]
            if s["rule"] == "one_se" and s["domain"] == domain
        ]
        contrasts = []
        ds = []
        for s in ss:
            ck = s["algorithm"] + "::" + s["chosen_subset"]
            fk = (
                s["algorithm"]
                + "::"
                + core.name(
                    (["H"] if s["has_history"] else [])
                    + [b for b in core.OPTIONAL if domain == "extended_K" or b != "K"]
                )
            )
            contrasts.append((s, ck, fk))
            ds.append(core.loss_vector(y, pred[ck]) - core.loss_vector(y, pred[fk]))
        d = np.column_stack(ds)
        means = d.mean(0)
        rng = np.random.default_rng(core.SEED + 99)
        boot = []
        for i in range(20):
            boot.append(
                rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=100)
                @ d
                / len(y)
            )
        boot = np.vstack(boot)
        q = np.quantile(np.max(means[None, :] - boot, axis=1), 0.95)

        have_refit = all(ck in refit and fk in refit for _, ck, fk in contrasts)
        rb = (
            np.column_stack(
                [(refit[ck] - refit[fk]).to_numpy() for _, ck, fk in contrasts]
            )
            if have_refit
            else None
        )
        rq = (
            np.quantile(np.max(means[None, :] - rb, axis=1), 0.95)
            if have_refit
            else None
        )
        for j, (s, ck, fk) in enumerate(contrasts):
            r = {
                "algorithm": s["algorithm"],
                "has_history": s["has_history"],
                "domain": domain,
                "candidate": ck,
                "reference": fk,
                "delta_log_loss": means[j],
                "pointwise_lower95_percentile": np.quantile(boot[:, j], 0.025),
                "pointwise_upper95_percentile": np.quantile(boot[:, j], 0.975),
                "pointwise_upper_one_sided95_basic": means[j]
                + np.quantile(means[j] - boot[:, j], 0.95),
                "selected_family_joint_upper95_basic": means[j] + q,
                "family_contrasts": 4,
                "fixed_prediction_bootstrap_reps": len(boot),
            }
            for delta in core.DELTAS:
                r[f"fixed_prediction_joint_NI_{delta}"] = bool(
                    r["selected_family_joint_upper95_basic"] < delta
                )
            if have_refit:
                r.update(
                    union_refit_joint_upper95_basic=means[j] + rq,
                    union_refit_pointwise_upper95_basic=means[j]
                    + np.quantile(means[j] - rb[:, j], 0.95),
                    union_refit_percentile95_low=np.quantile(rb[:, j], 0.025),
                    union_refit_percentile95_high=np.quantile(rb[:, j], 0.975),
                    union_refit_reps=len(rb),
                    refit_scope="fixed frozen subset and hyperparameters; reselection uncertainty excluded",
                )
                for delta in core.DELTAS:
                    r[f"union_refit_joint_NI_{delta}"] = bool(
                        r["union_refit_joint_upper95_basic"] < delta
                    )
            rows.append(r)
    pd.DataFrame(rows).to_csv(
        out / "frozen_selected_four_family_inference.csv", index=False
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
