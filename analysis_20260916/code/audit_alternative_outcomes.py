#!/usr/bin/env python3

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )

import hashlib
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs/alternative_outcomes"


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    development = pd.read_csv(BASE / "private/development.csv")
    evaluation = pd.read_csv(BASE / "private/evaluation.csv")
    disjoint = ~evaluation.id.isin(set(development.id)).to_numpy()
    assert (
        len(development) == 1565 and len(evaluation) == 1520 and disjoint.sum() == 879
    )
    assert development.id.is_unique and evaluation.id.is_unique
    report = {
        "status": "complete",
        "targets": {},
        "independent_checks": [
            "raw frequency versus binary coding",
            "256 candidates and complete tuning grids",
            "budget eligibility and deterministic selection",
            "freeze receipt and input hashes",
            "OOF loss checkpoint consistency",
            "metric recomputation from private predictions",
            "direct paired-loss bootstrap reconstruction",
            "within-family duplicate contrast removal",
            "identical self-comparison zero bounds",
            "shared overall/disjoint bootstrap weights",
        ],
    }
    weights = np.random.default_rng(20260909).multinomial(
        1520, np.ones(1520) / 1520, size=2000
    )
    for target, column, positive in (
        ("monthly", "outcome_monthly_use", [1, 2, 3]),
        ("regular", "outcome_regular_use", [1, 2]),
    ):
        target_out = OUT / target
        frozen_path = target_out / "frozen_selection.json"
        frozen = json.loads(frozen_path.read_text())
        receipt = json.loads((target_out / "freeze_receipt.json").read_text())
        assert file_hash(frozen_path) == receipt["sha256"]
        assert frozen["development_sha256"] == file_hash(
            BASE / "private/development.csv"
        )
        for frame in (development, evaluation):
            assert frame.outcome_frequency.notna().all()
            assert np.array_equal(
                frame[column], frame.outcome_frequency.isin(positive).astype(int)
            )
        y = evaluation[column].to_numpy(dtype=int)
        yd = development[column].to_numpy(dtype=int)
        raw = frozen["all_models"]
        assert len(raw) == 514
        for algorithm in ("ridge", "hgb"):
            candidates = [
                row
                for row in raw
                if row["algorithm"] == algorithm and row["subset"] != "P0"
            ]
            assert (
                len(candidates) == 256
                and len({row["subset"] for row in candidates}) == 256
            )
            assert sum(not row["has_history"] for row in candidates) == 128
            for row in candidates:
                grid = row["grid_results"]
                assert len(grid) == (5 if algorithm == "ridge" else 2)
                chosen = min(
                    range(len(grid)), key=lambda index: grid[index]["oof_log_loss"]
                )
                assert row["best_config"] == grid[chosen]["config"]
                assert row["oof_log_loss"] == grid[chosen]["oof_log_loss"]
                assert all(len(item["fold_losses"]) == 5 for item in grid)
                oof = np.load(
                    BASE
                    / "private/alternative_outcomes"
                    / target
                    / "cv_oof"
                    / (algorithm + "__" + row["subset"] + ".npz")
                )["oof"]
                p = np.clip(oof, 1e-15, 1 - 1e-15)
                ll = np.mean(-yd * np.log(p) - (1 - yd) * np.log1p(-p))
                assert abs(ll - row["oof_log_loss"]) < 1e-12
        policies = frozen["policies"]
        assert len(policies) == 124
        for policy in policies:
            eligible = [
                row
                for row in raw
                if row["algorithm"] == policy["algorithm"]
                and row["additional_items"] <= policy["budget"]
                and (policy["history_available"] or not row["has_history"])
            ]
            chosen = min(
                eligible,
                key=lambda row: (
                    row["oof_log_loss"],
                    row["additional_items"],
                    row["subset"],
                ),
            )
            assert policy["selected_subset"] == chosen["subset"]
            assert policy["selected_config"] == chosen["best_config"]
            assert policy["selected_additional_items"] <= policy["budget"]
        predictions = np.load(
            BASE
            / "private/alternative_outcomes"
            / target
            / "evaluation_predictions.npz"
        )
        metrics = pd.read_csv(target_out / "all_policy_metrics_and_inference.csv")
        assert len(metrics) == 248
        assert not metrics.duplicated(["domain", "policy_id"]).any()
        errors = {
            key: 0.0
            for key in [
                "log_loss",
                "roc_auc",
                "average_precision",
                "brier",
                "calibration_intercept",
                "calibration_slope",
                "delta",
                "joint_upper",
                "paired_interval",
                "bootstrap_se",
            ]
        }
        counts = []
        for domain, mask in (
            ("whole", np.ones(1520, dtype=bool)),
            ("disjoint", disjoint),
        ):
            metric_cache = {}
            for _, row in metrics.loc[metrics.domain == domain].iterrows():
                key = row.source_model
                if key not in metric_cache:
                    p = predictions[key][mask]
                    yy = y[mask]
                    pp = np.clip(p, 1e-15, 1 - 1e-15)
                    values = {
                        "log_loss": float(
                            np.mean(-yy * np.log(pp) - (1 - yy) * np.log1p(-pp))
                        ),
                        "roc_auc": roc_auc_score(yy, p),
                        "average_precision": average_precision_score(yy, p),
                        "brier": brier_score_loss(yy, p),
                    }
                    if np.std(p) > 1e-12:
                        with threadpool_limits(limits=1):
                            calibration = LogisticRegression(
                                C=1e8, solver="lbfgs", max_iter=2000
                            ).fit(logit(np.clip(p, 1e-8, 1 - 1e-8)).reshape(-1, 1), yy)
                        values["calibration_intercept"] = calibration.intercept_[0]
                        values["calibration_slope"] = calibration.coef_[0, 0]
                    else:
                        assert pd.isna(row.calibration_slope) and pd.isna(
                            row.calibration_intercept
                        )
                    metric_cache[key] = values
                assert row.n == int(mask.sum()) and row.events == int(y[mask].sum())
                for key, expected in metric_cache[row.source_model].items():
                    errors[key] = max(errors[key], abs(expected - row[key]))
            ww = weights[:, mask].astype(float)
            ww /= ww.sum(axis=1, keepdims=True)
            for algorithm in ("ridge", "hgb"):
                for availability in ("history_available", "history_unavailable"):
                    rows = metrics.loc[
                        (metrics.domain == domain)
                        & (metrics.algorithm == algorithm)
                        & (metrics.availability == availability)
                    ]
                    assert sorted(rows.budget.tolist()) == list(range(31))
                    baseline = (
                        "P0"
                        if availability == "history_available"
                        else algorithm + "::NONE"
                    )
                    full = (
                        algorithm
                        + "::"
                        + ("H_" if availability == "history_available" else "")
                        + "D_B_M_P_C_A_K"
                    )
                    pairs = list(
                        dict.fromkeys(
                            (key, ref)
                            for key in rows.source_model
                            for ref in (full, baseline)
                            if key != ref
                        )
                    )
                    differences = []
                    for key, ref in pairs:
                        pp, rr = np.clip(
                            predictions[key][mask], 1e-15, 1 - 1e-15
                        ), np.clip(predictions[ref][mask], 1e-15, 1 - 1e-15)
                        yy = y[mask]
                        differences.append(
                            (-yy * np.log(pp) - (1 - yy) * np.log1p(-pp))
                            - (-yy * np.log(rr) - (1 - yy) * np.log1p(-rr))
                        )
                    dm = np.column_stack(differences)
                    means = dm.mean(axis=0)
                    boots = ww @ dm
                    critical = float(np.quantile(np.max(means - boots, axis=1), 0.95))
                    pair_index = {pair: index for index, pair in enumerate(pairs)}
                    for _, row in rows.iterrows():
                        assert row.family_unique_contrasts == len(pairs)
                        for label, reference in (
                            ("baseline", baseline),
                            ("full", full),
                        ):
                            if row.source_model == reference:
                                assert (
                                    row["delta_vs_" + label]
                                    == row["joint_upper95_vs_" + label]
                                    == 0
                                )
                            else:
                                index = pair_index[(row.source_model, reference)]
                                errors["delta"] = max(
                                    errors["delta"],
                                    abs(means[index] - row["delta_vs_" + label]),
                                )
                                errors["joint_upper"] = max(
                                    errors["joint_upper"],
                                    abs(
                                        means[index]
                                        + critical
                                        - row["joint_upper95_vs_" + label]
                                    ),
                                )
                                bounds = np.quantile(boots[:, index], [0.025, 0.975])
                                errors["paired_interval"] = max(
                                    errors["paired_interval"],
                                    abs(bounds[0] - row["paired_low95_vs_" + label]),
                                    abs(bounds[1] - row["paired_high95_vs_" + label]),
                                )
                                errors["bootstrap_se"] = max(
                                    errors["bootstrap_se"],
                                    abs(
                                        np.std(boots[:, index], ddof=1)
                                        - row["bootstrap_se_vs_" + label]
                                    ),
                                )
                        assert bool(row.joint_superior_to_baseline) == (
                            row.joint_upper95_vs_baseline < 0
                        )
                    counts.append(
                        {
                            "domain": domain,
                            "algorithm": algorithm,
                            "availability": availability,
                            "contrasts": len(pairs),
                            "joint_superiority_rows": int(
                                rows.joint_superior_to_baseline.sum()
                            ),
                        }
                    )
        assert max(errors.values()) < 1e-10, errors
        report["targets"][target] = {
            "n_development": 1565,
            "n_whole": 1520,
            "n_disjoint": 879,
            "events_development": int(yd.sum()),
            "events_whole": int(y.sum()),
            "events_disjoint": int(y[disjoint].sum()),
            "max_absolute_errors": errors,
            "families": counts,
        }
    output = OUT / "independent_audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run()
