#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

for task_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[task_env] = "1"

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parents[1]
TARGETS = {
    "monthly": ("outcome_monthly_use", [1, 2, 3]),
    "regular": ("outcome_regular_use", [1, 2]),
}
REPRESENTATIVE_BUDGETS = [0, 1, 4, 11, 30]
BOOT_SEED = 20260909
REPS = 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def import_sources(source_root):
    sys.path.insert(0, str(source_root / "code"))
    import run_exhaustive as original

    spec = importlib.util.spec_from_file_location(
        "budget_sensitivity_source",
        source_root / "revision_20260907/code/run_budget_policy.py",
    )
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    return original, budget


def recode(frame, target):
    column, positive_codes = TARGETS[target]
    result = frame.copy()
    expected = result.outcome_frequency.isin(positive_codes).astype(int)
    assert np.array_equal(result[column].to_numpy(), expected.to_numpy())
    assert result[column].notna().all() and set(result[column].unique()) == {0, 1}
    result["outcome_any_use"] = result[column].astype(int)
    assert np.array_equal(result.prior_any_use, frame.prior_any_use)
    return result


def checked_cv_task(
    dev, folds, spec, algorithm, checkpoint, cache, source_root, context_hash
):

    original, _ = import_sources(source_root)
    tag = algorithm + "__" + spec["subset"]
    summary_path, pred_path = checkpoint / (tag + ".json"), cache / (tag + ".npz")
    if summary_path.exists() and pred_path.exists():
        cached = json.loads(summary_path.read_text())
        if cached.get("context_sha256") == context_hash:
            pp = np.load(pred_path)["oof"]
            assert np.isclose(
                original.loss_vector(dev.outcome_any_use.to_numpy(), pp).mean(),
                cached["result"]["oof_log_loss"],
                rtol=0,
                atol=1e-12,
            )
            return cached["result"]
    with threadpool_limits(limits=1):
        result, pred, _ = original.cv_task(dev, folds, spec, algorithm)
    np.savez_compressed(pred_path, oof=pred)
    dump(summary_path, {"context_sha256": context_hash, "result": result})
    return result


def fit_eval_task(dev, eva, row, source_root):
    original, _ = import_sources(source_root)
    with threadpool_limits(limits=1):
        if row["subset"] == "P0":
            pred, width = original.p0(dev, eva), 1
        else:
            pred, width = original.fit_predict(
                dev, eva, row["blocks"], row["algorithm"], row["best_config"]
            )
    return (
        ("P0" if row["subset"] == "P0" else row["algorithm"] + "::" + row["subset"]),
        pred,
        width,
    )


def infer_family(original, y, predictions, family, available, mask, weights):
    algorithm = family[0]["algorithm"]
    baseline = "P0" if available else algorithm + "::NONE"
    full = algorithm + "::" + ("H_" if available else "") + "D_B_M_P_C_A_K"
    keys = list(dict.fromkeys([p["source_model"] for p in family] + [baseline, full]))
    ix = {key: i for i, key in enumerate(keys)}
    losses = np.column_stack(
        [original.loss_vector(y[mask], predictions[key][mask]) for key in keys]
    )
    means = losses.mean(axis=0)
    ww = weights[:, mask].astype(float)
    ww /= ww.sum(axis=1, keepdims=True)
    boot = ww @ losses
    contrasts = list(
        dict.fromkeys(
            (p["source_model"], ref)
            for p in family
            for ref in (full, baseline)
            if p["source_model"] != ref
        )
    )
    if contrasts:
        delta = np.array([means[ix[k]] - means[ix[r]] for k, r in contrasts])
        db = np.column_stack([boot[:, ix[k]] - boot[:, ix[r]] for k, r in contrasts])
        q = float(np.quantile(np.max(delta[None, :] - db, axis=1), 0.95))
    else:
        q = 0.0
    rows = []
    for p in family:
        k = p["source_model"]
        result = {
            "policy_id": p["policy_id"],
            "family_unique_contrasts": len(contrasts),
            "joint_critical_value": q,
        }
        for label, ref in (("baseline", baseline), ("full", full)):
            delta = float(means[ix[k]] - means[ix[ref]])
            if k == ref:
                low = high = upper = se = 0.0
                assert delta == 0.0
            else:
                bd = boot[:, ix[k]] - boot[:, ix[ref]]
                low, high = np.quantile(bd, [0.025, 0.975])
                upper = delta + q
                se = np.std(bd, ddof=1)
            result.update(
                {
                    f"delta_vs_{label}": delta,
                    f"joint_upper95_vs_{label}": float(upper),
                    f"paired_low95_vs_{label}": float(low),
                    f"paired_high95_vs_{label}": float(high),
                    f"bootstrap_se_vs_{label}": float(se),
                }
            )
        result["joint_superior_to_baseline"] = result["joint_upper95_vs_baseline"] < 0
        rows.append(result)
    return rows, {
        "n": int(mask.sum()),
        "unique_policy_models": len(set(p["source_model"] for p in family)),
        "family_unique_contrasts": len(contrasts),
        "joint_critical_value": q,
        "bootstrap_reps": REPS,
        "bootstrap_seed": BOOT_SEED,
        "baseline_model": baseline,
        "full_model": full,
        "scope": "separate target/algorithm/history/domain fixed-prediction family",
    }


def run_target(args, target, original, budget):
    start = time.time()
    out = BASE / "outputs/alternative_outcomes" / target
    private = BASE / "private/alternative_outcomes" / target
    checkpoint, cv_cache = out / "cv_checkpoints", private / "cv_oof"
    for directory in (out, private, checkpoint, cv_cache):
        directory.mkdir(parents=True, exist_ok=True)
    dev_path, eva_path = (
        BASE / "private/development.csv",
        BASE / "private/evaluation.csv",
    )
    protocol_path = BASE / "alternative_outcome_protocol.md"
    assert protocol_path.exists()
    dev = recode(pd.read_csv(dev_path), target)
    assert len(dev) == 1565 and dev.id.is_unique
    y_dev = dev.outcome_any_use.to_numpy(dtype=int)
    folds = list(
        StratifiedKFold(5, shuffle=True, random_state=original.SEED).split(dev, y_dev)
    )
    assert all(
        set(dev.id.iloc[tr]).isdisjoint(set(dev.id.iloc[va])) for tr, va in folds
    )
    fold_assignment = np.zeros(len(dev), dtype=int)
    for fold, (_, va) in enumerate(folds):
        fold_assignment[va] = fold
    np.save(private / "development_fold_assignment.npy", fold_assignment)
    context = {
        "target": target,
        "target_column": TARGETS[target][0],
        "positive_codes": TARGETS[target][1],
        "development_sha256": sha(dev_path),
        "protocol_sha256": sha(protocol_path),
        "code_sha256": sha(__file__),
        "source_model_sha256": sha(args.source_root / "code/run_exhaustive.py"),
        "source_policy_sha256": sha(
            args.source_root / "revision_20260907/code/run_budget_policy.py"
        ),
        "seed": original.SEED,
        "ridge_grid": original.RIDGE_GRID,
        "hgb_grid": original.HGB_GRID,
    }
    context_hash = hashlib.sha256(
        json.dumps(context, sort_keys=True).encode()
    ).hexdigest()
    dump(
        out / "started_before_fitting.json",
        {
            **context,
            "start_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "status": "post hoc sensitivity; evaluation previously inspected; this selection phase receives development only",
            "development_n": len(dev),
            "development_events": int(y_dev.sum()),
            "jobs": args.jobs,
        },
    )
    raw = []
    for algorithm in ("ridge", "hgb"):
        print(
            f"{target}: CV {algorithm} 256 candidates started; elapsed={time.time()-start:.1f}s",
            flush=True,
        )
        results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(checked_cv_task)(
                dev,
                folds,
                spec,
                algorithm,
                checkpoint,
                cv_cache,
                args.source_root,
                context_hash,
            )
            for spec in original.configurations()
        )
        raw.extend(results)
        p0_row, p0_oof = budget.p0_candidate(dev, folds, algorithm)
        raw.append(p0_row)
        np.savez_compressed(private / (algorithm + "__P0_oof.npz"), oof=p0_oof)
        print(
            f"{target}: CV {algorithm} finished; elapsed={time.time()-start:.1f}s",
            flush=True,
        )
    policies = budget.policies(raw)
    frozen = {
        **context,
        "frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "evaluation_accessed_by_selection_function": False,
        "development_n": len(dev),
        "development_events": int(y_dev.sum()),
        "all_models": raw,
        "policies": policies,
    }
    frozen_path = out / "frozen_selection.json"
    dump(frozen_path, frozen)
    dump(
        out / "freeze_receipt.json",
        {"sha256": sha(frozen_path), "utc": frozen["frozen_utc"]},
    )
    budget.scalar_policy_table(policies).to_csv(
        out / "development_policies.csv", index=False
    )
    grid_rows = [
        {
            "target": target,
            "algorithm": r["algorithm"],
            "subset": r["subset"],
            "config_json": json.dumps(g["config"], sort_keys=True),
            "oof_log_loss": g["oof_log_loss"],
            "fold_mean": g["fold_mean"],
            "fold_se": g["fold_se"],
        }
        for r in raw
        if "grid_results" in r
        for g in r["grid_results"]
    ]
    assert len(grid_rows) == 256 * 7
    pd.DataFrame(grid_rows).to_csv(
        out / "development_hyperparameter_grid.csv", index=False
    )

    eva = recode(pd.read_csv(eva_path), target)
    assert len(eva) == 1520 and eva.id.is_unique
    assert set(dev.baseline_year) == {2023} and set(eva.baseline_year) == {2024}
    assert set(dev.outcome_year) == {2024} and set(eva.outcome_year) == {2025}
    disjoint = ~eva.id.isin(set(dev.id)).to_numpy()
    assert disjoint.sum() == 879
    y = eva.outcome_any_use.to_numpy(dtype=int)
    keys = set(p["source_model"] for p in policies)
    keys.update(["P0", "ridge::NONE", "hgb::NONE"])
    keys.update(
        a + "::" + h + "D_B_M_P_C_A_K" for a in ("ridge", "hgb") for h in ("", "H_")
    )
    selected_raw = []
    seen = set()
    for r in raw:
        key = "P0" if r["subset"] == "P0" else r["algorithm"] + "::" + r["subset"]
        if key in keys and key not in seen:
            selected_raw.append(r)
            seen.add(key)
    assert seen == keys
    fitted = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(fit_eval_task)(dev, eva, r, args.source_root)
        for r in selected_raw
    )
    preds = {key: pred for key, pred, _ in fitted}
    np.savez_compressed(private / "evaluation_predictions.npz", **preds)
    assert np.array_equal(preds["ridge::NONE"], preds["hgb::NONE"])
    weights = np.random.default_rng(BOOT_SEED).multinomial(
        len(y), np.ones(len(y)) / len(y), size=REPS
    )
    rows, reference_rows, infer_rows, family_rows = [], [], [], []
    for domain, mask in (
        ("whole", np.ones(len(y), dtype=bool)),
        ("disjoint", disjoint),
    ):
        for algorithm in ("ridge", "hgb"):
            for availability in ("history_available", "history_unavailable"):
                available = availability == "history_available"
                family = [
                    p
                    for p in policies
                    if p["algorithm"] == algorithm and p["availability"] == availability
                ]
                rr, fm = infer_family(
                    original, y, preds, family, available, mask, weights
                )
                fm.update(
                    target=target,
                    algorithm=algorithm,
                    availability=availability,
                    domain=domain,
                )
                family_rows.append(fm)
                infer_rows.extend(
                    {
                        **r,
                        "target": target,
                        "algorithm": algorithm,
                        "availability": availability,
                        "domain": domain,
                    }
                    for r in rr
                )
                for p in family:
                    with threadpool_limits(limits=1):
                        met = original.metrics(y[mask], preds[p["source_model"]][mask])
                    rows.append(
                        {
                            "target": target,
                            "domain": domain,
                            **{
                                k: p[k]
                                for k in (
                                    "policy_id",
                                    "algorithm",
                                    "availability",
                                    "budget",
                                    "selected_subset",
                                    "source_model",
                                    "uses_history",
                                    "selected_additional_items",
                                    "development_oof_log_loss",
                                )
                            },
                            **met,
                        }
                    )
                for role, key in (
                    ("baseline", fm["baseline_model"]),
                    ("full", fm["full_model"]),
                ):
                    with threadpool_limits(limits=1):
                        met = original.metrics(y[mask], preds[key][mask])
                    reference_rows.append(
                        {
                            "target": target,
                            "domain": domain,
                            "algorithm": algorithm,
                            "availability": availability,
                            "role": role,
                            "source_model": key,
                            **met,
                        }
                    )
    result = pd.DataFrame(rows)
    inference = pd.DataFrame(infer_rows)
    result = result.merge(
        inference,
        on=["target", "domain", "policy_id", "algorithm", "availability"],
        validate="one_to_one",
    )
    assert len(result) == 248 and len(family_rows) == 8
    assert (
        result[["log_loss", "roc_auc", "average_precision", "brier"]]
        .notna()
        .all()
        .all()
    )
    result.to_csv(out / "all_policy_metrics_and_inference.csv", index=False)
    result[result.budget.isin(REPRESENTATIVE_BUDGETS)].to_csv(
        out / "representative_policy_metrics.csv", index=False
    )
    pd.DataFrame(reference_rows).to_csv(
        out / "reference_model_metrics.csv", index=False
    )
    pd.DataFrame(family_rows).to_csv(out / "inference_families.csv", index=False)
    summary = {
        "target": target,
        "status": "complete",
        "elapsed_seconds": time.time() - start,
        "development_n": len(dev),
        "development_events": int(y_dev.sum()),
        "evaluation_n": len(eva),
        "evaluation_events": int(y.sum()),
        "disjoint_n": int(disjoint.sum()),
        "disjoint_events": int(y[disjoint].sum()),
        "candidates_per_algorithm": 256,
        "grid_configurations": len(grid_rows),
        "folds": 5,
        "candidate_fold_fits": 5 * len(grid_rows),
        "budgets": 31,
        "policy_metric_rows": len(result),
        "representative_rows": int(result.budget.isin(REPRESENTATIVE_BUDGETS).sum()),
        "evaluation_fitted_models": len(fitted),
        "joint_superiority_rows": int(result.joint_superior_to_baseline.sum()),
        "frozen_selection_sha256": sha(frozen_path),
        "development_sha256": sha(dev_path),
        "evaluation_sha256": sha(eva_path),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "limits": [
            "post hoc threshold sensitivity, not measurement-invariance identification",
            "no repeated nested CV rerun",
            "fixed-prediction bootstrap, no refit/selection uncertainty",
            "separate family coverage, no across-target guarantee",
            "constant-prediction calibration slope undefined",
        ],
    }
    dump(out / "completion_manifest.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return result, pd.DataFrame(reference_rows), pd.DataFrame(family_rows), summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--target", choices=["all", *TARGETS], default="all")
    ap.add_argument(
        "--source-root", type=Path, default=BASE.parent / "01_required_prior_analysis"
    )
    args = ap.parse_args()
    args.source_root = args.source_root.resolve()
    required_inputs = (
        args.source_root / "code/run_exhaustive.py",
        args.source_root / "revision_20260907/code/run_budget_policy.py",
        BASE / "alternative_outcome_protocol.md",
        BASE / "private/development.csv",
        BASE / "private/evaluation.csv",
    )
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "Required analysis inputs are missing:\n" + "\n".join(missing_inputs)
        )
    original, budget = import_sources(args.source_root)
    targets = list(TARGETS) if args.target == "all" else [args.target]
    records = [run_target(args, target, original, budget) for target in targets]
    out = BASE / "outputs/alternative_outcomes"
    allrows = pd.concat([r[0] for r in records], ignore_index=True)
    allrows.to_csv(out / "all_targets_policy_metrics.csv", index=False)
    allrows[allrows.budget.isin(REPRESENTATIVE_BUDGETS)].to_csv(
        out / "all_targets_representative_metrics.csv", index=False
    )
    pd.concat([r[1] for r in records], ignore_index=True).to_csv(
        out / "all_targets_reference_metrics.csv", index=False
    )
    pd.concat([r[2] for r in records], ignore_index=True).to_csv(
        out / "all_targets_inference_families.csv", index=False
    )
    dump(
        out / "completion_manifest.json",
        {"status": "complete", "targets": [r[3] for r in records]},
    )


if __name__ == "__main__":
    main()
