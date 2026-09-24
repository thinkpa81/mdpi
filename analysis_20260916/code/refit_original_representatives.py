if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )


def main() -> None:
    import os

    for setting in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[setting] = "1"
    from pathlib import Path
    from hashlib import sha256
    import importlib.util
    import json
    import platform
    import time
    import zipfile
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import joblib
    import threadpoolctl
    from threadpoolctl import threadpool_limits
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-fit frozen representative policies; compare aggregate metrics without retuning."
    )
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    ROOT = args.package_root.resolve()
    REVIEW = ROOT / "analysis_20260916/outputs/auxiliary_metrics"
    PRIOR = ROOT / "01_required_prior_analysis"
    OUT = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "analysis_20260916/outputs"
    )
    PRIVATE = (
        args.private_dir.resolve()
        if args.private_dir
        else ROOT / "analysis_20260916/private"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    code_path = PRIOR / "code/run_exhaustive.py"
    spec = importlib.util.spec_from_file_location("frozen_original_core", code_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    start = time.time()
    dev = pd.read_csv(PRIVATE / "development.csv")
    test = pd.read_csv(PRIVATE / "evaluation.csv")
    assert len(dev) == 1565 and int(dev.outcome_any_use.sum()) == 880
    assert len(test) == 1520 and int(test.outcome_any_use.sum()) == 853
    assert dev.id.is_unique and test.id.is_unique
    disjoint = ~test.id.isin(dev.id).to_numpy()
    assert (
        int(disjoint.sum()) == 879
        and int(test.loc[disjoint, "outcome_any_use"].sum()) == 471
    )
    frozen_path = PRIOR / "revision_20260907/outputs/frozen_budget_policies.json"
    frozen = json.loads(frozen_path.read_text())
    orig_frozen = json.loads(
        (PRIOR / "analysis_outputs/frozen_development_selection.json").read_text()
    )
    orig_models = {
        r["algorithm"] + "::" + r["subset"]: r for r in orig_frozen["all_models"]
    }
    policies = [r for r in frozen["policies"] if r["budget"] in [0, 1, 4, 11, 30]]
    assert len(policies) == 20
    stored = pd.read_csv(REVIEW / "representative_policy_metrics_verified.csv")
    stored_index = stored.set_index(["policy_id", "evaluation_domain"])
    metrics_names = [
        "log_loss",
        "roc_auc",
        "average_precision",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    ]
    records, differences, prediction_cache = ([], [], {})
    for r in policies:
        key = r["source_model"]
        if key not in prediction_cache:
            if key == "P0":
                p = core.p0(dev, test)
            else:
                record = orig_models[key]
                assert record["blocks"] == r["blocks"]
                assert record["best_config"] == r["selected_config"]
                p, _ = core.fit_predict(
                    dev, test, r["blocks"], r["algorithm"], r["selected_config"]
                )
            prediction_cache[key] = p
        else:
            p = prediction_cache[key]
        for domain, mask in [
            ("all", np.ones(len(test), dtype=bool)),
            ("development_disjoint_id", disjoint),
        ]:
            with threadpool_limits(limits=1):
                values = core.metrics(
                    test.loc[mask, "outcome_any_use"].to_numpy(dtype=int), p[mask]
                )
            row = {
                "policy_id": r["policy_id"],
                "evaluation_domain": domain,
                "algorithm": r["algorithm"],
                "availability": r["availability"],
                "budget": r["budget"],
                "selected_subset": r["selected_subset"],
                "source_model": key,
                **values,
            }
            records.append(row)
            original = stored_index.loc[r["policy_id"], domain]
            diffs = {}
            for metric in metrics_names:
                if pd.isna(original[metric]) and pd.isna(values[metric]):
                    diffs[metric] = 0.0
                else:
                    assert np.isfinite(original[metric]) and np.isfinite(values[metric])
                    diffs[metric] = float(values[metric] - original[metric])
            assert (
                values["n"] == original["n"] and values["events"] == original["events"]
            )
            differences.append(
                {"policy_id": r["policy_id"], "evaluation_domain": domain, **diffs}
            )
    pd.DataFrame(records).to_csv(
        OUT / "original_representative_policy_refit_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(differences).to_csv(
        OUT / "original_representative_policy_refit_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        PRIVATE / "original_representative_policy_predictions_refit.npz",
        **prediction_cache,
    )
    max_errors = {
        m: float(max((abs(r[m]) for r in differences))) for m in metrics_names
    }
    original_environment = json.loads(
        (ROOT / "ENVIRONMENT_RECORDED.json").read_text(encoding="utf-8")
    )["recorded_execution"]
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    env_diff = {
        k: {"original": v, "rerun": environment[k]}
        for k, v in original_environment.items()
        if k in environment and v != environment[k]
    }
    check = {
        "status": (
            "PASS"
            if all((v < 1e-08 for v in max_errors.values()))
            else "DIFFERENCES_REVIEW_REQUIRED"
        ),
        "operation": "Refit only frozen representative policies on development data and evaluate fixed predictions. No hyperparameter tuning or policy reselection.",
        "policies": 20,
        "unique_prediction_models": len(prediction_cache),
        "metric_rows": 40,
        "development_n": len(dev),
        "development_events": int(dev.outcome_any_use.sum()),
        "evaluation_n": len(test),
        "evaluation_events": int(test.outcome_any_use.sum()),
        "disjoint_n": int(disjoint.sum()),
        "disjoint_events": int(test.loc[disjoint, "outcome_any_use"].sum()),
        "max_absolute_differences_from_stored": max_errors,
        "comparison_tolerance": 1e-08,
        "development_file_sha256": sha256(
            (PRIVATE / "development.csv").read_bytes()
        ).hexdigest(),
        "evaluation_file_sha256": sha256(
            (PRIVATE / "evaluation.csv").read_bytes()
        ).hexdigest(),
        "original_development_file_sha256": frozen["development_sha256"],
        "frozen_policy_sha256": sha256(frozen_path.read_bytes()).hexdigest(),
        "original_core_code_sha256": sha256(code_path.read_bytes()).hexdigest(),
        "original_environment": original_environment,
        "rerun_environment": environment,
        "environment_differences": env_diff,
        "threads": 1,
        "parallel_jobs": 1,
        "elapsed_seconds": time.time() - start,
        "scope_limit": "This validates 20 original representative policies and their auxiliary metrics; it does not rerun nested selection or all 124 policies.",
    }
    (OUT / "original_representative_policy_refit_validation.json").write_text(
        json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
