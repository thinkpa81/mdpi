#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for task_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[task_env] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import run_exhaustive as original
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs"
PRIVATE = BASE / "private"
ALGORITHMS = ("ridge", "hgb")
AVAILABILITIES = ("history_available", "history_unavailable")
BUDGETS = list(range(31))
OUTER_SEEDS = (202609070, 202609071)
INNER_BASE_SEED = 202609170


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def pid(algorithm, availability, budget):
    return f"{algorithm}|{availability}|budget={budget:02d}"


def source_key(row):
    return "P0" if row["subset"] == "P0" else row["algorithm"] + "::" + row["subset"]


def choose(raw, algorithm, availability, budget):
    eligible = [
        r
        for r in raw
        if r["algorithm"] == algorithm
        and r["additional_items"] <= budget
        and (availability == "history_available" or not r["has_history"])
    ]
    assert eligible
    return min(
        eligible, key=lambda r: (r["oof_log_loss"], r["additional_items"], r["subset"])
    )


def policies(raw):
    result = []
    for algorithm in ALGORITHMS:
        for availability in AVAILABILITIES:
            for budget in BUDGETS:
                r = choose(raw, algorithm, availability, budget)
                result.append(
                    {
                        "policy_id": pid(algorithm, availability, budget),
                        "algorithm": algorithm,
                        "availability": availability,
                        "budget": budget,
                        "history_available": availability == "history_available",
                        "selected_subset": r["subset"],
                        "source_model": source_key(r),
                        "chosen_model": source_key(r),
                        "chosen_subset": r["subset"],
                        "selected_config": r["best_config"],
                        "blocks": r["blocks"],
                        "chosen_config": json.dumps(r["best_config"], sort_keys=True),
                        "uses_history": bool(r["has_history"]),
                        "selected_additional_items": int(r["additional_items"]),
                        "selected_items": int(r["additional_items"]),
                        "development_oof_log_loss": float(r["oof_log_loss"]),
                    }
                )
    assert len(result) == 124
    assert all(
        not p["uses_history"]
        for p in result
        if p["availability"] == "history_unavailable"
    )
    assert all(p["selected_additional_items"] <= p["budget"] for p in result)
    assert all(p["selected_additional_items"] == 0 for p in result if p["budget"] == 0)
    for algorithm in ALGORITHMS:
        for availability in AVAILABILITIES:
            vals = [
                p["development_oof_log_loss"]
                for p in result
                if p["algorithm"] == algorithm and p["availability"] == availability
            ]
            assert np.all(np.diff(vals) <= 1e-12)
    return result


def p0_candidate(dev, folds, algorithm):
    pred = np.zeros(len(dev))
    for tr, va in folds:
        pred[va] = original.p0(dev.iloc[tr], dev.iloc[va])
    return {
        "algorithm": algorithm,
        "subset": "P0",
        "blocks": ["H"],
        "additional_items": 0,
        "has_history": True,
        "best_config": None,
        "oof_log_loss": float(
            original.loss_vector(dev.outcome_any_use.to_numpy(), pred).mean()
        ),
    }, pred


def scalar_policy_table(rows):
    return pd.DataFrame(
        [
            {
                **{
                    k: v for k, v in r.items() if k not in ("blocks", "selected_config")
                },
                "selected_config_json": json.dumps(
                    r["selected_config"], sort_keys=True
                ),
                "blocks_json": json.dumps(r["blocks"]),
            }
            for r in rows
        ]
    )


def development(args):
    start = time.time()
    OUT.mkdir(exist_ok=True, parents=True)
    PRIVATE.mkdir(exist_ok=True, parents=True)
    dev_path = ROOT / "analysis_private/development.csv"
    dev = pd.read_csv(dev_path)
    assert (
        dev.id.is_unique and len(dev) == 1565 and int(dev.outcome_any_use.sum()) == 880
    )
    assert set(dev.baseline_year) == {2023} and set(dev.outcome_year) == {2024}
    frozen_path = ROOT / "analysis_outputs/frozen_development_selection.json"
    previous = json.loads(frozen_path.read_text())
    assert previous["development_sha256"] == sha(dev_path)
    previous_receipt = json.loads(
        (ROOT / "analysis_outputs/development_freeze_receipt.json").read_text()
    )
    assert previous_receipt["sha256"] == sha(frozen_path)
    raw = [dict(r) for r in previous["all_models"]]
    y = dev.outcome_any_use.astype(int).to_numpy()
    previous_folds = list(
        StratifiedKFold(5, shuffle=True, random_state=original.SEED).split(dev, y)
    )
    p0_original = None
    for algo in ALGORITHMS:
        row, p0_original = p0_candidate(dev, previous_folds, algo)
        raw.append(row)

    dev_cache_path = ROOT / "analysis_private/development_oof_predictions.npz"
    dev_cache = np.load(dev_cache_path)
    for r in raw:
        pred = p0_original if r["subset"] == "P0" else dev_cache[source_key(r)]
        assert np.isclose(
            original.loss_vector(y, pred).mean(), r["oof_log_loss"], atol=1e-12, rtol=0
        )
    fixed = policies(raw)
    frozen = {
        "status": "retrospective reanalysis; evaluation outcomes previously examined",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "evaluation_accessed_by_development": False,
        "development_sha256": sha(dev_path),
        "original_development_freeze_sha256": sha(frozen_path),
        "original_development_oof_sha256": sha(dev_cache_path),
        "protocol_sha256": sha(BASE / "budget_policy_protocol.md"),
        "code_sha256": sha(__file__),
        "original_model_code_sha256": sha(ROOT / "code/run_exhaustive.py"),
        "selection_rule": "minimum inner OOF log loss, then actual additional items, then subset lexicographic",
        "availability": {
            "history_available": "all 256 subsets plus P0; H may be ignored",
            "history_unavailable": "128 subsets excluding H, including NONE",
        },
        "budgets": BUDGETS,
        "outer_seeds": OUTER_SEEDS,
        "inner_folds": 5,
        "policies": fixed,
    }
    dump(OUT / "frozen_budget_policies.json", frozen)
    dump(
        OUT / "budget_policy_freeze_receipt.json",
        {
            "sha256": sha(OUT / "frozen_budget_policies.json"),
            "created_utc": frozen["created_utc"],
        },
    )
    scalar_policy_table(fixed).to_csv(
        OUT / "development_budget_policies.csv", index=False
    )
    np.savez_compressed(
        PRIVATE / "development_original_policy_oof.npz",
        **{
            r["policy_id"]: (
                p0_original
                if r["source_model"] == "P0"
                else dev_cache[r["source_model"]]
            )
            for r in fixed
        },
    )
    print("BUDGET POLICIES FROZEN: 124 policies; no evaluation data opened", flush=True)
    specs = original.configurations()
    all_selected, all_fold_metrics = [], []
    all_oof = {
        pid(a, h, b): np.full((2, len(dev)), np.nan)
        for a in ALGORITHMS
        for h in AVAILABILITIES
        for b in BUDGETS
    }
    all_oof["benchmark_P0"] = np.full((2, len(dev)), np.nan)
    all_oof["benchmark_NONE"] = np.full((2, len(dev)), np.nan)
    assignment = []
    for repeat, outer_seed in enumerate(OUTER_SEEDS):
        outer_folds = list(
            StratifiedKFold(5, shuffle=True, random_state=outer_seed).split(dev, y)
        )
        for outer_fold, (train_idx, val_idx) in enumerate(outer_folds):
            outer_train, outer_valid = dev.iloc[train_idx].reset_index(
                drop=True
            ), dev.iloc[val_idx].reset_index(drop=True)
            assert set(outer_train.id).isdisjoint(set(outer_valid.id))
            inner_seed = INNER_BASE_SEED + repeat * 10 + outer_fold
            inner_folds = list(
                StratifiedKFold(5, shuffle=True, random_state=inner_seed).split(
                    outer_train, outer_train.outcome_any_use
                )
            )
            for tr, va in inner_folds:
                assert set(outer_train.id.iloc[tr]).isdisjoint(
                    set(outer_train.id.iloc[va])
                )
            fold_raw = []
            for algo in ALGORITHMS:
                print(
                    f"NESTED repeat={repeat+1}/2 outer={outer_fold+1}/5 algorithm={algo} start elapsed={time.time()-start:.1f}s",
                    flush=True,
                )
                cv_result = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                    joblib.delayed(original.cv_task)(outer_train, inner_folds, s, algo)
                    for s in specs
                )
                fold_raw.extend(r for r, _, _ in cv_result)
                del cv_result
                p0row, _ = p0_candidate(outer_train, inner_folds, algo)
                fold_raw.append(p0row)

            dump(
                OUT / f"nested_inner_grid_repeat{repeat+1}_fold{outer_fold+1}.json",
                fold_raw,
            )
            selected = policies(fold_raw)
            fit_cache = {}
            for p in selected:
                cache_key = (
                    p["source_model"]
                    + "|"
                    + json.dumps(p["selected_config"], sort_keys=True)
                )
                if cache_key not in fit_cache:
                    if p["source_model"] == "P0":
                        pp = original.p0(outer_train, outer_valid)
                    else:
                        pp, _ = original.fit_predict(
                            outer_train,
                            outer_valid,
                            p["blocks"],
                            p["algorithm"],
                            p["selected_config"],
                        )
                    fit_cache[cache_key] = pp
                pred = fit_cache[cache_key]
                all_oof[p["policy_id"]][repeat, val_idx] = pred
                all_selected.append(
                    {
                        **p,
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold + 1,
                        "outer_training_n": len(train_idx),
                        "outer_validation_n": len(val_idx),
                    }
                )
                all_fold_metrics.append(
                    {
                        "policy_id": p["policy_id"],
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold + 1,
                        "algorithm": p["algorithm"],
                        "availability": p["availability"],
                        "budget": p["budget"],
                        "log_loss": float(
                            original.loss_vector(y[val_idx], pred).mean()
                        ),
                        "n": len(val_idx),
                    }
                )
            all_oof["benchmark_P0"][repeat, val_idx] = original.p0(
                outer_train, outer_valid
            )
            all_oof["benchmark_NONE"][repeat, val_idx] = (
                outer_train.outcome_any_use.sum() + 1
            ) / (len(outer_train) + 2)
            assignment.extend(
                {
                    "id": str(dev.id.iloc[i]),
                    "repeat": repeat + 1,
                    "outer_fold": outer_fold + 1,
                }
                for i in val_idx
            )
            print(
                f"NESTED repeat={repeat+1}/2 outer={outer_fold+1}/5 done; unique fitted policies={len(fit_cache)} elapsed={time.time()-start:.1f}s",
                flush=True,
            )
    assert all(np.isfinite(p).all() for p in all_oof.values())
    np.savez_compressed(PRIVATE / "nested_policy_oof_predictions.npz", **all_oof)
    pd.DataFrame(assignment).to_csv(
        PRIVATE / "nested_outer_assignments.csv", index=False
    )
    scalar_policy_table(all_selected).to_csv(
        OUT / "nested_selected_policies.csv", index=False
    )
    pd.DataFrame(all_fold_metrics).to_csv(
        OUT / "nested_outer_fold_metrics.csv", index=False
    )
    repeat_rows = []
    for p in fixed:
        for repeat in range(2):
            repeat_rows.append(
                {
                    "policy_id": p["policy_id"],
                    "algorithm": p["algorithm"],
                    "availability": p["availability"],
                    "budget": p["budget"],
                    "repeat": repeat + 1,
                    **original.metrics(y, all_oof[p["policy_id"]][repeat]),
                }
            )
    rep = pd.DataFrame(repeat_rows)
    rep.to_csv(OUT / "nested_repeat_metrics.csv", index=False)
    summary = []
    for p in fixed:
        rr = rep.loc[rep.policy_id == p["policy_id"]]
        choices = [r for r in all_selected if r["policy_id"] == p["policy_id"]]
        counts = pd.Series([r["selected_subset"] for r in choices]).value_counts()
        benchmark = (
            "benchmark_P0"
            if p["availability"] == "history_available"
            else "benchmark_NONE"
        )
        bench_ll = [
            original.loss_vector(y, all_oof[benchmark][r]).mean() for r in range(2)
        ]
        summary.append(
            {
                **{
                    k: p[k]
                    for k in [
                        "policy_id",
                        "algorithm",
                        "availability",
                        "budget",
                        "selected_subset",
                        "selected_additional_items",
                        "development_oof_log_loss",
                    ]
                },
                "nested_log_loss_mean": float(rr.log_loss.mean()),
                "nested_log_loss_repeat_min": float(rr.log_loss.min()),
                "nested_log_loss_repeat_max": float(rr.log_loss.max()),
                "nested_minus_non_nested_log_loss": float(
                    rr.log_loss.mean() - p["development_oof_log_loss"]
                ),
                "nested_benchmark_log_loss_mean": float(np.mean(bench_ll)),
                "nested_delta_benchmark": float(rr.log_loss.mean() - np.mean(bench_ll)),
                "nested_roc_auc_mean": float(rr.roc_auc.mean()),
                "nested_brier_mean": float(rr.brier.mean()),
                "selection_unique_subsets": len(counts),
                "selection_modal_subset": str(counts.index[0]),
                "selection_modal_count_out_of10": int(counts.iloc[0]),
                "selection_matches_final_count_out_of10": sum(
                    r["selected_subset"] == p["selected_subset"] for r in choices
                ),
                "selection_uses_history_count_out_of10": sum(
                    r["uses_history"] for r in choices
                ),
                "selection_actual_items_mean": float(
                    np.mean([r["selected_additional_items"] for r in choices])
                ),
            }
        )
    pd.DataFrame(summary).to_csv(OUT / "nested_budget_summary.csv", index=False)
    frequency = []
    for p in fixed:
        choices = [r for r in all_selected if r["policy_id"] == p["policy_id"]]
        for subset, count in (
            pd.Series([r["selected_subset"] for r in choices]).value_counts().items()
        ):
            frequency.append(
                {
                    "policy_id": p["policy_id"],
                    "algorithm": p["algorithm"],
                    "availability": p["availability"],
                    "budget": p["budget"],
                    "subset": subset,
                    "count": int(count),
                    "denominator": 10,
                }
            )
    pd.DataFrame(frequency).to_csv(
        OUT / "nested_selection_frequencies.csv", index=False
    )
    dump(
        OUT / "budget_development_completion.json",
        {
            "status": "complete",
            "seconds": time.time() - start,
            "outer_folds": 10,
            "evaluation_accessed": False,
            "policies": 124,
            "development_n": len(dev),
            "frozen_budget_sha256": sha(OUT / "frozen_budget_policies.json"),
            "inference_note": "Two repeated partitions do not double sample size. Selection frequencies are descriptive.",
        },
    )
    print(f"DEVELOPMENT COMPLETE elapsed={time.time()-start:.1f}s", flush=True)


def evaluation(args):
    start = time.time()
    frozen_path = OUT / "frozen_budget_policies.json"
    frozen = json.loads(frozen_path.read_text())
    receipt = json.loads((OUT / "budget_policy_freeze_receipt.json").read_text())
    assert receipt["sha256"] == sha(frozen_path)
    dev_path = ROOT / "analysis_private/development.csv"
    eval_path = ROOT / "analysis_private/evaluation.csv"
    assert frozen["development_sha256"] == sha(dev_path)
    prior_receipt = json.loads(
        (ROOT / "analysis_outputs/evaluation_start_receipt.json").read_text()
    )
    assert prior_receipt["evaluation_sha256"] == sha(eval_path)
    assert (
        prior_receipt["frozen_sha256"] == frozen["original_development_freeze_sha256"]
    )
    dev, test = pd.read_csv(dev_path), pd.read_csv(eval_path)
    ids = pd.read_csv(ROOT / "analysis_private/evaluation_ids_private.csv")
    assert np.array_equal(ids.id, test.id) and np.array_equal(
        ids.outcome, test.outcome_any_use
    )
    assert (
        len(test) == 1520
        and int(test.outcome_any_use.sum()) == 853
        and test.id.is_unique
    )
    assert set(test.baseline_year) == {2024} and set(test.outcome_year) == {2025}
    cache_path = ROOT / "analysis_private/evaluation_predictions_private.npz"
    cache = np.load(cache_path)
    y = test.outcome_any_use.astype(int).to_numpy()
    fixed = frozen["policies"]
    unique = {p["source_model"]: p for p in fixed}
    checks = []
    for key, p in unique.items():
        if key == "P0":
            pred = original.p0(dev, test)
        else:
            pred, _ = original.fit_predict(
                dev, test, p["blocks"], p["algorithm"], p["selected_config"]
            )
        error = float(np.max(np.abs(pred - cache[key])))
        assert error < 1e-12, (key, error)
        checks.append(
            {"source_model": key, "recomputed_max_abs_prediction_difference": error}
        )
    private_predictions = {p["policy_id"]: cache[p["source_model"]] for p in fixed}
    private_predictions["P0"] = cache["P0"]
    private_predictions["prevalence"] = cache["prevalence"]
    np.savez_compressed(
        PRIVATE / "temporal_budget_policy_predictions.npz", **private_predictions
    )
    pd.DataFrame({"id": test.id, "outcome": y}).to_csv(
        PRIVATE / "temporal_policy_ids.csv", index=False
    )
    overlap = set(dev.id) & set(test.id)
    assert len(overlap) == 641
    masks = {
        "all": np.ones(len(test), dtype=bool),
        "development_disjoint_id": ~test.id.isin(overlap).to_numpy(),
        "development_shared_id": test.id.isin(overlap).to_numpy(),
    }
    rows = []
    for domain, mask in masks.items():
        for p in fixed:
            full_subset = (
                "H_D_B_M_P_C_A_K"
                if p["availability"] == "history_available"
                else "D_B_M_P_C_A_K"
            )
            fullkey = p["algorithm"] + "::" + full_subset
            basekey = "P0" if p["availability"] == "history_available" else "prevalence"
            budget30 = next(
                q
                for q in fixed
                if q["algorithm"] == p["algorithm"]
                and q["availability"] == p["availability"]
                and q["budget"] == 30
            )
            pred = cache[p["source_model"]][mask]
            loss = original.loss_vector(y[mask], pred).mean()
            row = {
                **{
                    k: p[k]
                    for k in [
                        "policy_id",
                        "algorithm",
                        "availability",
                        "budget",
                        "selected_subset",
                        "source_model",
                        "uses_history",
                        "selected_additional_items",
                        "development_oof_log_loss",
                    ]
                },
                "evaluation_domain": domain,
                **original.metrics(y[mask], pred),
                "full_model": fullkey,
                "baseline_model": basekey,
                "unrestricted_policy_source": budget30["source_model"],
                "delta_full_log_loss": float(
                    loss - original.loss_vector(y[mask], cache[fullkey][mask]).mean()
                ),
                "delta_baseline_log_loss": float(
                    loss - original.loss_vector(y[mask], cache[basekey][mask]).mean()
                ),
                "delta_budget30_policy_log_loss": float(
                    loss
                    - original.loss_vector(
                        y[mask], cache[budget30["source_model"]][mask]
                    ).mean()
                ),
            }
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "temporal_budget_policy_metrics.csv", index=False)
    dump(
        OUT / "budget_evaluation_provenance.json",
        {
            "status": "complete",
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "status_of_evaluation": "previously examined 2025 outcomes; retrospective exploratory reanalysis",
            "frozen_budget_policy_sha256": sha(frozen_path),
            "development_sha256": sha(dev_path),
            "evaluation_sha256": sha(eval_path),
            "original_evaluation_prediction_cache_sha256": sha(cache_path),
            "original_evaluation_receipt_sha256": sha(
                ROOT / "analysis_outputs/evaluation_start_receipt.json"
            ),
            "checks": checks,
            "all_cache_checks_passed": True,
            "n_unique_policy_models": len(unique),
            "seconds": time.time() - start,
        },
    )
    nested = pd.read_csv(OUT / "nested_budget_summary.csv")
    memo = [
        "# 예산별 정책 분석 요약",
        "",
        "모든 신규 정책은 후향적 재분석이다. 예산별 최종 조합은 개발 OOF만으로 선택되었다.",
        "",
    ]
    for algo in ALGORITHMS:
        for availability in AVAILABILITIES:
            domain = result[
                (result.algorithm == algo)
                & (result.availability == availability)
                & (result.evaluation_domain == "all")
            ]
            memo.append(f"## {algo} / {availability}")
            for budget in (0, 1, 4, 8, 11, 15, 20, 30):
                r = domain[domain.budget == budget].iloc[0]
                n = nested[nested.policy_id == r.policy_id].iloc[0]
                memo.append(
                    f"- 예산 {budget}: {r.selected_subset}, 실제 추가 {int(r.selected_additional_items)}문항, 개발 OOF {r.development_oof_log_loss:.6f}, 중첩 {n.nested_log_loss_mean:.6f}, 시간순 {r.log_loss:.6f}, 기준선 차이 {r.delta_baseline_log_loss:+.6f}; 최빈 조합 {n.selection_modal_count_out_of10}/10."
                )
            memo.append("")
    (OUT / "budget_policy_quick_summary.md").write_text(
        "\n".join(memo), encoding="utf-8"
    )
    print(
        f"EVALUATION COMPLETE unique cached policies verified={len(unique)} elapsed={time.time()-start:.1f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["develop", "evaluate"])
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    if args.phase == "develop":
        development(args)
    else:
        evaluation(args)


if __name__ == "__main__":
    main()
