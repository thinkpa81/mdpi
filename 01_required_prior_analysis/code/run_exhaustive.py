#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse, hashlib, itertools, json, os, platform, sys, time
from pathlib import Path

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import norm
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20260907
BLOCKS = {
    "H": {"numeric": ["prior_any_use"], "categorical": [], "raw_items": 1},
    "D": {
        "numeric": [],
        "categorical": ["gender", "age_group", "education", "region"],
        "raw_items": 4,
    },
    "B": {"numeric": ["benefit_expectation"], "categorical": [], "raw_items": 1},
    "M": {"numeric": ["data_management"], "categorical": [], "raw_items": 1},
    "P": {"numeric": ["safeguard_expectation"], "categorical": [], "raw_items": 11},
    "C": {"numeric": [], "categorical": ["consent_strategy"], "raw_items": 1},
    "A": {"numeric": ["genai_experience"], "categorical": [], "raw_items": 1},
    "K": {"numeric": ["digital_competence_11"], "categorical": [], "raw_items": 11},
}
OPTIONAL = ["D", "B", "M", "P", "C", "A", "K"]
RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
HGB_GRID = [
    {"max_leaf_nodes": 3, "learning_rate": 0.05},
    {"max_leaf_nodes": 7, "learning_rate": 0.05},
]
DELTAS = [0.0025, 0.005, 0.01, 0.02]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def name(blocks):
    return "_".join(blocks) if blocks else "NONE"


def configurations():
    out = []
    for has_h in (True, False):
        for mask in range(2 ** len(OPTIONAL)):
            blocks = (["H"] if has_h else []) + [
                b for k, b in enumerate(OPTIONAL) if mask & (1 << k)
            ]
            out.append(
                {
                    "subset": name(blocks),
                    "blocks": blocks,
                    "has_history": has_h,
                    "n_blocks": len(blocks),
                    "raw_items": sum(BLOCKS[b]["raw_items"] for b in blocks),
                    "additional_items": sum(
                        BLOCKS[b]["raw_items"] for b in blocks if b != "H"
                    ),
                    "model_input_variables": sum(
                        len(BLOCKS[b]["numeric"]) + len(BLOCKS[b]["categorical"])
                        for b in blocks
                    ),
                }
            )
    assert len(out) == 256 and len({x["subset"] for x in out}) == 256
    return out


def make_model(blocks, algorithm, config, seed=SEED):
    num = sum([BLOCKS[b]["numeric"] for b in blocks], [])
    cat = sum([BLOCKS[b]["categorical"] for b in blocks], [])
    tr = []
    if num:
        tr.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        tr.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat,
            )
        )
    if algorithm == "ridge":
        model = LogisticRegression(
            C=float(config), solver="lbfgs", max_iter=2000, random_state=seed
        )
    else:
        model = HistGradientBoostingClassifier(
            **config,
            max_iter=100,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    return Pipeline([("prep", ColumnTransformer(tr)), ("model", model)])


def fit_predict(train, target, blocks, algorithm, config, seed=SEED):
    if not blocks:
        return (
            np.repeat(
                (train.outcome_any_use.sum() + 1) / (len(train) + 2), len(target)
            ),
            0,
        )
    model = make_model(blocks, algorithm, config, seed)
    with threadpool_limits(limits=1):
        model.fit(train, train.outcome_any_use.astype(int))
        p = model.predict_proba(target)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    if algorithm == "ridge":
        assert max(model.named_steps["model"].n_iter_) < 2000
    return p, model.named_steps["prep"].transform(target.iloc[:1]).shape[1]


def p0(train, target):
    probs = {
        h: (train.loc[train.prior_any_use == h, "outcome_any_use"].sum() + 1)
        / (sum(train.prior_any_use == h) + 2)
        for h in (0, 1)
    }
    return target.prior_any_use.map(probs).to_numpy(dtype=float)


def loss_vector(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def metrics(y, p, weights=None):
    out = {
        "n": len(y),
        "events": int(np.sum(y)),
        "event_rate": float(np.average(y, weights=weights)),
        "log_loss": float(np.average(loss_vector(y, p), weights=weights)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "roc_auc": (
            float(roc_auc_score(y, p, sample_weight=weights))
            if len(set(y)) == 2
            else np.nan
        ),
        "average_precision": (
            float(average_precision_score(y, p, sample_weight=weights))
            if len(set(y)) == 2
            else np.nan
        ),
    }
    if weights is None and np.std(p) > 1e-12 and len(set(y)) == 2:
        lm = LogisticRegression(C=1e8, solver="lbfgs", max_iter=2000).fit(
            logit(np.clip(p, 1e-8, 1 - 1e-8)).reshape(-1, 1), y
        )
        out.update(
            calibration_intercept=float(lm.intercept_[0]),
            calibration_slope=float(lm.coef_[0, 0]),
        )
    else:
        out.update(calibration_intercept=np.nan, calibration_slope=np.nan)
    return out


def cv_task(dev, folds, spec, algorithm):
    grid = RIDGE_GRID if algorithm == "ridge" else HGB_GRID
    candidates = []
    preds = []
    for config in grid:
        oof = np.zeros(len(dev))
        fold_losses = []
        for tr, va in folds:
            pp, _ = fit_predict(
                dev.iloc[tr], dev.iloc[va], spec["blocks"], algorithm, config
            )
            oof[va] = pp
            fold_losses.append(log_loss(dev.outcome_any_use.iloc[va], pp))
        candidates.append(
            {
                "config": config,
                "oof_log_loss": float(log_loss(dev.outcome_any_use, oof)),
                "fold_mean": float(np.mean(fold_losses)),
                "fold_se": float(np.std(fold_losses, ddof=1) / np.sqrt(len(folds))),
                "fold_losses": fold_losses,
            }
        )
        preds.append(oof)
    best = int(np.argmin([x["oof_log_loss"] for x in candidates]))
    return (
        {
            **spec,
            "algorithm": algorithm,
            "best_config": grid[best],
            "oof_log_loss": candidates[best]["oof_log_loss"],
            "fold_se": candidates[best]["fold_se"],
            "grid_results": candidates,
        },
        preds[best],
        preds[2] if algorithm == "ridge" else None,
    )


def develop(args):
    start = time.time()
    out = Path(args.out)
    private = Path(args.private)
    out.mkdir(exist_ok=True, parents=True)
    private.mkdir(exist_ok=True, parents=True)
    dev = pd.read_csv(args.development)
    y = dev.outcome_any_use.astype(int).to_numpy()
    assert dev.id.is_unique and set(y) == {0, 1}
    assert len(dev) == 1565 and y.sum() == 880
    assert set(dev.baseline_year) == {2023} and set(dev.outcome_year) == {2024}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(dev, y))
    for tr, va in folds:
        assert set(dev.id.iloc[tr]).isdisjoint(set(dev.id.iloc[va]))
    cfgs = configurations()
    raw = []
    oof = {}
    fixed = {}
    for algorithm in ("ridge", "hgb"):
        print(f"DEVELOP {algorithm}: {len(cfgs)} subsets", flush=True)
        res = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
            joblib.delayed(cv_task)(dev, folds, s, algorithm) for s in cfgs
        )
        for r, p, pfix in res:
            raw.append(r)
            oof[algorithm + "::" + r["subset"]] = p
            if pfix is not None:
                fixed["ridge_fixed::" + r["subset"]] = pfix
        print(f"DEVELOP {algorithm} done elapsed={time.time()-start:.1f}s", flush=True)
    selection = []
    for algo in ("ridge", "hgb"):
        for has_h in (True, False):
            for domain in ("extended_K", "original_noK"):
                family = [
                    r
                    for r in raw
                    if r["algorithm"] == algo
                    and r["has_history"] == has_h
                    and (domain == "extended_K" or "K" not in r["blocks"])
                ]
                best = min(family, key=lambda x: x["oof_log_loss"])
                rules = [("one_se", best["fold_se"])] + [
                    (f"delta_{d}", d) for d in DELTAS
                ]
                for rule, tol in rules:
                    eligible = [
                        r
                        for r in family
                        if r["oof_log_loss"] <= best["oof_log_loss"] + tol + 1e-12
                    ]
                    chosen = min(
                        eligible,
                        key=lambda x: (
                            x["additional_items"],
                            x["oof_log_loss"],
                            x["subset"],
                        ),
                    )
                    selection.append(
                        {
                            "algorithm": algo,
                            "has_history": has_h,
                            "domain": domain,
                            "rule": rule,
                            "best_development_subset": best["subset"],
                            "best_development_log_loss": best["oof_log_loss"],
                            "tolerance": tol,
                            "chosen_subset": chosen["subset"],
                            "chosen_config": chosen["best_config"],
                            "chosen_development_log_loss": chosen["oof_log_loss"],
                            "chosen_additional_items": chosen["additional_items"],
                        }
                    )
    frozen = {
        "seed": SEED,
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "development_sha256": sha(args.development),
        "n_development": len(dev),
        "events_development": int(y.sum()),
        "folds": 5,
        "cv": "stratified participant-disjoint development-only folds; not temporal CV",
        "evaluation_accessed_by_development": False,
        "selection_primary": "one_se then minimum additional raw-item count then OOF log loss",
        "status": "retrospective exploratory reanalysis; holdout reused from earlier research",
        "margins_status": "illustrative post-hoc sensitivity; no preregistration",
        "blocks": BLOCKS,
        "all_models": raw,
        "selections": selection,
    }
    dump_json(out / "frozen_development_selection.json", frozen)
    np.savez_compressed(private / "development_oof_predictions.npz", **oof, **fixed)
    pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ["grid_results", "blocks"]}
            for r in raw
        ]
    ).to_csv(out / "development_all_subsets.csv", index=False)
    pd.DataFrame(selection).to_csv(
        out / "development_selected_portfolios.csv", index=False
    )
    gr = []
    for r in raw:
        for g in r["grid_results"]:
            gr.append({"algorithm": r["algorithm"], "subset": r["subset"], **g})
    pd.DataFrame(gr).to_csv(out / "development_hyperparameter_grid.csv", index=False)
    pd.DataFrame(
        {
            "id": dev.id,
            "fold": [
                next(k for k, (_, va) in enumerate(folds) if i in va)
                for i in range(len(dev))
            ],
        }
    ).to_csv(private / "development_folds_private.csv", index=False)
    dump_json(
        out / "development_freeze_receipt.json",
        {
            "file": "frozen_development_selection.json",
            "sha256": sha(out / "frozen_development_selection.json"),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    print("FREEZE COMPLETE", sha(out / "frozen_development_selection.json"), flush=True)


def eval_task(dev, test, r):
    p, n = fit_predict(dev, test, r["blocks"], r["algorithm"], r["best_config"])
    fix, _ = (
        fit_predict(dev, test, r["blocks"], "ridge", 0.1)
        if r["algorithm"] == "ridge"
        else (None, None)
    )
    return r, p, n, fix


def simultaneous_bounds(y, pred, keys, reference, reps=2000, seed=SEED):
    d = np.column_stack(
        [loss_vector(y, pred[k]) - loss_vector(y, pred[reference]) for k in keys]
    )
    means = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, reps, 100):
        m = min(100, reps - start)
        weights = rng.multinomial(len(y), np.repeat(1 / len(y), len(y)), size=m) / len(
            y
        )
        boots.append(weights @ d)
    b = np.vstack(boots)

    simultaneous_q = float(np.quantile(np.max(means[None, :] - b, axis=1), 0.95))
    rows = []
    for j, k in enumerate(keys):
        se = float(np.std(b[:, j], ddof=1))
        low, high = np.quantile(b[:, j], [0.025, 0.975])
        upper_point = float(means[j] + np.quantile(means[j] - b[:, j], 0.95))
        rr = {
            "model": k,
            "reference": reference,
            "delta_log_loss": means[j],
            "paired_percentile_95_low": low,
            "paired_percentile_95_high": high,
            "pointwise_upper95_basic": upper_point,
            "simultaneous_upper95_basic": float(means[j] + simultaneous_q),
            "bootstrap_se": se,
            "approx_mde_one_sided80": float((norm.ppf(0.95) + norm.ppf(0.8)) * se),
            "family_contrasts": len(keys),
            "bootstrap_reps": reps,
        }
        for delta in DELTAS:
            rr[f"noninferior_margin_{delta}"] = bool(
                rr["simultaneous_upper95_basic"] < delta
            )
        rows.append(rr)
    return rows


def evaluate(args):
    start = time.time()
    out = Path(args.out)
    private = Path(args.private)
    frozen = json.loads((out / "frozen_development_selection.json").read_text())
    receipt = json.loads((out / "development_freeze_receipt.json").read_text())
    assert receipt["sha256"] == sha(out / "frozen_development_selection.json")
    assert frozen["development_sha256"] == sha(args.development)
    dev = pd.read_csv(args.development)
    test = pd.read_csv(args.evaluation)
    assert dev.id.is_unique and test.id.is_unique
    y = test.outcome_any_use.astype(int).to_numpy()
    assert len(test) == 1520 and y.sum() == 853
    assert set(dev.baseline_year) == {2023} and set(dev.outcome_year) == {2024}
    assert set(test.baseline_year) == {2024} and set(test.outcome_year) == {2025}
    overlap = set(dev.id) & set(test.id)
    assert len(overlap) == 641
    dump_json(
        out / "evaluation_start_receipt.json",
        {
            "evaluation_phase_read_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "frozen_sha256": receipt["sha256"],
            "evaluation_sha256": sha(args.evaluation),
        },
    )
    results = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(eval_task)(dev, test, r) for r in frozen["all_models"]
    )
    preds = {}
    meta = {}
    allrows = []
    for r, p, n, pfix in results:
        key = r["algorithm"] + "::" + r["subset"]
        preds[key] = p
        mm = {k: v for k, v in r.items() if k not in ["grid_results", "blocks"]}
        mm["encoded_features"] = n
        meta[key] = mm
        allrows.append({"model": key, **mm, **metrics(y, p)})
        if pfix is not None:
            key = "ridge_fixed::" + r["subset"]
            preds[key] = pfix
            meta[key] = {
                **mm,
                "algorithm": "ridge_fixed",
                "best_config": 0.1,
                "oof_log_loss": r["grid_results"][2]["oof_log_loss"],
                "fold_se": r["grid_results"][2]["fold_se"],
            }
            allrows.append({"model": key, **meta[key], **metrics(y, pfix)})
    preds["P0"] = p0(dev, test)
    preds["prevalence"] = np.repeat(
        (dev.outcome_any_use.sum() + 1) / (len(dev) + 2), len(test)
    )
    for k, cost in [("P0", 1), ("prevalence", 0)]:
        meta[k] = {
            "subset": k,
            "algorithm": k,
            "raw_items": cost,
            "additional_items": 0,
            "model_input_variables": cost,
            "encoded_features": cost,
        }
        allrows.append({"model": k, **meta[k], **metrics(y, preds[k])})
    pd.DataFrame(allrows).to_csv(out / "evaluation_all_subsets.csv", index=False)
    np.savez_compressed(private / "evaluation_predictions_private.npz", **preds)
    pd.DataFrame({"id": test.id, "outcome": y}).to_csv(
        private / "evaluation_ids_private.csv", index=False
    )
    infer = []
    for algorithm in ("ridge", "ridge_fixed", "hgb"):
        for has_h in (True, False):
            for domain in ("extended_K", "original_noK"):
                specs = [
                    s
                    for s in configurations()
                    if s["has_history"] == has_h
                    and (domain == "extended_K" or "K" not in s["blocks"])
                ]
                full = max(specs, key=lambda x: x["raw_items"])["subset"]
                ref = algorithm + "::" + full
                keys = [
                    algorithm + "::" + s["subset"] for s in specs if s["subset"] != full
                ]
                rr = simultaneous_bounds(y, preds, keys, ref, args.bootstrap)
                for r in rr:
                    r.update(
                        algorithm=algorithm,
                        has_history=has_h,
                        domain=domain,
                        family=f"{algorithm}|history={has_h}|{domain}",
                    )
                infer.extend(rr)
    pd.DataFrame(infer).to_csv(
        out / "all_subset_noninferiority_exploratory.csv", index=False
    )

    selected = [s for s in frozen["selections"] if s["rule"] == "one_se"]
    selected_keys = set(["P0", "prevalence"])
    selected_compare = []
    for s in selected:
        ck = s["algorithm"] + "::" + s["chosen_subset"]
        fk = (
            s["algorithm"]
            + "::"
            + name(
                (["H"] if s["has_history"] else [])
                + [b for b in OPTIONAL if s["domain"] == "extended_K" or b != "K"]
            )
        )
        selected_keys |= {ck, fk}
        refrows = [r for r in infer if r["model"] == ck and r["reference"] == fk]
        base = (
            refrows[0]
            if refrows
            else {
                "delta_log_loss": 0.0,
                "pointwise_upper95_basic": 0.0,
                "simultaneous_upper95_basic": 0.0,
                "paired_percentile_95_low": 0.0,
                "paired_percentile_95_high": 0.0,
            }
        )
        selected_compare.append(
            {
                **s,
                "model": ck,
                "reference": fk,
                **base,
                "selected_evaluation_log_loss": metrics(y, preds[ck])["log_loss"],
                "full_evaluation_log_loss": metrics(y, preds[fk])["log_loss"],
            }
        )
    pd.DataFrame(selected_compare).to_csv(
        out / "selected_portfolios_evaluation.csv", index=False
    )
    subgroups = {
        "all": np.ones(len(test), dtype=bool),
        "development_shared_id": test.id.isin(overlap).to_numpy(),
        "development_disjoint_id": ~test.id.isin(overlap).to_numpy(),
        "prior_nonuser": (test.prior_any_use == 0).to_numpy(),
        "prior_user": (test.prior_any_use == 1).to_numpy(),
    }
    for value in sorted(test.age_group.dropna().unique()):
        subgroups["age_group_" + str(value)] = (test.age_group == value).to_numpy()
    subrows = []
    subci = []
    for label, mask in subgroups.items():
        for k in sorted(selected_keys):
            subrows.append(
                {"subgroup": label, "model": k, **metrics(y[mask], preds[k][mask])}
            )
        if label in [
            "all",
            "development_shared_id",
            "development_disjoint_id",
            "prior_nonuser",
            "prior_user",
        ]:
            for s in selected:
                ck = s["algorithm"] + "::" + s["chosen_subset"]
                fk = (
                    s["algorithm"]
                    + "::"
                    + name(
                        (["H"] if s["has_history"] else [])
                        + [
                            b
                            for b in OPTIONAL
                            if s["domain"] == "extended_K" or b != "K"
                        ]
                    )
                )
                if ck == fk:
                    continue
                rr = simultaneous_bounds(
                    y[mask],
                    {k: v[mask] for k, v in preds.items()},
                    [ck],
                    fk,
                    args.bootstrap,
                )
                for r in rr:
                    r.update(
                        subgroup=label,
                        domain=s["domain"],
                        has_history=s["has_history"],
                        algorithm=s["algorithm"],
                    )
                subci.extend(rr)
    pd.DataFrame(subrows).to_csv(out / "selected_subgroup_metrics.csv", index=False)
    pd.DataFrame(subci).to_csv(
        out / "selected_subgroup_paired_intervals.csv", index=False
    )

    frontier = []
    for has_h in (True, False):
        for algorithm in ("ridge", "ridge_fixed", "hgb"):
            for domain in ("extended_K", "original_noK"):
                kk = [
                    algorithm + "::" + s["subset"]
                    for s in configurations()
                    if s["has_history"] == has_h
                    and (domain == "extended_K" or "K" not in s["blocks"])
                ] + (["P0", "prevalence"] if has_h else ["prevalence"])
                for k in kk:
                    cost = (
                        meta[k]["additional_items"] if has_h else meta[k]["raw_items"]
                    )
                    ll = float(loss_vector(y, preds[k]).mean())
                    other = [
                        (
                            (
                                meta[j]["additional_items"]
                                if has_h
                                else meta[j]["raw_items"]
                            ),
                            float(loss_vector(y, preds[j]).mean()),
                        )
                        for j in kk
                    ]
                    dominated = any(
                        (c <= cost and l <= ll) and (c < cost or l < ll)
                        for c, l in other
                    )
                    frontier.append(
                        {
                            "algorithm": algorithm,
                            "has_history": has_h,
                            "domain": domain,
                            "model": k,
                            "collection_items": cost,
                            "log_loss": ll,
                            "pareto_nondominated": not dominated,
                        }
                    )
    pd.DataFrame(frontier).to_csv(
        out / "cost_frontier_with_benchmarks_descriptive.csv", index=False
    )
    dca = []
    for threshold in np.arange(0.1, 0.91, 0.05):
        for k in sorted(selected_keys):
            decision = preds[k] >= threshold
            nb = (
                np.sum(decision & (y == 1))
                - np.sum(decision & (y == 0)) * threshold / (1 - threshold)
            ) / len(y)
            dca.append(
                {
                    "threshold": threshold,
                    "model": k,
                    "net_benefit": nb,
                    "scope": "hypothetical classification utility; not intervention effectiveness",
                }
            )
        dca.extend(
            [
                {"threshold": threshold, "model": "act_none", "net_benefit": 0.0},
                {
                    "threshold": threshold,
                    "model": "act_all",
                    "net_benefit": float(
                        y.mean() - (1 - y.mean()) * threshold / (1 - threshold)
                    ),
                },
            ]
        )
    pd.DataFrame(dca).to_csv(out / "decision_curve_hypothetical.csv", index=False)
    transitions = []
    for period, frame in [("development", dev), ("evaluation", test)]:
        for h in (0, 1):
            for outcome in (0, 1):
                transitions.append(
                    {
                        "period": period,
                        "prior_use": h,
                        "next_use": outcome,
                        "n": int(
                            (
                                (frame.prior_any_use == h)
                                & (frame.outcome_any_use == outcome)
                            ).sum()
                        ),
                    }
                )
    pd.DataFrame(transitions).to_csv(out / "use_transition_counts.csv", index=False)
    weighted = []
    if "outcome_long_weight" in test:
        w = test.outcome_long_weight.to_numpy()
        valid = np.isfinite(w) & (w > 0)
        for k in sorted(selected_keys):
            weighted.append(
                {
                    "model": k,
                    "estimand": "longitudinal-weighted selected analytic subset, not general population",
                    **metrics(y[valid], preds[k][valid], w[valid]),
                }
            )
        pd.DataFrame(weighted).to_csv(
            out / "selected_weighted_sensitivity.csv", index=False
        )
    make_figures(out, allrows, frontier, selected_compare, preds, y, dca)
    dump_json(
        out / "evaluation_summary.json",
        {
            "development_n": len(dev),
            "evaluation_n": len(test),
            "overlap_n": len(overlap),
            "disjoint_n": len(test) - len(overlap),
            "overlap_fraction": len(overlap) / len(test),
            "candidate_subsets": 256,
            "learners": ["ridge tuned", "ridge fixed C=.1", "HGB tuned"],
            "all_models_with_benchmarks": len(preds),
            "bootstrap_reps": args.bootstrap,
            "inference_scope": "Each algorithm x history regime x full-domain has separate simultaneous family (127 or 63 comparisons). No cross-family multiplicity guarantee.",
            "fixed_prediction_bootstrap_scope": "Evaluation sampling conditional on fitted models and development selection; union-participant refit bootstrap supplied separately.",
            "elapsed_seconds": time.time() - start,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "sklearn": sklearn.__version__,
            },
        },
    )
    print(
        "EVALUATION COMPLETE",
        len(preds),
        "models elapsed",
        time.time() - start,
        flush=True,
    )


def make_figures(out, allrows, frontier, selected, preds, y, dca):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"ridge": "#006D77", "hgb": "#B45309"}
    for ax, has_h in zip(axs, [True, False]):
        for algo in ("ridge", "hgb"):
            rr = [
                r
                for r in frontier
                if r["algorithm"] == algo
                and r["has_history"] == has_h
                and r["domain"] == "extended_K"
            ]
            ax.scatter(
                [r["collection_items"] for r in rr],
                [r["log_loss"] for r in rr],
                s=15,
                alpha=0.4,
                color=colors[algo],
                label=algo.upper(),
            )
            pp = sorted(
                [r for r in rr if r["pareto_nondominated"]],
                key=lambda r: r["collection_items"],
            )
            ax.plot(
                [r["collection_items"] for r in pp],
                [r["log_loss"] for r in pp],
                color=colors[algo],
                lw=1.4,
            )
            ss = [
                r
                for r in selected
                if r["algorithm"] == algo
                and r["has_history"] == has_h
                and r["domain"] == "extended_K"
            ][0]
            cost = next(r["collection_items"] for r in rr if r["model"] == ss["model"])
            ax.scatter(
                [cost],
                [ss["selected_evaluation_log_loss"]],
                marker="*",
                s=150,
                color=colors[algo],
                edgecolor="black",
                zorder=5,
            )
        ax.axhline(
            loss_vector(y, preds["prevalence"]).mean(),
            color="gray",
            ls=":",
            label="Prevalence",
        )
        if has_h:
            ax.axhline(
                loss_vector(y, preds["P0"]).mean(), color="black", ls="--", label="P0"
            )
        ax.set(
            title="Prior-use history available" if has_h else "No prior-use history",
            xlabel="Additional raw survey items" if has_h else "Raw survey items",
            ylabel="Temporal evaluation log loss",
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Declared-block exhaustive audit: descriptive test frontier\nStars: portfolios frozen using development data only",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "figure_cost_frontier.png", bbox_inches="tight")
    fig.savefig(out / "figure_cost_frontier.pdf", bbox_inches="tight")
    plt.close(fig)
    primary = [r for r in selected if r["domain"] == "extended_K"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(primary):
        x = r["delta_log_loss"]
        lo = r["paired_percentile_95_low"]
        hi = r["paired_percentile_95_high"]
        ax.errorbar(
            x,
            i,
            xerr=[[max(0, x - lo)], [max(0, hi - x)]],
            fmt="o",
            color=colors[r["algorithm"]],
            capsize=4,
        )
        ax.plot(r["simultaneous_upper95_basic"], i, ">", color=colors[r["algorithm"]])
    ax.axvline(0, color="gray", lw=1)
    ax.axvline(0.01, color="red", ls="--", label="Illustrative margin 0.01")
    ax.set_yticks(
        range(len(primary)),
        [f"{r['algorithm'].upper()}: {r['chosen_subset']}" for r in primary],
    )
    ax.set_xlabel("Log loss: selected portfolio minus full information")
    ax.set_title(
        "Exploratory noninferiority of development-selected portfolios\nBars: paired 95% intervals; triangles: simultaneous upper 95% bound"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figure_selected_noninferiority.png", bbox_inches="tight")
    plt.close(fig)


def refit_task(rep, dev, test, configs, union):
    rng = np.random.default_rng(SEED + 100000 + rep)
    count = rng.multinomial(len(union), np.repeat(1 / len(union), len(union)))
    mp = dict(zip(union, count))
    di = np.repeat(np.arange(len(dev)), [mp[i] for i in dev.id])
    ti = np.repeat(np.arange(len(test)), [mp[i] for i in test.id])
    dd = dev.iloc[di]
    yy = test.outcome_any_use.to_numpy()[ti]
    result = {"replicate": rep, "dev_boot_n": len(di), "eval_boot_n": len(ti)}
    for k, c in configs.items():
        pp = (
            p0(dd, test)
            if k == "P0"
            else fit_predict(
                dd, test, c["blocks"], c["algorithm"], c["best_config"], SEED + rep
            )[0]
        )
        result[k] = float(loss_vector(yy, pp[ti]).mean())
    return result


def refit(args):
    out = Path(args.out)
    dev = pd.read_csv(args.development)
    test = pd.read_csv(args.evaluation)
    fr = json.loads((out / "frozen_development_selection.json").read_text())
    allmodels = {r["algorithm"] + "::" + r["subset"]: r for r in fr["all_models"]}
    configs = {"P0": None}
    pairs = []
    for s in fr["selections"]:
        if s["rule"] != "one_se" or (
            s["algorithm"] == "hgb" and s["domain"] != "extended_K"
        ):
            continue
        ck = s["algorithm"] + "::" + s["chosen_subset"]
        fk = (
            s["algorithm"]
            + "::"
            + name(
                (["H"] if s["has_history"] else [])
                + [b for b in OPTIONAL if s["domain"] == "extended_K" or b != "K"]
            )
        )
        configs[ck] = allmodels[ck]
        configs[fk] = allmodels[fk]
        pairs.append(
            {
                "candidate": ck,
                "reference": fk,
                "domain": s["domain"],
                "has_history": s["has_history"],
            }
        )
        if s["has_history"]:
            pairs.append(
                {
                    "candidate": ck,
                    "reference": "P0",
                    "domain": s["domain"],
                    "has_history": True,
                }
            )
    union = np.union1d(dev.id, test.id)
    print(
        "REFIT bootstrap", args.refits, "replicates", len(configs), "models", flush=True
    )
    rows = joblib.Parallel(n_jobs=args.jobs, verbose=5)(
        joblib.delayed(refit_task)(b, dev, test, configs, union)
        for b in range(args.refits)
    )
    rr = pd.DataFrame(rows)
    rr.to_csv(out / "union_participant_refit_bootstrap_replicates.csv", index=False)
    preds = np.load(Path(args.private) / "evaluation_predictions_private.npz")
    y = test.outcome_any_use.to_numpy()
    summ = []
    for p in pairs:
        k, r = p["candidate"], p["reference"]
        vals = rr[k] - rr[r]
        lo, hi = np.quantile(vals, [0.025, 0.975])
        point = float((loss_vector(y, preds[k]) - loss_vector(y, preds[r])).mean())
        summ.append(
            {
                **p,
                "delta_log_loss": point,
                "refit_percentile95_low": lo,
                "refit_percentile95_high": hi,
                "refit_upper95_percentile": float(np.quantile(vals, 0.95)),
                "refit_se": float(vals.std()),
                "replicates": len(vals),
                "union_unique_participants": len(union),
                "selection_uncertainty": "conditional on frozen subset and hyperparameters; selection not repeated",
            }
        )
    pd.DataFrame(summ).to_csv(
        out / "union_participant_refit_intervals.csv", index=False
    )
    print("REFIT COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["develop", "evaluate", "refit"])
    p.add_argument("--development", default="analysis_private/development.csv")
    p.add_argument("--evaluation", default="analysis_private/evaluation.csv")
    p.add_argument("--out", default="analysis_outputs")
    p.add_argument("--private", default="analysis_private")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--refits", type=int, default=500)
    a = p.parse_args()
    {"develop": develop, "evaluate": evaluate, "refit": refit}[a.phase](a)


if __name__ == "__main__":
    main()
