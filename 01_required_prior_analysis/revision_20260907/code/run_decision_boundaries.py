#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse, hashlib, json, os, sys
from pathlib import Path

for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
sys.path.insert(0, str(ROOT / "code"))
from run_exhaustive import loss_vector, configurations, OPTIONAL


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args):
    out = REV / "outputs"
    out.mkdir(exist_ok=True)
    predfile = ROOT / "analysis_private/evaluation_predictions_private.npz"
    datafile = ROOT / "analysis_private/evaluation.csv"
    receipt = json.loads(
        (ROOT / "analysis_outputs/evaluation_start_receipt.json").read_text()
    )
    assert sha(datafile) == receipt["evaluation_sha256"]
    test = pd.read_csv(datafile)
    dev = pd.read_csv(ROOT / "analysis_private/development.csv")
    assert test.id.is_unique and dev.id.is_unique
    ids = pd.read_csv(ROOT / "analysis_private/evaluation_ids_private.csv")
    assert test.id.astype(str).tolist() == ids.id.astype(str).tolist()
    pred = dict(np.load(predfile))
    y = test.outcome_any_use.to_numpy(dtype=int)
    assert len(y) == 1520 and y.sum() == 853
    disjoint = ~test.id.isin(set(dev.id)).to_numpy()
    assert disjoint.sum() == 879
    policy = pd.read_csv(args.policy)
    policy["history_available"] = policy.history_available.map(
        lambda x: str(x).lower() in ("true", "1")
    )
    assert len(policy) == 124
    specs = configurations()
    specmap = {s["subset"]: s for s in specs}
    rng = np.random.default_rng(args.seed)
    weights = rng.multinomial(len(y), np.ones(len(y)) / len(y), size=args.bootstrap)
    metadata = []
    candidate_rows = []
    policy_rows = []
    protocol_checks = []
    for algo in ("ridge", "hgb"):
        for available in (False, True):
            allowed = [
                algo + "::" + s["subset"]
                for s in specs
                if available or not s["has_history"]
            ]
            if available:
                allowed += ["P0"]

            assert np.array_equal(pred[algo + "::NONE"], pred["prevalence"])
            baseline = "P0" if available else algo + "::NONE"
            full = (
                algo
                + "::"
                + ("_".join(["H"] + OPTIONAL) if available else "_".join(OPTIONAL))
            )
            fam = policy[
                (policy.algorithm == algo) & (policy.history_available == available)
            ]
            assert sorted(fam.budget.tolist()) == list(range(31))
            assert all(k in allowed or k == baseline for k in fam.chosen_model)
            for _, r in fam.iterrows():
                expected = (
                    0
                    if r.chosen_model in ("P0", "prevalence")
                    else specmap[r.chosen_subset]["additional_items"]
                )
                assert int(r.selected_items) == expected and expected <= r.budget
            for domain, mask in [
                ("whole", np.ones(len(y), dtype=bool)),
                ("disjoint", disjoint),
            ]:

                keys = list(dict.fromkeys(allowed + [baseline, full]))
                index = {k: i for i, k in enumerate(keys)}
                losses = np.column_stack(
                    [loss_vector(y[mask], pred[k][mask]) for k in keys]
                )
                means = losses.mean(axis=0)
                w = weights[:, mask].astype(float)
                w /= w.sum(axis=1, keepdims=True)
                boot = w @ losses

                error = means[None, :] - boot
                q_pairs = float(
                    np.quantile(error.max(axis=1) - error.min(axis=1), 0.95)
                )
                best = int(np.argmin(means))
                contrasts = list(
                    dict.fromkeys(
                        (k, ref)
                        for k in fam.chosen_model
                        for ref in (full, baseline)
                        if k != ref
                    )
                )
                if contrasts:
                    delta = np.array(
                        [means[index[k]] - means[index[ref]] for k, ref in contrasts]
                    )
                    db = np.column_stack(
                        [
                            boot[:, index[k]] - boot[:, index[ref]]
                            for k, ref in contrasts
                        ]
                    )
                    q_policy = float(
                        np.quantile(np.max(delta[None, :] - db, axis=1), 0.95)
                    )
                else:
                    q_policy = 0.0
                metadata.append(
                    dict(
                        algorithm=algo,
                        history_available=available,
                        domain=domain,
                        n=int(mask.sum()),
                        candidates=len(keys),
                        directed_pair_contrasts=len(keys) * (len(keys) - 1),
                        policy_unique_nonzero_contrasts=len(contrasts),
                        pairwise_critical_value=q_pairs,
                        policy_critical_value=q_policy,
                        bootstrap_reps=args.bootstrap,
                        coverage="approximate 95% separately per algorithm/availability/domain and inference purpose; fixed fitted predictions",
                    )
                )
                for k in keys:
                    j = index[k]
                    subset = k.split("::")[-1]
                    cost = (
                        0
                        if k in ("P0", "prevalence")
                        else specmap[subset]["additional_items"]
                    )
                    candidate_rows.append(
                        dict(
                            algorithm=algo,
                            history_available=available,
                            domain=domain,
                            model=k,
                            subset=subset,
                            items=cost,
                            log_loss=means[j],
                            empirical_best_model=keys[best],
                            empirical_best_log_loss=means[best],
                            empirical_regret=means[j] - means[best],
                            upper95_regret_to_best_fitted_family=means[j]
                            - means[best]
                            + q_pairs,
                            status="post-evaluation descriptive audit; fixed-model simultaneous pairwise bound, not universal optimality",
                        )
                    )
                for _, r in fam.iterrows():
                    k = r.chosen_model
                    j = index[k]
                    d_full = means[j] - means[index[full]]
                    d_base = means[j] - means[index[baseline]]
                    u_full = 0.0 if k == full else d_full + q_policy
                    u_base = 0.0 if k == baseline else d_base + q_policy
                    policy_rows.append(
                        {
                            **r.to_dict(),
                            "domain": domain,
                            "n": int(mask.sum()),
                            "log_loss": means[j],
                            "full_model": full,
                            "benchmark": baseline,
                            "delta_vs_full": d_full,
                            "upper95_vs_full": u_full,
                            "delta_vs_benchmark": d_base,
                            "upper95_vs_benchmark": u_base,
                            "loss_allowance_infimum_vs_full": max(0.0, u_full),
                            "superior_to_benchmark": bool(u_base < 0.0),
                            "empirical_best_model": keys[best],
                            "empirical_regret": means[j] - means[best],
                            "upper95_regret_to_best_fitted_family": means[j]
                            - means[best]
                            + q_pairs,
                            "policy_family_unique_contrasts": len(contrasts),
                            "interpretation": "positive allowance requires delta > upper95; zero means any positive delta when upper95 <= 0",
                        }
                    )
    p = pd.DataFrame(policy_rows)
    c = pd.DataFrame(candidate_rows)
    assert len(p) == 248
    p.to_csv(out / "budget_policy_loss_boundaries.csv", index=False)
    c.to_csv(out / "all_candidate_regret_audit.csv", index=False)
    pd.DataFrame(metadata).to_csv(
        out / "loss_boundary_inference_families.csv", index=False
    )

    gates = []
    for (algo, avail, domain), g in p.groupby(
        ["algorithm", "history_available", "domain"]
    ):
        for delta in [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]:
            eligible = g[(g.upper95_vs_full < delta) & g.superior_to_benchmark]
            gates.append(
                dict(
                    algorithm=algo,
                    history_available=bool(avail),
                    domain=domain,
                    illustrative_delta=delta,
                    minimum_budget=(
                        None if eligible.empty else int(eligible.budget.min())
                    ),
                    passing_budgets=int(len(eligible)),
                    status="retrospective sensitivity; not a calibrated business tolerance",
                )
            )
    pd.DataFrame(gates).to_csv(
        out / "budget_policy_illustrative_allowances.csv", index=False
    )
    receipt_out = {
        "status": "PASS",
        "input_evaluation_sha256": sha(datafile),
        "prediction_sha256": sha(predfile),
        "policy_sha256": sha(args.policy),
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap,
        "policy_rows": len(p),
        "candidate_rows": len(c),
        "domains": {"whole": 1520, "disjoint": 879},
        "inference": "fixed-model bootstrap; per-family coverage, no across-family guarantee; not selection/refit uncertainty",
        "safeguards": [
            "history availability does not force use",
            "no evaluation-driven policy selection",
            "all budget checks passed",
            "evaluation ID alignment verified",
            "source hash verified",
        ],
    }
    (out / "decision_boundaries_receipt.json").write_text(
        json.dumps(receipt_out, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt_out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--policy", type=Path, default=REV / "outputs/development_budget_policies.csv"
    )
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260909)
    run(ap.parse_args())
