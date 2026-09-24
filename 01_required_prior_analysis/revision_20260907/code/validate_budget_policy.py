#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
OUT = BASE / "outputs"
PRIVATE = BASE / "private"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def main() -> None:
    checks = []

    def require(name, condition, details=""):
        checks.append({"check": name, "passed": bool(condition), "details": details})
        if not condition:
            raise AssertionError(name)

    dev = pd.read_csv(ROOT / "analysis_private/development.csv", dtype={"id": str})
    test = pd.read_csv(ROOT / "analysis_private/evaluation.csv", dtype={"id": str})
    frozen = json.loads((OUT / "frozen_budget_policies.json").read_text())
    receipt = json.loads((OUT / "budget_policy_freeze_receipt.json").read_text())
    require(
        "frozen_policy_receipt",
        receipt["sha256"] == digest(OUT / "frozen_budget_policies.json"),
    )
    require(
        "executed_code_matches_current",
        frozen["code_sha256"] == digest(BASE / "code/run_budget_policy.py"),
    )
    require(
        "protocol_matches_execution",
        frozen["protocol_sha256"] == digest(BASE / "budget_policy_protocol.md"),
    )
    require(
        "original_model_code_unchanged",
        frozen["original_model_code_sha256"] == digest(ROOT / "code/run_exhaustive.py"),
    )
    require(
        "development_input_unchanged",
        frozen["development_sha256"]
        == digest(ROOT / "analysis_private/development.csv"),
    )
    policies = pd.read_csv(OUT / "development_budget_policies.csv")
    require(
        "124_complete_unique_policies",
        len(policies) == 124 and policies.policy_id.nunique() == 124,
    )
    require(
        "complete_budget_grid",
        all(
            (
                set(g.budget) == set(range(31))
                for _, g in policies.groupby(["algorithm", "availability"])
            )
        ),
    )
    require(
        "selected_costs_feasible",
        bool((policies.selected_items <= policies.budget).all()),
    )
    require(
        "budget_zero_no_collection",
        bool((policies.loc[policies.budget == 0, "selected_items"] == 0).all()),
    )
    require(
        "unavailable_history_never_used",
        not policies.loc[
            policies.availability == "history_unavailable", "uses_history"
        ].any(),
    )
    require(
        "nested_feasible_set_minimum_monotone",
        all(
            (
                np.all(
                    np.diff(g.sort_values("budget").development_oof_log_loss) <= 1e-12
                )
                for _, g in policies.groupby(["algorithm", "availability"])
            )
        ),
    )
    assign = pd.read_csv(PRIVATE / "nested_outer_assignments.csv", dtype={"id": str})
    require(
        "original_string_ids_preserved",
        set(assign.id) == set(dev.id),
        "ID strings with underscores must remain unchanged.",
    )
    require(
        "one_outer_prediction_per_participant_per_repeat",
        len(assign) == 2 * len(dev) and (not assign.duplicated(["repeat", "id"]).any()),
    )
    require(
        "five_outer_folds_each_repeat",
        all(
            (set(g.outer_fold) == {1, 2, 3, 4, 5} for _, g in assign.groupby("repeat"))
        ),
    )
    selection = pd.read_csv(OUT / "nested_selected_policies.csv")
    require(
        "ten_nested_selections_each_policy",
        len(selection) == 1240 and set(selection.groupby("policy_id").size()) == {10},
    )
    require(
        "all_nested_choices_budget_feasible",
        bool((selection.selected_items <= selection.budget).all()),
    )
    require(
        "nested_unavailable_history_never_used",
        not selection.loc[
            selection.availability == "history_unavailable", "uses_history"
        ].any(),
    )
    outer = np.load(PRIVATE / "nested_policy_oof_predictions.npz")
    require(
        "nested_predictions_complete_and_finite",
        all(
            (
                outer[k].shape == (2, len(dev)) and np.isfinite(outer[k]).all()
                for k in outer.files
            )
        ),
    )
    repeat = pd.read_csv(OUT / "nested_repeat_metrics.csv")
    error = []
    y = dev.outcome_any_use.to_numpy()
    for _, r in repeat.iterrows():
        error.append(
            abs(mean_loss(y, outer[r.policy_id][int(r["repeat"]) - 1]) - r.log_loss)
        )
    require(
        "nested_metrics_recompute_from_individual_predictions",
        max(error) < 1e-12,
        f"max error={max(error):.3g}",
    )
    summary = pd.read_csv(OUT / "nested_budget_summary.csv")
    err = max(
        (
            abs(
                float(repeat.loc[repeat.policy_id == r.policy_id, "log_loss"].mean())
                - r.nested_log_loss_mean
            )
            for _, r in summary.iterrows()
        )
    )
    require(
        "repeat_mean_is_mean_loss_not_ensemble_loss",
        err < 1e-12,
        f"max error={err:.3g}",
    )
    temporal = pd.read_csv(OUT / "temporal_budget_policy_metrics.csv")
    pred = np.load(PRIVATE / "temporal_budget_policy_predictions.npz")
    ids = pd.read_csv(PRIVATE / "temporal_policy_ids.csv", dtype={"id": str})
    require(
        "temporal_ids_and_outcomes_in_original_order",
        np.array_equal(ids.id, test.id)
        and np.array_equal(ids.outcome, test.outcome_any_use),
    )
    shared = test.id.isin(set(dev.id)).to_numpy()
    require(
        "participant_overlap_matches_source",
        int(shared.sum()) == 641 and int((~shared).sum()) == 879,
    )
    mask = {
        "all": np.ones(len(test), dtype=bool),
        "development_shared_id": shared,
        "development_disjoint_id": ~shared,
    }
    errors = []
    y = test.outcome_any_use.to_numpy()
    for _, r in temporal.iterrows():
        m = mask[r.evaluation_domain]
        errors.append(abs(mean_loss(y[m], pred[r.policy_id][m]) - r.log_loss))
    require(
        "temporal_metrics_recompute_all_domains",
        max(errors) < 1e-12,
        f"max error={max(errors):.3g}",
    )
    provenance = json.loads((OUT / "budget_evaluation_provenance.json").read_text())
    require(
        "source_cache_verified_against_refitted_models",
        provenance["all_cache_checks_passed"]
        and all(
            (
                r["recomputed_max_abs_prediction_difference"] < 1e-12
                for r in provenance["checks"]
            )
        ),
    )
    require(
        "source_cache_unchanged",
        provenance["original_evaluation_prediction_cache_sha256"]
        == digest(ROOT / "analysis_private/evaluation_predictions_private.npz"),
    )
    require(
        "temporal_evaluation_input_unchanged",
        provenance["evaluation_sha256"]
        == digest(ROOT / "analysis_private/evaluation.csv"),
    )
    report = {
        "status": "PASS",
        "checks_passed": len(checks),
        "checks": checks,
        "scope": "Numerical, identity, provenance, and budget consistency checks; not a guarantee of external validity or acceptance.",
    }
    (OUT / "budget_policy_validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "checks_passed": len(checks)}))


if __name__ == "__main__":
    main()
