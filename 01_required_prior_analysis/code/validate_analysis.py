#!/usr/bin/env python3

import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_exhaustive as c

out = Path("analysis_outputs")
pr = Path("analysis_private")
checks = []


def gate(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append({"check": name, "status": "PASS"})


def main():
    d = pd.read_csv(pr / "development.csv")
    e = pd.read_csv(pr / "evaluation.csv")
    f = json.loads((out / "frozen_development_selection.json").read_text())
    r = json.loads((out / "development_freeze_receipt.json").read_text())
    start = json.loads((out / "evaluation_start_receipt.json").read_text())
    gate(
        "chronology development 2023 to 2024; evaluation 2024 to 2025",
        set(d.baseline_year) == {2023}
        and set(d.outcome_year) == {2024}
        and set(e.baseline_year) == {2024}
        and set(e.outcome_year) == {2025},
    )
    gate(
        "participant uniqueness within each transition",
        d.id.is_unique and e.id.is_unique,
    )
    gate(
        "sample counts and target events",
        len(d) == 1565
        and int(d.outcome_any_use.sum()) == 880
        and len(e) == 1520
        and int(e.outcome_any_use.sum()) == 853,
    )
    gate(
        "shared/disjoint participants",
        len(set(d.id) & set(e.id)) == 641
        and len(set(e.id) - set(d.id)) == 879
        and len(set(d.id) | set(e.id)) == 2444,
    )
    gate(
        "immutable development file matches freeze",
        f["development_sha256"] == c.sha(pr / "development.csv"),
    )
    gate(
        "frozen configuration hash matches receipt",
        r["sha256"]
        == c.sha(out / "frozen_development_selection.json")
        == start["frozen_sha256"],
    )
    gate(
        "evaluation file hash matches receipt",
        start["evaluation_sha256"] == c.sha(pr / "evaluation.csv"),
    )
    gate(
        "freeze receipt precedes evaluation phase read",
        pd.Timestamp(r["created_utc"])
        <= pd.Timestamp(start["evaluation_phase_read_utc"]),
    )
    specs = c.configurations()
    gate(
        "256 unique block combinations including intercept only",
        len(specs) == 256
        and len({s["subset"] for s in specs}) == 256
        and "NONE" in {s["subset"] for s in specs},
    )
    gate(
        "history regimes each contain 128 combinations",
        sum(s["has_history"] for s in specs) == 128,
    )
    gate(
        "no outcome or identifying field enters model predictors",
        not any(
            x.startswith("outcome") or x in ["id", "baseline_year", "outcome_year"]
            for b in c.BLOCKS.values()
            for x in b["numeric"] + b["categorical"]
        ),
    )
    fold = pd.read_csv(pr / "development_folds_private.csv")
    gate(
        "each development participant occurs in one OOF validation fold",
        fold.id.is_unique
        and set(fold.id) == set(d.id)
        and set(fold.fold) == set(range(5)),
    )
    a = pd.read_csv(out / "evaluation_all_subsets.csv")
    p = np.load(pr / "evaluation_predictions_private.npz")
    gate(
        "770 complete probability vectors including benchmarks",
        len(a) == len(p.files) == 770
        and all(
            p[k].shape == (1520,)
            and np.isfinite(p[k]).all()
            and ((p[k] > 0) & (p[k] < 1)).all()
            for k in p.files
        ),
    )
    gate(
        "all reported log losses match participant predictions",
        all(
            abs(
                float(row.log_loss)
                - np.mean(c.loss_vector(e.outcome_any_use.to_numpy(), p[row.model]))
            )
            < 1e-12
            for _, row in a.iterrows()
        ),
    )
    selected = pd.read_csv(out / "selected_portfolios_evaluation.csv")
    gate(
        "eight one-SE choices use frozen selections",
        len(selected) == 8
        and all(
            any(
                s["rule"] == "one_se"
                and s["algorithm"] == row.algorithm
                and s["has_history"] == row.has_history
                and s["domain"] == row.domain
                and s["chosen_subset"] == row.chosen_subset
                for s in f["selections"]
            )
            for _, row in selected.iterrows()
        ),
    )
    gate(
        "selected contrasts equal raw log-loss subtraction",
        np.allclose(
            selected.delta_log_loss,
            selected.selected_evaluation_log_loss - selected.full_evaluation_log_loss,
            atol=1e-12,
        ),
    )
    inf = pd.read_csv(out / "all_subset_noninferiority_exploratory.csv")
    gate(
        "all intended exhaustive contrast families complete",
        len(inf) == 1140 and set(inf.family_contrasts) == {63, 127},
    )
    fi = pd.read_csv(out / "frozen_selected_four_family_inference.csv")
    gate(
        "primary four and secondary four contrast families separate",
        len(fi) == 8 and set(fi.family_contrasts) == {4},
    )
    rr = pd.read_csv(out / "union_participant_refit_bootstrap_replicates.csv")
    gate(
        "500 paired union-participant refit replicates",
        len(rr) == 500
        and rr.replicate.is_unique
        and rr.drop(columns=["replicate"]).notna().all().all(),
    )
    gate(
        "all four expanded frozen choices have refit inference",
        fi.loc[fi.domain == "extended_K", "union_refit_joint_upper95_basic"]
        .notna()
        .all(),
    )
    sh = pd.read_csv(out / "post_evaluation_exploratory_shortlist.csv")
    gate(
        "all shortlist rows jointly satisfy both criteria",
        ((sh.upper95_vs_full_joint < 0.01) & (sh.upper95_vs_benchmark_joint < 0)).all(),
    )
    gate(
        "fixed ridge is not counted as an independent learner",
        sh.learner_families_passing_same_subset.max() <= 2,
    )
    w = pd.read_csv(out / "selected_weighted_sensitivity.csv")
    gate(
        "weighted analysis explicitly uses 1369 eligible evaluations",
        set(w.n) == {1369},
    )
    gate(
        "itemized full sensitivity has eight models",
        len(pd.read_csv(out / "itemized_full_model_sensitivity.csv")) == 8,
    )
    c.dump_json(
        out / "analysis_validation_report.json",
        {
            "status": "PASS",
            "checks": checks,
            "limits": "Consistency gates do not turn a reused retrospective evaluation wave into prospective confirmation.",
        },
    )
    print(json.dumps({"status": "PASS", "checks_passed": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
