if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
from pathlib import Path
from hashlib import sha256
import argparse
import json
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate stored representative-policy metrics; no participant data required."
    )
    parser.add_argument(
        "--package-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--document",
        type=Path,
        help="Optional current manuscript to cross-check Table 3; requires python-docx.",
    )
    args = parser.parse_args()
    ROOT = args.package_root.resolve()
    OUT = (
        args.output_dir.resolve()
        if args.output_dir
        else ROOT / "analysis_20260916/outputs/auxiliary_metrics"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    CURRENT = "00_current_revision_20260909/outputs/"
    PRIOR = "01_required_prior_analysis/revision_20260907/outputs/"
    FILES = {
        "selected": CURRENT + "selected_policy_auxiliary_metrics.csv",
        "temporal": PRIOR + "temporal_budget_policy_metrics.csv",
        "development": PRIOR + "development_budget_policies.csv",
        "calibration": CURRENT + "calibration_numerical_validation.csv",
    }
    source_hashes = {
        k: {"path": p, "sha256": sha256((ROOT / p).read_bytes()).hexdigest()}
        for k, p in FILES.items()
    }
    selected = pd.read_csv(ROOT / FILES["selected"])
    temporal = pd.read_csv(ROOT / FILES["temporal"])
    development = pd.read_csv(ROOT / FILES["development"])
    calibration = pd.read_csv(ROOT / FILES["calibration"])
    keys = ["policy_id", "evaluation_domain"]
    assert len(selected) == 60 and (not selected.duplicated(keys).any())
    pd.testing.assert_frame_equal(
        selected.reset_index(drop=True),
        temporal[temporal.budget.isin([0, 1, 4, 11, 30])].reset_index(drop=True),
    )
    columns = [
        "algorithm",
        "availability",
        "budget",
        "selected_subset",
        "source_model",
        "selected_additional_items",
        "development_oof_log_loss",
    ]
    joined = selected.merge(
        development[["policy_id"] + columns],
        on="policy_id",
        suffixes=("", "_development"),
        validate="many_to_one",
    )
    for c in columns:
        if c == "development_oof_log_loss":
            assert np.allclose(
                joined[c], joined[c + "_development"], atol=1e-15, rtol=0
            )
        else:
            assert joined[c].equals(joined[c + "_development"]), c
    row_order = [
        f"{algorithm}|{history}|budget={budget:02d}"
        for history in ["history_available", "history_unavailable"]
        for algorithm in ["ridge", "hgb"]
        for budget in [0, 1, 4, 11, 30]
    ]
    document_checked = False
    if args.document:
        from docx import Document

        observed = []
        for row in Document(args.document).tables[2].rows[1:]:
            r = [c.text for c in row.cells]
            history = "history_available" if r[0] == "가용" else "history_unavailable"
            algorithm = "ridge" if r[1] == "릿지" else "hgb"
            pid = f"{algorithm}|{history}|budget={int(r[2]):02d}"
            z = selected[
                (selected.policy_id == pid) & (selected.evaluation_domain == "all")
            ].iloc[0]
            assert z.selected_subset.replace("_", "+") == r[3]
            assert int(z.selected_additional_items) == int(r[4])
            assert (
                f"{z.development_oof_log_loss:.4f}" == r[5]
                and f"{z.log_loss:.4f}" == r[7]
            )
            observed.append(pid)
        assert len(observed) == len(set(observed)) == 20
        row_order = observed
        document_checked = True
    domain_counts = {}
    for domain, (n, events) in {
        "all": (1520, 853),
        "development_disjoint_id": (879, 471),
        "development_shared_id": (641, 382),
    }.items():
        x = selected[selected.evaluation_domain == domain]
        assert len(x) == 20 and (x.n == n).all() and (x.events == events).all()
        assert np.allclose(x.event_rate, events / n)
        domain_counts[domain] = {"n": n, "events": events, "event_rate": events / n}
    for pid, x in selected.groupby("policy_id"):
        x = x.set_index("evaluation_domain")
        for c in ["log_loss", "brier"]:
            assert (
                abs(
                    x.at["all", c]
                    - (
                        879 * x.at["development_disjoint_id", c]
                        + 641 * x.at["development_shared_id", c]
                    )
                    / 1520
                )
                < 1e-14
            )
    none = selected[selected.selected_subset == "NONE"]
    assert (
        none.calibration_slope.isna().all()
        and np.allclose(none.roc_auc, 0.5)
        and np.allclose(none.average_precision, none.event_rate)
    )
    subset = selected[
        selected.evaluation_domain.isin(["all", "development_disjoint_id"])
    ].copy()
    subset["table_row_order"] = subset.policy_id.map(
        {p: i for i, p in enumerate(row_order)}
    )
    subset["domain_order"] = subset.evaluation_domain.map(
        {"all": 0, "development_disjoint_id": 1}
    )
    subset = subset.sort_values(["domain_order", "table_row_order"]).drop(
        columns="domain_order"
    )
    subset = subset.merge(
        calibration[
            keys
            + [
                "tight_mle_calibration_slope",
                "tight_mle_calibration_intercept",
                "slope_numerical_difference",
                "gradient_inf",
            ]
        ],
        on=keys,
        validate="one_to_one",
    )
    subset.to_csv(
        OUT / "representative_policy_metrics_verified.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tables = {}
    for domain in ["all", "development_disjoint_id"]:
        rows = [
            [
                "이용이력",
                "예측모형",
                "원문항 예산",
                "선택안",
                "ROC AUC",
                "AP",
                "Brier 점수",
                "보정 기울기",
            ]
        ]
        for _, r in subset[subset.evaluation_domain == domain].iterrows():
            rows.append(
                [
                    "가용" if r.availability == "history_available" else "미가용",
                    "릿지" if r.algorithm == "ridge" else "HGB",
                    str(r.budget),
                    r.selected_subset.replace("_", "+"),
                    f"{r.roc_auc:.4f}",
                    f"{r.average_precision:.4f}",
                    f"{r.brier:.4f}",
                    (
                        "—"
                        if pd.isna(r.calibration_slope)
                        else f"{r.calibration_slope:.4f}"
                    ),
                ]
            )
        tables[domain] = {"rows": rows, **domain_counts[domain]}
    (OUT / "representative_policy_metric_tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "status": "PASS",
        "scope": "Stored aggregate-output validation; refit validation is provided separately.",
        "output_rows": len(subset),
        "policy_count": 20,
        "source_files": source_hashes,
        "document_table3_checked_this_run": document_checked,
        "domains": domain_counts,
        "max_abs_strict_mle_slope_difference_in_40_rows": float(
            subset.slope_numerical_difference.abs().max()
        ),
        "calibration_slope_for_document": "Original stored calibration_slope preserved; strict MLE slope is a separate numerical check only.",
    }
    (OUT / "representative_policy_metrics_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(subset),
                "document_table3_checked": document_checked,
            }
        )
    )


if __name__ == "__main__":
    main()
