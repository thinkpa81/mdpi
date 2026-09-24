#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )

import hashlib
import json
import os
from pathlib import Path
import sys

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
PRIOR_ROOT = (
    ROOT / "01_required_prior_analysis"
    if (ROOT / "01_required_prior_analysis").is_dir()
    else ROOT
)
DATA_ROOT = ROOT if (ROOT / "analysis_private").is_dir() else PRIOR_ROOT
OLD = PRIOR_ROOT / "revision_20260907"
OUT = REV / "outputs"
sys.path.insert(0, str(PRIOR_ROOT / "code"))
from run_exhaustive import OPTIONAL, loss_vector

REPS = 2000
SEED = 20260909
TOL = 1e-12
DEGENERATE_SE = 1e-14


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_inputs():
    protocol = REV / "ridge_sensitivity_protocol.md"
    assert (
        protocol.exists()
    ), "The result-independent protocol must exist before execution."

    evaluation_path = DATA_ROOT / "analysis_private/evaluation.csv"
    development_path = DATA_ROOT / "analysis_private/development.csv"
    prediction_path = DATA_ROOT / "analysis_private/evaluation_predictions_private.npz"
    policy_path = OLD / "outputs/development_budget_policies.csv"
    old_family_path = OLD / "outputs/standardized_policy_families.csv"
    old_bound_path = OLD / "outputs/standardized_policy_bounds.csv"
    old_primary_path = OLD / "outputs/budget_policy_loss_boundaries.csv"

    receipt = json.loads((OLD / "outputs/decision_boundaries_receipt.json").read_text())
    assert receipt["input_evaluation_sha256"] == sha256(evaluation_path)
    assert receipt["prediction_sha256"] == sha256(prediction_path)
    assert receipt["policy_sha256"] == sha256(policy_path)
    assert receipt["seed"] == SEED and receipt["bootstrap_reps"] == REPS

    evaluation = pd.read_csv(evaluation_path, dtype={"id": str})
    development = pd.read_csv(development_path, dtype={"id": str})
    ids = pd.read_csv(
        DATA_ROOT / "analysis_private/evaluation_ids_private.csv", dtype={"id": str}
    )
    assert evaluation.id.is_unique and development.id.is_unique
    assert np.array_equal(evaluation.id.to_numpy(), ids.id.to_numpy())
    y = evaluation.outcome_any_use.to_numpy(dtype=int)
    assert np.array_equal(y, ids.outcome.to_numpy(dtype=int))
    disjoint = ~evaluation.id.isin(set(development.id)).to_numpy()
    assert len(y) == 1520 and int(y.sum()) == 853 and int(disjoint.sum()) == 879

    policies = pd.read_csv(policy_path)
    old_families = pd.read_csv(old_family_path)
    old_bounds = pd.read_csv(old_bound_path)
    old_primary = pd.read_csv(old_primary_path)
    assert len(policies) == 124 and len(old_families) == 8 and len(old_bounds) == 248
    assert len(old_primary) == 248

    assert set(old_families.algorithm) == {"ridge", "hgb"}
    assert not old_families.duplicated(
        ["algorithm", "history_available", "domain"]
    ).any()
    assert (old_families.groupby("algorithm").size() == 4).all()

    hashes = {
        "protocol_sha256": sha256(protocol),
        "evaluation_sha256": sha256(evaluation_path),
        "development_sha256": sha256(development_path),
        "prediction_sha256": sha256(prediction_path),
        "policy_sha256": sha256(policy_path),
        "old_family_sha256": sha256(old_family_path),
        "old_standardized_bounds_sha256": sha256(old_bound_path),
        "old_primary_bounds_sha256": sha256(old_primary_path),
    }
    return (
        y,
        disjoint,
        np.load(prediction_path),
        policies,
        old_families,
        old_bounds,
        old_primary,
        hashes,
    )


def bootstrap_weights(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=REPS)


def ridge_only_recalculation(
    y, disjoint, predictions, policies, old_families, old_bounds, old_primary, weights
):

    result_rows = []
    family_rows = []
    reconstruction_error = {
        "q_primary_absolute": 0.0,
        "q_standardized": 0.0,
        "delta": 0.0,
        "bootstrap_se": 0.0,
        "primary_upper": 0.0,
        "standardized_upper": 0.0,
    }

    for available in (False, True):
        availability = "history_available" if available else "history_unavailable"
        family = policies[
            (policies.algorithm == "ridge") & (policies.availability == availability)
        ].copy()
        assert len(family) == 31 and sorted(family.budget) == list(range(31))
        baseline = "P0" if available else "ridge::NONE"
        full = "ridge::" + (
            "_".join(["H"] + OPTIONAL) if available else "_".join(OPTIONAL)
        )
        pairs = list(
            dict.fromkeys(
                (model, reference)
                for model in family.chosen_model
                for reference in (full, baseline)
                if model != reference
            )
        )

        assert all(not model.startswith("hgb::") for pair in pairs for model in pair)
        keys = list(
            dict.fromkeys([key for pair in pairs for key in pair] + [full, baseline])
        )
        assert all(not key.startswith("hgb::") for key in keys)
        key_index = {key: index for index, key in enumerate(keys)}
        pair_index = {pair: index for index, pair in enumerate(pairs)}

        for domain, mask in (
            ("whole", np.ones(len(y), dtype=bool)),
            ("disjoint", disjoint),
        ):
            losses = np.column_stack(
                [loss_vector(y[mask], predictions[key][mask]) for key in keys]
            )
            means = losses.mean(axis=0)
            domain_weights = weights[:, mask].astype(float)
            domain_weights /= domain_weights.sum(axis=1, keepdims=True)
            boot = domain_weights @ losses
            deltas = np.array(
                [means[key_index[a]] - means[key_index[b]] for a, b in pairs]
            )
            boot_deltas = np.column_stack(
                [boot[:, key_index[a]] - boot[:, key_index[b]] for a, b in pairs]
            )
            errors = deltas[None, :] - boot_deltas
            ses = boot_deltas.std(axis=0, ddof=1)
            assert (ses > DEGENERATE_SE).all()
            q_primary = float(np.quantile(errors.max(axis=1), 0.95))
            q_standardized = float(np.quantile((errors / ses).max(axis=1), 0.95))

            old_family = old_families[
                (old_families.algorithm == "ridge")
                & (old_families.history_available == available)
                & (old_families.domain == domain)
            ].iloc[0]
            reconstruction_error["q_primary_absolute"] = max(
                reconstruction_error["q_primary_absolute"],
                abs(q_primary - old_family.primary_absolute_critical_value),
            )
            reconstruction_error["q_standardized"] = max(
                reconstruction_error["q_standardized"],
                abs(q_standardized - old_family.standardized_maximum_critical_value),
            )
            assert len(pairs) == int(old_family.contrasts)
            family_rows.append(
                {
                    "algorithm": "ridge",
                    "history_available": available,
                    "domain": domain,
                    "n": int(mask.sum()),
                    "unique_nonself_contrasts": len(pairs),
                    "q_primary_absolute_recalculated": q_primary,
                    "q_primary_absolute_existing": float(
                        old_family.primary_absolute_critical_value
                    ),
                    "q_standardized_recalculated": q_standardized,
                    "q_standardized_existing": float(
                        old_family.standardized_maximum_critical_value
                    ),
                    "hgb_prediction_keys_used": 0,
                    "hgb_removal_changes_ridge_family": False,
                    "bootstrap_reps": REPS,
                    "seed": SEED,
                }
            )

            for _, policy in family.iterrows():
                model = policy.chosen_model
                row = {
                    "policy_id": policy.policy_id,
                    "availability": availability,
                    "history_available": available,
                    "domain": domain,
                    "n": int(mask.sum()),
                    "budget": int(policy.budget),
                    "chosen_model": model,
                    "chosen_subset": policy.chosen_subset,
                    "baseline": baseline,
                    "full": full,
                    "family_unique_nonself_contrasts": len(pairs),
                    "q_primary_absolute": q_primary,
                    "q_standardized": q_standardized,
                }
                for label, reference in (("full", full), ("benchmark", baseline)):
                    if model == reference:
                        delta = se = primary_upper = standardized_upper = (
                            marginal_upper
                        ) = 0.0
                    else:
                        j = pair_index[(model, reference)]
                        delta = float(deltas[j])
                        se = float(ses[j])
                        primary_upper = delta + q_primary
                        standardized_upper = delta + q_standardized * se
                        marginal_upper = delta + float(np.quantile(errors[:, j], 0.95))
                    old = old_bounds[
                        (old_bounds.algorithm == "ridge")
                        & (old_bounds.history_available == available)
                        & (old_bounds.domain == domain)
                        & (old_bounds.budget == policy.budget)
                    ].iloc[0]
                    old_p = old_primary[
                        (old_primary.algorithm == "ridge")
                        & (old_primary.history_available == available)
                        & (old_primary.domain == domain)
                        & (old_primary.budget == policy.budget)
                    ].iloc[0]
                    reconstruction_error["delta"] = max(
                        reconstruction_error["delta"],
                        abs(delta - old[f"delta_vs_{label}"]),
                    )
                    reconstruction_error["bootstrap_se"] = max(
                        reconstruction_error["bootstrap_se"],
                        abs(se - old[f"bootstrap_se_vs_{label}"]),
                    )
                    reconstruction_error["primary_upper"] = max(
                        reconstruction_error["primary_upper"],
                        abs(primary_upper - old_p[f"upper95_vs_{label}"]),
                    )
                    reconstruction_error["standardized_upper"] = max(
                        reconstruction_error["standardized_upper"],
                        abs(
                            standardized_upper - old[f"standardized_upper95_vs_{label}"]
                        ),
                    )
                    row[f"delta_vs_{label}"] = delta
                    row[f"bootstrap_se_vs_{label}"] = se
                    row[f"primary_upper95_vs_{label}"] = primary_upper
                    row[f"standardized_upper95_vs_{label}"] = standardized_upper
                    row[f"marginal_upper95_vs_{label}"] = marginal_upper
                result_rows.append(row)

    results = pd.DataFrame(result_rows)
    families = pd.DataFrame(family_rows)
    assert len(results) == 124 and len(families) == 4
    assert all(
        value <= TOL for value in reconstruction_error.values()
    ), reconstruction_error
    results.to_csv(OUT / "ridge_only_policy_sensitivity.csv", index=False)
    families.to_csv(OUT / "ridge_only_inference_families.csv", index=False)
    return results, families, reconstruction_error


def four_condition_analysis(y, disjoint, predictions, policies, weights):
    condition_specs = []
    for algorithm in ("ridge", "hgb"):
        for available in (False, True):
            availability = "history_available" if available else "history_unavailable"
            policy = policies[
                (policies.algorithm == algorithm)
                & (policies.availability == availability)
                & (policies.budget == 1)
            ].iloc[0]
            baseline = "P0" if available else algorithm + "::NONE"
            assert policy.chosen_subset == ("H_B" if available else "B")
            condition_specs.append(
                {
                    "algorithm": algorithm,
                    "history_available": available,
                    "availability": availability,
                    "policy_id": policy.policy_id,
                    "model": policy.chosen_model,
                    "subset": policy.chosen_subset,
                    "baseline": baseline,
                }
            )

    all_rows = []
    global_rows = []
    for domain, mask in (
        ("whole", np.ones(len(y), dtype=bool)),
        ("disjoint", disjoint),
    ):
        domain_weights = weights[:, mask].astype(float)
        domain_weights /= domain_weights.sum(axis=1, keepdims=True)
        individual = []
        bootstrap = []
        for spec in condition_specs:
            difference = loss_vector(
                y[mask], predictions[spec["model"]][mask]
            ) - loss_vector(y[mask], predictions[spec["baseline"]][mask])
            theta = float(difference.mean())
            boot = domain_weights @ difference
            error = theta - boot
            se = float(boot.std(ddof=1))
            marginal_low = theta + float(np.quantile(error, 0.025))
            marginal_high = theta + float(np.quantile(error, 0.975))
            marginal_upper = theta + float(np.quantile(error, 0.95))
            individual.append(difference)
            bootstrap.append(boot)
            all_rows.append(
                {
                    **spec,
                    "domain": domain,
                    "n": int(mask.sum()),
                    "delta_log_loss": theta,
                    "benefit_log_loss": -theta,
                    "bootstrap_se": se,
                    "marginal_delta95_low": marginal_low,
                    "marginal_delta95_high": marginal_high,
                    "marginal_delta_upper95": marginal_upper,
                    "direction_favors_budget1": bool(theta < 0),
                    "marginal_superiority_unadjusted": bool(marginal_upper < 0),
                }
            )

        individual_matrix = np.column_stack(individual)
        bootstrap_matrix = np.column_stack(bootstrap)
        composite_individual = individual_matrix.mean(axis=1)
        composite_boot = bootstrap_matrix.mean(axis=1)
        theta = float(composite_individual.mean())
        assert np.isclose(
            theta, np.mean([row["delta_log_loss"] for row in all_rows[-4:]])
        )
        error = theta - composite_boot
        se = float(composite_boot.std(ddof=1))
        analytic_se = float(composite_individual.std(ddof=1) / np.sqrt(mask.sum()))
        low = theta + float(np.quantile(error, 0.025))
        high = theta + float(np.quantile(error, 0.975))
        upper = theta + float(np.quantile(error, 0.95))

        centered = composite_boot - theta
        p_boot = float((1 + np.sum(centered <= theta)) / (REPS + 1))
        z = theta / analytic_se
        p_normal = float(norm.cdf(z))
        covariance = np.cov(individual_matrix, rowvar=False, ddof=1) / mask.sum()
        assert np.isclose(
            analytic_se**2,
            np.ones(4) @ covariance @ np.ones(4) / 16,
            atol=1e-14,
            rtol=0,
        )
        global_rows.append(
            {
                "domain": domain,
                "n": int(mask.sum()),
                "conditions": 4,
                "negative_point_estimates": int(
                    sum(row["delta_log_loss"] < 0 for row in all_rows[-4:])
                ),
                "equal_weight_composite_delta": theta,
                "equal_weight_composite_benefit": -theta,
                "paired_bootstrap_se": se,
                "analytic_dependence_aware_se": analytic_se,
                "basic_bootstrap_delta95_low": low,
                "basic_bootstrap_delta95_high": high,
                "basic_bootstrap_delta_upper95": upper,
                "one_sided_null_centered_bootstrap_p": p_boot,
                "one_sided_normal_p": p_normal,
                "independent_sign_test_one_sided_reference_p": float(
                    binomtest(4, 4, 0.5, alternative="greater").pvalue
                ),
                "independent_sign_test_two_sided_reference_p": float(
                    binomtest(4, 4, 0.5, alternative="two-sided").pvalue
                ),
                "sign_test_valid_for_inference": False,
                "status": "post-hoc equal-weight fixed-prediction composite; dependence preserved by paired resampling",
            }
        )

    condition_table = pd.DataFrame(all_rows)
    global_table = pd.DataFrame(global_rows)
    assert len(condition_table) == 8 and len(global_table) == 2
    condition_table.to_csv(OUT / "budget1_four_condition_contrasts.csv", index=False)
    global_table.to_csv(OUT / "budget1_equal_weight_global_contrast.csv", index=False)
    return condition_table, global_table


def stability_audit(policies):
    frequencies = pd.read_csv(OLD / "outputs/nested_selection_frequencies.csv")
    rows = []
    for algorithm in ("ridge", "hgb"):
        for availability in ("history_available", "history_unavailable"):
            frozen = policies[
                (policies.algorithm == algorithm)
                & (policies.availability == availability)
                & (policies.budget == 30)
            ].iloc[0]
            family = frequencies[
                (frequencies.algorithm == algorithm)
                & (frequencies.availability == availability)
                & (frequencies.budget == 30)
            ]
            assert int(family["count"].sum()) == 10
            selected_count = family.loc[family.subset == frozen.chosen_subset, "count"]
            selected_count = 0 if selected_count.empty else int(selected_count.iloc[0])
            block_counts = {}
            for block in ["H"] + OPTIONAL:
                block_counts[block] = int(
                    sum(
                        r["count"]
                        for _, r in family.iterrows()
                        if block in str(r["subset"]).split("_")
                    )
                )
            rows.append(
                {
                    "algorithm": algorithm,
                    "availability": availability,
                    "budget": 30,
                    "frozen_full_development_subset": frozen.chosen_subset,
                    "frozen_subset_outer_selection_count": selected_count,
                    "outer_partitions": 10,
                    "unique_selected_subsets": int(len(family)),
                    **{
                        f"block_{block}_inclusion_count": count
                        for block, count in block_counts.items()
                    },
                    "zero_match_implies_random": False,
                    "interpretation": "low exact-subset stability; randomness is not identified by this frequency table",
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "budget30_selection_stability_audit.csv", index=False)
    return table


def independent_checks(y, disjoint, predictions, policies, conditions):

    def manual_loss(outcome, probability):
        probability = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
        return -(outcome * np.log(probability) + (1 - outcome) * np.log1p(-probability))

    precision = pd.read_csv(REV / "outputs/policy_marginal_intervals_and_precision.csv")
    max_errors = {
        "manual_delta_vs_new": 0.0,
        "existing_precision_delta_vs_new": 0.0,
        "existing_precision_se_vs_new": 0.0,
        "existing_precision_marginal_low_vs_new": 0.0,
        "existing_precision_marginal_high_vs_new": 0.0,
        "existing_precision_marginal_upper_vs_new": 0.0,
    }
    corr_rows = []
    corr_summaries = []
    labels = []
    for algorithm in ("ridge", "hgb"):
        for available in (False, True):
            labels.append(f"{algorithm}|H{int(available)}")

    for domain, mask in (
        ("whole", np.ones(len(y), dtype=bool)),
        ("disjoint", disjoint),
    ):
        manual_differences = []
        for algorithm in ("ridge", "hgb"):
            for available in (False, True):
                row = conditions[
                    (conditions.algorithm == algorithm)
                    & (conditions.history_available == available)
                    & (conditions.domain == domain)
                ].iloc[0]
                difference = manual_loss(
                    y[mask], predictions[row.model][mask]
                ) - manual_loss(y[mask], predictions[row.baseline][mask])
                manual_differences.append(difference)
                max_errors["manual_delta_vs_new"] = max(
                    max_errors["manual_delta_vs_new"],
                    abs(float(difference.mean()) - row.delta_log_loss),
                )
                old = precision[
                    (precision.algorithm == algorithm)
                    & (precision.history_available == available)
                    & (precision.domain == domain)
                    & (precision.budget == 1)
                    & (precision.reference_label == "benchmark")
                ].iloc[0]
                for old_name, new_name, error_name in (
                    (
                        "delta_log_loss",
                        "delta_log_loss",
                        "existing_precision_delta_vs_new",
                    ),
                    ("bootstrap_se", "bootstrap_se", "existing_precision_se_vs_new"),
                    (
                        "marginal_delta95_low",
                        "marginal_delta95_low",
                        "existing_precision_marginal_low_vs_new",
                    ),
                    (
                        "marginal_delta95_high",
                        "marginal_delta95_high",
                        "existing_precision_marginal_high_vs_new",
                    ),
                    (
                        "marginal_delta_upper95",
                        "marginal_delta_upper95",
                        "existing_precision_marginal_upper_vs_new",
                    ),
                ):
                    max_errors[error_name] = max(
                        max_errors[error_name],
                        abs(float(old[old_name]) - float(row[new_name])),
                    )

        corr = np.corrcoef(np.column_stack(manual_differences), rowvar=False)
        assert corr.shape == (4, 4) and np.allclose(np.diag(corr), 1)
        offdiag = corr[np.triu_indices(4, 1)]
        corr_summaries.append(
            {
                "domain": domain,
                "minimum_offdiagonal_correlation": float(offdiag.min()),
                "maximum_offdiagonal_correlation": float(offdiag.max()),
                "maximum_absolute_offdiagonal_correlation": float(
                    np.abs(offdiag).max()
                ),
                "all_offdiagonal_zero": bool(np.all(np.abs(offdiag) < 1e-12)),
            }
        )
        for i, row_label in enumerate(labels):
            for j, column_label in enumerate(labels):
                corr_rows.append(
                    {
                        "domain": domain,
                        "row_condition": row_label,
                        "column_condition": column_label,
                        "individual_loss_difference_correlation": float(corr[i, j]),
                    }
                )

    assert all(value <= TOL for value in max_errors.values()), max_errors
    pd.DataFrame(corr_rows).to_csv(
        OUT / "budget1_condition_loss_difference_correlations.csv", index=False
    )
    checks = {
        "status": "PASS",
        "manual_log_loss_formula_used": True,
        "existing_precision_table_crosschecked": True,
        "max_abs_errors": max_errors,
        "condition_dependence": corr_summaries,
        "independent_signs_supported": False,
    }
    write_json(OUT / "ridge_sensitivity_independent_checks.json", checks)
    return checks


def build_audit(
    ridge, families, errors, conditions, global_table, stability, independent, hashes
):
    b1 = ridge[(ridge.budget == 1)].copy()
    whole = conditions[conditions.domain == "whole"]
    lines = [
        "# R1·R3 독립 검증 보고",
        "",
        "## 종합 판정",
        "",
        "- **R1의 통계적 전제는 틀림**: 기존 공동보정은 릿지와 HGB를 한 가족으로 묶지 않고, 알고리즘×이력 가용성×평가집단의 8개 가족으로 분리하였다. 따라서 HGB를 삭제해도 릿지의 대조 수와 최대통계량 임계값은 변하지 않는다.",
        "- **HGB의 0/10은 무작위 선택의 증거가 아님**: 0/10은 전체 개발자료에서 동결된 정확한 조합이 외부 학습부분에서 재선택되지 않았다는 뜻이며, 정확조합 안정성이 낮다는 증거이다. 블록 포함 양상은 구조를 보이므로 이 표만으로 무작위성을 판정할 수 없다.",
        "- **R3의 독립 부호검정은 부적절**: 네 대조는 동일 대상자와 중첩된 예측·기준선을 공유한다. 독립성을 가정한 참고 p값은 단측 0.0625, 양측 0.125로, 독립성을 가정해도 통상적인 5% 기준을 넘지 못한다.",
        "- **의존성 보존 결합대조는 사후 보조분석**: 동일가중 평균대조와 쌍체 부트스트랩 결과를 아래에 제시하되, 네 조건 각각의 우월성이나 보편적 개선으로 해석하지 않는다.",
        "",
        "## R1. 릿지 단독 재계산",
        "",
        "| 이력 | 집단 | 대조 수 | 기존/재계산 표준화 임계값 | 기존/재계산 비표준화 임계값 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in families.iterrows():
        lines.append(
            f"| {'가용' if row.history_available else '미가용'} | {row.domain} | {int(row.unique_nonself_contrasts)} | "
            f"{row.q_standardized_existing:.9f}/{row.q_standardized_recalculated:.9f} | "
            f"{row.q_primary_absolute_existing:.9f}/{row.q_primary_absolute_recalculated:.9f} |"
        )
    lines += [
        "",
        "1문항 B 또는 H+B 정책의 기준선 대비 결과:",
        "",
        "| 이력 | 집단 | 로그손실 차이 | 표준화 공동상한 | 주변 단측상한 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in b1.iterrows():
        lines.append(
            f"| {'가용' if row.history_available else '미가용'} | {row.domain} | "
            f"{row.delta_vs_benchmark:.6f} | {row.standardized_upper95_vs_benchmark:.6f} | "
            f"{row.marginal_upper95_vs_benchmark:.6f} |"
        )
    lines += [
        "",
        "HGB를 보충자료로 이동하는 것은 가독성 목적의 편집 선택으로는 가능하다. 그러나 평가결과를 본 뒤 HGB를 제거하여 릿지를 주 분석으로 재정의하는 것은 결과의존적 사양 선택이며, 기존 공동보정 결과를 바꾸지도 않는다.",
        "",
        "## R3. 네 조건의 방향과 의존성 보존 결합대조",
        "",
        "전체표본의 네 조건:",
        "",
        "| 알고리즘 | 이력 | 로그손실 차이 | 주변 95% 구간 |",
        "|---|---|---:|---:|",
    ]
    for _, row in whole.iterrows():
        lines.append(
            f"| {row.algorithm} | {'가용' if row.history_available else '미가용'} | "
            f"{row.delta_log_loss:.6f} | [{row.marginal_delta95_low:.6f}, {row.marginal_delta95_high:.6f}] |"
        )
    lines += [
        "",
        "| 집단 | 네 조건 중 음수 | 동일가중 평균 차이 | 쌍체 SE | 기본 부트스트랩 95% 구간 | 단측 p(중심화 부트스트랩) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in global_table.iterrows():
        lines.append(
            f"| {row.domain} | {int(row.negative_point_estimates)}/4 | "
            f"{row.equal_weight_composite_delta:.6f} | {row.paired_bootstrap_se:.6f} | "
            f"[{row.basic_bootstrap_delta95_low:.6f}, {row.basic_bootstrap_delta95_high:.6f}] | "
            f"{row.one_sided_null_centered_bootstrap_p:.4f} |"
        )
    lines += [
        "",
        "동일가중 결합대조는 서로 다른 두 기준선(P0와 유병률 기준) 및 두 학습계열을 평균한다. 따라서 이는 방향 일치에 대한 제한적인 사후 요약이며, 실제 수집정책 채택 근거로 단독 사용해서는 안 된다.",
        "",
        "## 재현성 검산",
        "",
        f"- 릿지 124개 정책행과 4개 가족을 재계산하였다.",
        f"- 기존 결과와 최대 절대오차: {max(errors.values()):.3e}",
        f"- 별도 로그손실 공식 및 기존 정밀도 표와의 최대 절대오차: {max(independent['max_abs_errors'].values()):.3e}",
        f"- 부트스트랩: {REPS}회, seed={SEED}",
        "- 입력 해시는 `ridge_sensitivity_validation.json`에 기록하였다.",
        "",
        "## 논문 반영 권고",
        "",
        "1. HGB는 비선형 강건성 분석으로 명확히 구분할 수 있으나, 통계적 유의성을 얻기 위한 제외로 설명하지 않는다.",
        "2. 릿지 단독 재계산이 기존 릿지 결과와 동일하다는 사실을 보충자료에 제시한다.",
        "3. ‘4/4 방향 일치’는 기술통계로만 쓰고 독립 부호검정의 유의성 주장을 하지 않는다.",
        "4. 동일가중 결합대조를 제시할 경우 반드시 사후적·고정예측 조건부·서로 다른 기준선의 평균이라는 한계를 함께 쓴다.",
    ]
    path = REV / "audit_r1_r3.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = {
        "status": "PASS",
        "analysis_status": "post-hoc sensitivity; primary analysis retained",
        "input_hashes": hashes,
        "code_sha256": sha256(Path(__file__)),
        "bootstrap_reps": REPS,
        "seed": SEED,
        "ridge_policy_rows": int(len(ridge)),
        "ridge_families": int(len(families)),
        "condition_rows": int(len(conditions)),
        "global_contrast_rows": int(len(global_table)),
        "selection_stability_rows": int(len(stability)),
        "max_abs_existing_ridge_reconstruction_errors": errors,
        "independent_checks": independent,
        "hgb_was_in_same_family_as_ridge": False,
        "hgb_removal_changes_ridge_critical_values": False,
        "independent_sign_test_valid": False,
        "primary_analysis_changed": False,
    }
    write_json(OUT / "ridge_sensitivity_validation.json", validation)
    return validation


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (
        y,
        disjoint,
        predictions,
        policies,
        old_families,
        old_bounds,
        old_primary,
        hashes,
    ) = load_inputs()
    weights = bootstrap_weights(len(y))
    ridge, families, errors = ridge_only_recalculation(
        y,
        disjoint,
        predictions,
        policies,
        old_families,
        old_bounds,
        old_primary,
        weights,
    )
    conditions, global_table = four_condition_analysis(
        y, disjoint, predictions, policies, weights
    )
    stability = stability_audit(policies)
    independent = independent_checks(y, disjoint, predictions, policies, conditions)
    validation = build_audit(
        ridge,
        families,
        errors,
        conditions,
        global_table,
        stability,
        independent,
        hashes,
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
