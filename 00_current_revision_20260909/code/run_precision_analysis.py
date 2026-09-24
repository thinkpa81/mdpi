#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import os

for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm, rankdata
from scipy.optimize import minimize

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
from run_exhaustive import loss_vector, OPTIONAL, metrics

REPS = 2000
POLICY_SEED = 20260909
ATTRIBUTION_SEED = 2026090717
TOL = 1e-10
Z80 = float(norm.ppf(0.8))
Z95 = float(norm.ppf(0.95))
Z975 = float(norm.ppf(0.975))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_inputs():
    testpath = DATA_ROOT / "analysis_private/evaluation.csv"
    devpath = DATA_ROOT / "analysis_private/development.csv"
    predpath = DATA_ROOT / "analysis_private/evaluation_predictions_private.npz"
    receipt = json.loads((OLD / "outputs/decision_boundaries_receipt.json").read_text())
    assert receipt["input_evaluation_sha256"] == sha(testpath)
    assert receipt["prediction_sha256"] == sha(predpath)
    assert receipt["policy_sha256"] == sha(
        OLD / "outputs/development_budget_policies.csv"
    )
    assert receipt["seed"] == POLICY_SEED and receipt["bootstrap_reps"] == REPS
    test = pd.read_csv(testpath, dtype={"id": str})
    dev = pd.read_csv(devpath, dtype={"id": str})
    ids = pd.read_csv(
        DATA_ROOT / "analysis_private/evaluation_ids_private.csv", dtype={"id": str}
    )
    assert test.id.is_unique and dev.id.is_unique
    assert np.array_equal(test.id, ids.id)
    y = test.outcome_any_use.to_numpy(dtype=int)
    assert np.array_equal(y, ids.outcome)
    disjoint = ~test.id.isin(set(dev.id)).to_numpy()
    assert len(test) == 1520 and y.sum() == 853 and disjoint.sum() == 879
    return (
        y,
        disjoint,
        np.load(predpath),
        {
            "development_sha256": sha(devpath),
            "evaluation_sha256": sha(testpath),
            "evaluation_predictions_sha256": sha(predpath),
            "development_policy_sha256": sha(
                OLD / "outputs/development_budget_policies.csv"
            ),
        },
    )


def policy_precision(y, disjoint, pred):
    policy = pd.read_csv(OLD / "outputs/development_budget_policies.csv")
    old_primary = pd.read_csv(OLD / "outputs/budget_policy_loss_boundaries.csv")
    old_standard = pd.read_csv(OLD / "outputs/standardized_policy_bounds.csv")
    families = pd.read_csv(OLD / "outputs/standardized_policy_families.csv")
    w_all = np.random.default_rng(POLICY_SEED).multinomial(
        len(y), np.ones(len(y)) / len(y), size=REPS
    )
    rows = []
    family_rows = []
    maxerrors = {
        "delta": 0.0,
        "primary_bound": 0.0,
        "standardized_bound": 0.0,
        "bootstrap_se": 0.0,
    }
    for algorithm in ("ridge", "hgb"):
        for available in (False, True):
            availability = "history_available" if available else "history_unavailable"
            fam = policy[
                (policy.algorithm == algorithm) & (policy.availability == availability)
            ]
            assert len(fam) == 31 and sorted(fam.budget) == list(range(31))
            baseline = "P0" if available else algorithm + "::NONE"
            full = (
                algorithm
                + "::"
                + ("_".join(["H"] + OPTIONAL) if available else "_".join(OPTIONAL))
            )
            assert np.array_equal(pred[algorithm + "::NONE"], pred["prevalence"])
            pairs = list(
                dict.fromkeys(
                    (k, r) for k in fam.chosen_model for r in (full, baseline) if k != r
                )
            )
            keys = list(
                dict.fromkeys([k for pair in pairs for k in pair] + [full, baseline])
            )
            ix = {k: i for i, k in enumerate(keys)}
            for domain, mask in [
                ("whole", np.ones(len(y), dtype=bool)),
                ("disjoint", disjoint),
            ]:
                losses = np.column_stack(
                    [loss_vector(y[mask], pred[k][mask]) for k in keys]
                )
                means = losses.mean(axis=0)
                w = w_all[:, mask].astype(float)
                w /= w.sum(axis=1, keepdims=True)
                boot = w @ losses
                delta = np.array([means[ix[k]] - means[ix[r]] for k, r in pairs])
                bdelta = np.column_stack(
                    [boot[:, ix[k]] - boot[:, ix[r]] for k, r in pairs]
                )
                error = delta[None, :] - bdelta
                se = bdelta.std(axis=0, ddof=1)
                assert (se > 1e-14).all()
                q_abs = float(np.quantile(error.max(axis=1), 0.95))
                q_std = float(np.quantile((error / se).max(axis=1), 0.95))
                oldfam = families[
                    (families.algorithm == algorithm)
                    & (families.history_available == available)
                    & (families.domain == domain)
                ].iloc[0]
                assert np.isclose(
                    q_abs, oldfam.primary_absolute_critical_value, atol=TOL, rtol=0
                )
                assert np.isclose(
                    q_std, oldfam.standardized_maximum_critical_value, atol=TOL, rtol=0
                )
                assert len(pairs) == oldfam.contrasts
                family_rows.append(
                    {
                        "algorithm": algorithm,
                        "history_available": available,
                        "domain": domain,
                        "n": int(mask.sum()),
                        "unique_nonself_contrasts": len(pairs),
                        "q_primary_absolute": q_abs,
                        "q_standardized": q_std,
                        "coverage": "separately for each algorithm/availability/domain family",
                        "bootstrap_reps": REPS,
                        "seed": POLICY_SEED,
                    }
                )
                pi = {pair: i for i, pair in enumerate(pairs)}
                for _, p in fam.iterrows():
                    match = (
                        (old_primary.algorithm == algorithm)
                        & (old_primary.history_available == available)
                        & (old_primary.domain == domain)
                        & (old_primary.budget == p.budget)
                    )
                    op = old_primary[match].iloc[0]
                    os = old_standard[
                        (old_standard.algorithm == algorithm)
                        & (old_standard.history_available == available)
                        & (old_standard.domain == domain)
                        & (old_standard.budget == p.budget)
                    ].iloc[0]
                    for reference_label, reference in [
                        ("full", full),
                        ("benchmark", baseline),
                    ]:
                        self_contrast = p.chosen_model == reference
                        row = {
                            **p.to_dict(),
                            "domain": domain,
                            "n": int(mask.sum()),
                            "reference_label": reference_label,
                            "reference_model": reference,
                            "self_contrast": self_contrast,
                            "family_unique_nonself_contrasts": len(pairs),
                            "q_primary_absolute": q_abs,
                            "q_standardized": q_std,
                            "inference_status": "post hoc; fixed-model conditional local-shift approximation, not achieved power",
                        }
                        if self_contrast:
                            d = sej = lo = hi = up = uprim = ustd = 0.0
                            extra = {
                                key: np.nan
                                for key in (
                                    "critical_primary",
                                    "critical_standardized",
                                    "critical_marginal_normal",
                                    "critical_marginal_bootstrap",
                                    "mdd80_primary_normal",
                                    "mdd80_standardized_normal",
                                    "mdd80_marginal_normal",
                                    "mdd80_primary_bootstrap_shift",
                                    "mdd80_standardized_bootstrap_shift",
                                    "mdd80_marginal_bootstrap_shift",
                                )
                            }
                        else:
                            j = pi[(p.chosen_model, reference)]
                            d = float(delta[j])
                            sej = float(se[j])
                            e = error[:, j]
                            lo = d + float(np.quantile(e, 0.025))
                            hi = d + float(np.quantile(e, 0.975))
                            qm = float(np.quantile(e, 0.95))
                            up = d + qm
                            uprim = d + q_abs
                            ustd = d + q_std * sej
                            shift80 = float(np.quantile(-e, 0.8))
                            extra = {
                                "critical_primary": q_abs,
                                "critical_standardized": q_std * sej,
                                "critical_marginal_normal": Z95 * sej,
                                "critical_marginal_bootstrap": qm,
                                "mdd80_primary_normal": q_abs + Z80 * sej,
                                "mdd80_standardized_normal": (q_std + Z80) * sej,
                                "mdd80_marginal_normal": (Z95 + Z80) * sej,
                                "mdd80_primary_bootstrap_shift": q_abs + shift80,
                                "mdd80_standardized_bootstrap_shift": q_std * sej
                                + shift80,
                                "mdd80_marginal_bootstrap_shift": qm + shift80,
                            }
                        maxerrors["delta"] = max(
                            maxerrors["delta"],
                            abs(d - op["delta_vs_" + reference_label]),
                        )
                        maxerrors["primary_bound"] = max(
                            maxerrors["primary_bound"],
                            abs(uprim - op["upper95_vs_" + reference_label]),
                        )
                        maxerrors["standardized_bound"] = max(
                            maxerrors["standardized_bound"],
                            abs(
                                ustd - os["standardized_upper95_vs_" + reference_label]
                            ),
                        )
                        maxerrors["bootstrap_se"] = max(
                            maxerrors["bootstrap_se"],
                            abs(sej - os["bootstrap_se_vs_" + reference_label]),
                        )
                        row.update(
                            delta_log_loss=d,
                            benefit_log_loss=-d,
                            bootstrap_se=sej,
                            marginal_delta95_low=lo,
                            marginal_delta95_high=hi,
                            marginal_delta_upper95=up,
                            marginal_benefit95_low=-hi,
                            marginal_benefit95_high=-lo,
                            marginal_benefit_lower95=-up,
                            primary_delta_upper95=uprim,
                            standardized_delta_upper95=ustd,
                            primary_benchmark_or_full_superiority=bool(uprim < 0),
                            standardized_benchmark_or_full_superiority=bool(ustd < 0),
                            marginal_superiority_unadjusted=bool(up < 0),
                            **extra,
                        )
                        rows.append(row)
    result = pd.DataFrame(rows)
    assert len(result) == 496 and len(family_rows) == 8
    assert all(v < TOL for v in maxerrors.values()), maxerrors
    assert (
        result[result.self_contrast][
            ["delta_log_loss", "bootstrap_se", "marginal_delta_upper95"]
        ]
        .to_numpy()
        .sum()
        == 0
    )
    assert result[result.self_contrast].mdd80_primary_normal.isna().all()
    assert np.allclose(result.marginal_benefit95_low, -result.marginal_delta95_high)
    result.to_csv(OUT / "policy_marginal_intervals_and_precision.csv", index=False)
    pd.DataFrame(family_rows).to_csv(OUT / "precision_policy_families.csv", index=False)
    result[
        (result.reference_label == "benchmark") & result.budget.isin([1, 4, 11, 30])
    ].to_csv(OUT / "policy_precision_selected_budgets.csv", index=False)
    return result, {
        "original_policy_rows_crosschecked": 248,
        "output_contrast_rows": 496,
        "unique_families": 8,
        "max_abs_reconstruction_errors": maxerrors,
        "self_contrast_rows": int(result.self_contrast.sum()),
    }


def attribution_precision():
    path = OLD / "private/block_attribution_individual_private.npz"
    cached = np.load(path)
    phi = cached["evaluation_phi"]
    disjoint = cached["evaluation_disjoint"]
    boot = cached["evaluation_aggregate_bootstrap"]
    means = np.r_[phi.mean(axis=0), phi[disjoint].mean(axis=0)]
    assert boot.shape == (REPS, 56)
    primary = pd.read_csv(OLD / "outputs/block_attribution_evaluation.csv")
    standard = pd.read_csv(OLD / "outputs/standardized_block_attribution_bounds.csv")
    manifest = json.loads((OLD / "outputs/block_attribution_manifest.json").read_text())
    assert (
        manifest["bootstrap"]["seed"] == ATTRIBUTION_SEED
        and manifest["bootstrap"]["reps"] == REPS
    )
    expected = []
    for domain in ("whole", "person_disjoint"):
        for key in cached["column_keys"]:
            algo, hflag, block = str(key).split("::")
            expected.append((domain, algo, hflag == "H1", block))
    assert expected == [
        (r.domain, r.algorithm, bool(r.has_history), r.block)
        for _, r in primary.iterrows()
    ]
    err = means[None, :] - boot
    se = boot.std(axis=0, ddof=1)
    q_abs = float(np.quantile(np.abs(err).max(axis=1), 0.95))
    q_std = float(np.quantile(np.abs(err / se).max(axis=1), 0.95))
    assert np.allclose(means, primary.shapley_log_loss_reduction, atol=TOL, rtol=0)
    assert np.allclose(se, primary.bootstrap_se, atol=TOL, rtol=0)
    assert np.isclose(
        q_abs, manifest["bootstrap"]["critical_absolute_log_loss"], atol=TOL, rtol=0
    )
    assert np.allclose(
        means - q_std * se, standard.standardized_joint95_low, atol=TOL, rtol=0
    )
    assert np.allclose(
        means + q_std * se, standard.standardized_joint95_high, atol=TOL, rtol=0
    )
    result = primary.copy()
    result["marginal95_low"] = means + np.quantile(err, 0.025, axis=0)
    result["marginal95_high"] = means + np.quantile(err, 0.975, axis=0)
    result["marginal_one_sided_lower95"] = means + np.quantile(err, 0.05, axis=0)
    result["q_primary_absolute"] = q_abs
    result["q_standardized"] = q_std
    result["standardized_joint95_low"] = means - q_std * se
    result["standardized_joint95_high"] = means + q_std * se
    result["critical_primary_positive_direction"] = q_abs
    result["critical_standardized_positive_direction"] = q_std * se
    result["critical_marginal_two_sided_normal"] = Z975 * se
    result["mdd80_primary_positive_direction_normal"] = q_abs + Z80 * se
    result["mdd80_standardized_positive_direction_normal"] = (q_std + Z80) * se
    result["mdd80_marginal_two_sided_positive_direction_normal"] = (Z975 + Z80) * se
    shift80 = -np.quantile(boot - means, 0.2, axis=0)
    result["mdd80_primary_positive_direction_bootstrap_shift"] = q_abs + shift80
    result["mdd80_standardized_positive_direction_bootstrap_shift"] = (
        q_std * se + shift80
    )

    result["mdd80_marginal_two_sided_positive_direction_bootstrap_shift"] = (
        -np.quantile(err, 0.025, axis=0) + shift80
    )
    result["precision_status"] = (
        "post hoc fixed-model local shift; positive direction crossing lower bound of two-sided family; not two-sided overall power"
    )
    result.to_csv(OUT / "attribution_marginal_intervals_and_precision.csv", index=False)
    return result, {
        "effects": 56,
        "q_primary_absolute": q_abs,
        "q_standardized": q_std,
        "bootstrap_sha256": sha(path),
        "positive_direction_only": True,
    }


def independent_metrics(y, p):
    n = len(y)
    n1 = int(y.sum())
    n0 = n - n1
    brier = float(np.mean((p - y) ** 2))

    ranks = rankdata(p, method="average")
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    order = np.argsort(-p, kind="mergesort")
    sp = p[order]
    sy = y[order]
    last = np.r_[np.flatnonzero(np.diff(sp) != 0), n - 1]
    tp = np.cumsum(sy)[last]
    total = last + 1
    ap = float(np.sum(np.diff(np.r_[0, tp]) / n1 * tp / total))
    return {
        "log_loss": float(loss_vector(y, p).mean()),
        "brier": brier,
        "roc_auc": auc,
        "average_precision": ap,
    }


def calibration_mle(y, p):
    if np.std(p) <= 1e-12:
        return None
    x = logit(np.clip(p, 1e-8, 1 - 1e-8))
    design = np.column_stack([np.ones(len(y)), x])

    def fun(beta):
        z = design @ beta
        return float(np.mean(np.logaddexp(0, z) - y * z))

    def jac(beta):
        return design.T @ (expit(design @ beta) - y) / len(y)

    def hess(beta):
        pp = expit(design @ beta)
        return (design.T * (pp * (1 - pp))) @ design / len(y)

    result = minimize(
        fun,
        np.array([0.0, 1.0]),
        jac=jac,
        hess=hess,
        method="trust-exact",
        options={"gtol": 1e-10, "maxiter": 500},
    )
    assert np.linalg.norm(jac(result.x), ord=np.inf) < 1e-8
    return {
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
        "gradient_inf": float(np.max(np.abs(jac(result.x)))),
    }


def metric_validation(y, disjoint, pred):
    source = pd.read_csv(OLD / "outputs/temporal_budget_policy_metrics.csv")
    assert len(source) == 372
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "development_disjoint_id": disjoint,
        "development_shared_id": ~disjoint,
    }
    errors = {
        k: 0.0
        for k in [
            "log_loss",
            "brier",
            "roc_auc",
            "average_precision",
            "calibration_intercept",
            "calibration_slope",
        ]
    }
    computed = {}
    checkrows = []
    for _, r in source.iterrows():
        key = (r.evaluation_domain, r.source_model)
        mask = masks[r.evaluation_domain]
        if key not in computed:
            pp = pred[r.source_model][mask]
            recalc = independent_metrics(y[mask], pp)

            original = metrics(y[mask], pp)
            mle = calibration_mle(y[mask], pp)
            computed[key] = (recalc, original, mle)
        recalc, original, mle = computed[key]
        assert r.n == mask.sum() and r.events == y[mask].sum()
        for name in ("log_loss", "brier", "roc_auc", "average_precision"):
            errors[name] = max(errors[name], abs(recalc[name] - r[name]))
        for name in ("calibration_intercept", "calibration_slope"):
            if pd.isna(r[name]):
                assert np.isnan(original[name])
            else:
                errors[name] = max(errors[name], abs(original[name] - r[name]))
        checkrows.append(
            {
                "policy_id": r.policy_id,
                "evaluation_domain": r.evaluation_domain,
                "source_model": r.source_model,
                "constant_prediction": mle is None,
                "existing_calibration_slope": r.calibration_slope,
                "tight_mle_calibration_slope": np.nan if mle is None else mle["slope"],
                "slope_numerical_difference": (
                    np.nan if mle is None else mle["slope"] - r.calibration_slope
                ),
                "tight_mle_calibration_intercept": (
                    np.nan if mle is None else mle["intercept"]
                ),
                "gradient_inf": np.nan if mle is None else mle["gradient_inf"],
            }
        )
    assert all(v < TOL for v in errors.values()), errors
    checks = pd.DataFrame(checkrows)
    checks.to_csv(OUT / "calibration_numerical_validation.csv", index=False)
    source[source.budget.isin([0, 1, 4, 11, 30])].to_csv(
        OUT / "selected_policy_auxiliary_metrics.csv", index=False
    )
    return {
        "source_rows": len(source),
        "unique_model_domain_pairs": len(computed),
        "all_existing_metrics_reproduced": True,
        "maximum_abs_errors": errors,
        "max_abs_tighter_mle_slope_difference": float(
            checks.slope_numerical_difference.abs().max()
        ),
        "constant_prediction_slope_not_identifiable": True,
        "note": "Selected-policy table preserves existing values; tighter unpenalized MLE is a numerical diagnostic.",
    }


def summary(policy, attribution, validation):
    lines = [
        "# 추가 정밀도 진단 및 성능 검산 결과",
        "",
        "기존 자료와 결과를 확인한 후 실시한 후향적 보조분석이다. 모든 기존 정책과 주 공동구간을 유지하였다. MDD80은 현재 적합모형, 현재 표준오차 및 고정 임계값을 전제로 한 국소 이동 근사이며 달성 검출력이나 전향적 설계값이 아니다.",
        "",
        "## 정책 비교집합과 계산",
        "",
        "정책은 학습기·이력 가용성·평가집단별 8개 비교집합이다. 31개 예산에서 중복·자기대비를 제거한 전체정보 및 기준선 대비를 각 집합 안에서 함께 보정했다. 56개 귀속은 이들과 별개의 양측 비교집합이다. 따라서 56개 효과와 모든 정책을 하나로 보정했다는 평가는 정확하지 않다.",
        "",
        "정책 이득 = 참조 로그손실 − 정책 로그손실. 양수이면 정책이 우수하다. MDD80 ≈ c + 0.841621×SE. c는 주 공동구간의 q_abs 또는 표준화 공동구간의 q_std×SE이며, 주변 단측 정규근사의 경우 1.644854×SE이다.",
        "",
        "## 전체 평가집단: 기준선 대비 정밀도",
        "",
        "| 학습기 | 이력 가용 | 예산 | 이득 | 주변 95% 양측 구간 | 주 MDD80 | 표준화 MDD80 | 주변 MDD80 |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    selected = policy[
        (policy.domain == "whole")
        & (policy.reference_label == "benchmark")
        & policy.budget.isin([1, 4, 11, 30])
    ]
    for _, r in selected.iterrows():
        lines.append(
            f"| {r.algorithm} | {r.history_available} | {r.budget} | {r.benefit_log_loss:.6f} | [{r.marginal_benefit95_low:.6f}, {r.marginal_benefit95_high:.6f}] | {r.mdd80_primary_normal:.6f} | {r.mdd80_standardized_normal:.6f} | {r.mdd80_marginal_normal:.6f} |"
        )
    lines += ["", "## 가상 이득 규모와의 비교", ""]
    for domain in ("whole", "disjoint"):
        g = policy[
            (policy.domain == domain)
            & (policy.reference_label == "benchmark")
            & (~policy.self_contrast)
        ].drop_duplicates(
            ["algorithm", "history_available", "chosen_model", "reference_model"]
        )
        lines.append(
            f"- {domain}: 기준선 대비 서로 다른 {len(g)}개 비자기대비. 표준화 MDD80 범위 {g.mdd80_standardized_normal.min():.6f}–{g.mdd80_standardized_normal.max():.6f}, 주변 MDD80 범위 {g.mdd80_marginal_normal.min():.6f}–{g.mdd80_marginal_normal.max():.6f}."
        )
        for value in (0.005, 0.01):
            lines.append(
                f"  - 가상 이득 {value:.3f}가 근사 MDD80 이상인 대비 수: 주 {int((g.mdd80_primary_normal<=value).sum())}, 표준화 {int((g.mdd80_standardized_normal<=value).sum())}, 주변 {int((g.mdd80_marginal_normal<=value).sum())}."
            )
    lines += [
        "",
        "추가 문항을 실제로 수집하는 정책만 구분하면 다음과 같다. 예산 0의 HGB H와 P0는 거의 같은 예측이어서 매우 작은 조건부 표준오차가 나오므로, 위 전체 범위의 최솟값을 추가 수집의 정밀도로 해석하지 않는다.",
    ]
    for domain in ("whole", "disjoint"):
        g = policy[
            (policy.domain == domain)
            & (policy.reference_label == "benchmark")
            & (policy.selected_items > 0)
            & (~policy.self_contrast)
        ].drop_duplicates(
            ["algorithm", "history_available", "chosen_model", "reference_model"]
        )
        lines.append(
            f"- {domain}, 실제 추가 문항 수 > 0: 서로 다른 {len(g)}개 대비. 표준화 MDD80 {g.mdd80_standardized_normal.min():.6f}–{g.mdd80_standardized_normal.max():.6f}, 주변 MDD80 {g.mdd80_marginal_normal.min():.6f}–{g.mdd80_marginal_normal.max():.6f}. 가상 이득 .005 이상/미만의 단정 대신 개별 행을 보고한다."
        )
    lines += [
        "",
        "## 블록 귀속의 양의 방향 정밀도",
        "",
        "아래 MDD80은 양측 구간의 하한이 0을 초과하는 양의 방향 발견에 관한 근사이다. 양·음 어느 방향이든 발견할 전체 양측 검출력을 계산한 값이 아니다.",
        "",
        "| 집단 | 학습기 | H 고정 포함 | 블록 | 귀속 | 주변 95% 구간 | 표준화 MDD80 |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for _, r in attribution[attribution.block.isin(["D", "K"])].iterrows():
        lines.append(
            f"| {r.domain} | {r.algorithm} | {r.has_history} | {r.block} | {r.shapley_log_loss_reduction:.6f} | [{r.marginal95_low:.6f}, {r.marginal95_high:.6f}] | {r.mdd80_standardized_positive_direction_normal:.6f} |"
        )
    bm = policy[policy.reference_label == "benchmark"]
    lines += [
        "",
        "## 해석과 검증",
        "",
        f"- 주변 단측 95% 상한에서 기준선 우월성이 나타나는 예산·집단 행 수는 {int(bm.marginal_superiority_unadjusted.sum())}개이다. 같은 자료에서 다수 대비를 확인한 주변 결과이므로 이를 공동보정 결과를 대체하는 확정 수집 권고로 사용하지 않는다.",
        f"- 주 및 표준화 공동구간의 기준선 우월 행 수는 각각 {int(bm.primary_benchmark_or_full_superiority.sum())}개와 {int(bm.standardized_benchmark_or_full_superiority.sum())}개로 기존 결과가 유지된다.",
        "- 모든 작은 효과가 탐지 불가능하다는 일괄 주장은 피한다. 대비별 짝지은 손실 차이의 분산, 비교집합의 임계값, 평가집단에 따라 정밀도가 다르다. 효과가 확인되지 않았다는 결과와 효과가 없다는 결론을 구분한다.",
        "- 기존 372개 성능행의 Log loss·ROC-AUC·Average Precision·Brier를 독립 산식으로 재계산하고, 보정 기울기를 기존 구현으로 재현하였다. 상수 예측의 보정 기울기는 식별되지 않아 공란으로 유지했다.",
        f'- 더 엄격한 무벌점 우도 최적화와 기존 보정 기울기의 최대 수치 차이는 {validation["metrics"]["max_abs_tighter_mle_slope_difference"]:.6f}이었다. 이 진단은 예측모형 재보정이나 정책 변경을 수행하지 않았다.',
        "- 검증·입력 해시·정확한 수치는 precision_validation.json 및 CSV에 제시한다.",
    ]
    (OUT / "precision_analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = REV / "precision_protocol.md"
    assert protocol.exists()
    y, disjoint, pred, input_hashes = load_inputs()
    policy, pv = policy_precision(y, disjoint, pred)
    attribution, av = attribution_precision()
    mv = metric_validation(y, disjoint, pred)
    result = {
        "status": "PASS",
        "protocol_sha256": sha(protocol),
        "code_sha256": sha(__file__),
        "input_hashes": input_hashes,
        "policy": pv,
        "attribution": av,
        "metrics": mv,
        "normal_quantiles": {"z80": Z80, "z95": Z95, "z975": Z975},
        "post_hoc": True,
        "original_policies_unchanged": True,
        "original_primary_inference_retained": True,
        "limits": "conditional fixed-model local additive-shift precision, not achieved power, exact prospective MDD, future sample-size calculation, training-selection or adaptive-reuse uncertainty",
    }
    dump(OUT / "precision_validation.json", result)
    summary(policy, attribution, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
