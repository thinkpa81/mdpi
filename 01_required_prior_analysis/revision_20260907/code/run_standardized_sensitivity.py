#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import os

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[env_name] = "1"
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parents[1]
ROOT = REV.parent
OUT = REV / "outputs"
PRIVATE = REV / "private"
sys.path.insert(0, str(ROOT / "code"))
from run_exhaustive import loss_vector, configurations, OPTIONAL

REPS = 2000
POLICY_SEED = 20260909
ATTRIBUTION_SEED = 2026090717
DEGENERATE_SE = 1e-14


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def standardized_maximum(means, bootstrap, two_sided=False):
    se = bootstrap.std(axis=0, ddof=1)
    valid = se > DEGENERATE_SE
    centered = means[None, :] - bootstrap
    if (~valid).any():
        assert np.max(np.abs(centered[:, ~valid])) < 1e-12
    if valid.any():
        z = centered[:, valid] / se[None, valid]
        maximum = np.max(np.abs(z) if two_sided else z, axis=1)
        q = float(np.quantile(maximum, 0.95))
    else:
        q = 0.0
    return se, q, valid


def policy_sensitivity():
    test_path = ROOT / "analysis_private/evaluation.csv"
    dev_path = ROOT / "analysis_private/development.csv"
    pred_path = ROOT / "analysis_private/evaluation_predictions_private.npz"
    old_receipt = json.loads(
        (ROOT / "analysis_outputs/evaluation_start_receipt.json").read_text()
    )
    assert old_receipt["evaluation_sha256"] == sha(test_path)
    primary_receipt = json.loads((OUT / "decision_boundaries_receipt.json").read_text())
    assert primary_receipt["prediction_sha256"] == sha(pred_path)
    assert primary_receipt["policy_sha256"] == sha(
        OUT / "development_budget_policies.csv"
    )
    assert (
        primary_receipt["seed"] == POLICY_SEED
        and primary_receipt["bootstrap_reps"] == REPS
    )
    test = pd.read_csv(test_path, dtype={"id": str})
    dev = pd.read_csv(dev_path, dtype={"id": str})
    ids = pd.read_csv(
        ROOT / "analysis_private/evaluation_ids_private.csv", dtype={"id": str}
    )
    assert np.array_equal(ids.id, test.id) and np.array_equal(
        ids.outcome, test.outcome_any_use
    )
    y = test.outcome_any_use.to_numpy(dtype=int)
    disjoint = ~test.id.isin(set(dev.id)).to_numpy()
    assert len(test) == 1520 and disjoint.sum() == 879
    pred = np.load(pred_path)
    policy = pd.read_csv(OUT / "development_budget_policies.csv")
    primary = pd.read_csv(OUT / "budget_policy_loss_boundaries.csv")
    primary_families = pd.read_csv(OUT / "loss_boundary_inference_families.csv")
    weights = np.random.default_rng(POLICY_SEED).multinomial(
        len(test), np.ones(len(test)) / len(test), size=REPS
    )
    rows, families = [], []
    for algorithm in ("ridge", "hgb"):
        for available in (False, True):
            availability = "history_available" if available else "history_unavailable"
            fam = policy[
                (policy.algorithm == algorithm) & (policy.availability == availability)
            ]
            assert len(fam) == 31
            baseline = "P0" if available else algorithm + "::NONE"
            assert np.array_equal(pred[algorithm + "::NONE"], pred["prevalence"])
            full = (
                algorithm
                + "::"
                + ("_".join(["H"] + OPTIONAL) if available else "_".join(OPTIONAL))
            )
            contrasts = list(
                dict.fromkeys(
                    (k, ref)
                    for k in fam.chosen_model
                    for ref in (full, baseline)
                    if k != ref
                )
            )
            keys = list(
                dict.fromkeys(
                    [k for pair in contrasts for k in pair] + [baseline, full]
                )
            )
            index = {k: j for j, k in enumerate(keys)}
            for domain, mask in [
                ("whole", np.ones(len(test), dtype=bool)),
                ("disjoint", disjoint),
            ]:
                losses = np.column_stack(
                    [loss_vector(y[mask], pred[k][mask]) for k in keys]
                )
                means = losses.mean(axis=0)
                w = weights[:, mask].astype(float)
                w /= w.sum(axis=1, keepdims=True)
                boot = w @ losses
                contrast_means = np.array(
                    [means[index[k]] - means[index[ref]] for k, ref in contrasts]
                )
                contrast_boot = np.column_stack(
                    [boot[:, index[k]] - boot[:, index[ref]] for k, ref in contrasts]
                )
                se, q, valid = standardized_maximum(contrast_means, contrast_boot)
                ix = {pair: j for j, pair in enumerate(contrasts)}

                q_primary = float(
                    np.quantile(
                        np.max(contrast_means[None, :] - contrast_boot, axis=1), 0.95
                    )
                )
                f0 = primary_families[
                    (primary_families.algorithm == algorithm)
                    & (primary_families.history_available == available)
                    & (primary_families.domain == domain)
                ].iloc[0]
                assert len(contrasts) == f0.policy_unique_nonzero_contrasts
                assert np.isclose(
                    q_primary, f0.policy_critical_value, atol=1e-12, rtol=0
                )
                families.append(
                    {
                        "algorithm": algorithm,
                        "history_available": available,
                        "domain": domain,
                        "n": int(mask.sum()),
                        "contrasts": len(contrasts),
                        "nondegenerate_contrasts": int(valid.sum()),
                        "standardized_maximum_critical_value": q,
                        "primary_absolute_critical_value": q_primary,
                        "bootstrap_reps": REPS,
                        "seed": POLICY_SEED,
                        "minimum_nondegenerate_se": float(se[valid].min()),
                        "maximum_se": float(se.max()),
                        "method": "fixed-bootstrap-SE standardized maximum, one-sided 95%; not bootstrap-t",
                    }
                )
                for _, p in fam.iterrows():
                    key = p.chosen_model
                    original = primary[
                        (primary.algorithm == algorithm)
                        & (primary.history_available == available)
                        & (primary.domain == domain)
                        & (primary.budget == p.budget)
                    ].iloc[0]
                    result = {
                        **p.to_dict(),
                        "domain": domain,
                        "n": int(mask.sum()),
                        "baseline": baseline,
                        "full": full,
                        "family_contrasts": len(contrasts),
                        "standardized_critical_value": q,
                    }
                    for label, reference in [("full", full), ("benchmark", baseline)]:
                        if key == reference:
                            delta, standard_error, upper = 0.0, 0.0, 0.0
                        else:
                            j = ix[(key, reference)]
                            delta = float(contrast_means[j])
                            standard_error = float(se[j])
                            upper = delta + q * standard_error if valid[j] else delta
                        assert np.isclose(
                            delta, original[f"delta_vs_{label}"], atol=1e-12, rtol=0
                        )
                        result[f"delta_vs_{label}"] = delta
                        result[f"bootstrap_se_vs_{label}"] = standard_error
                        result[f"primary_upper95_vs_{label}"] = float(
                            original[f"upper95_vs_{label}"]
                        )
                        result[f"standardized_upper95_vs_{label}"] = upper
                    result["primary_superior_to_benchmark"] = bool(
                        original.upper95_vs_benchmark < 0
                    )
                    result["standardized_superior_to_benchmark"] = bool(
                        result["standardized_upper95_vs_benchmark"] < 0
                    )
                    result["primary_noninferior_margin_0.01"] = bool(
                        original.upper95_vs_full < 0.01
                    )
                    result["standardized_noninferior_margin_0.01"] = bool(
                        result["standardized_upper95_vs_full"] < 0.01
                    )
                    result["primary_dual_pass_margin_0.01"] = (
                        result["primary_superior_to_benchmark"]
                        and result["primary_noninferior_margin_0.01"]
                    )
                    result["standardized_dual_pass_margin_0.01"] = (
                        result["standardized_superior_to_benchmark"]
                        and result["standardized_noninferior_margin_0.01"]
                    )
                    result["status"] = (
                        "post-primary retrospective sensitivity; retain primary result; fixed fitted predictions"
                    )
                    rows.append(result)
    table = pd.DataFrame(rows)
    assert len(table) == 248
    table.to_csv(OUT / "standardized_policy_bounds.csv", index=False)
    pd.DataFrame(families).to_csv(OUT / "standardized_policy_families.csv", index=False)
    return (
        table,
        families,
        {
            "development_sha256": sha(dev_path),
            "evaluation_sha256": sha(test_path),
            "predictions_sha256": sha(pred_path),
            "development_policy_sha256": sha(OUT / "development_budget_policies.csv"),
            "primary_policy_bounds_sha256": sha(
                OUT / "budget_policy_loss_boundaries.csv"
            ),
        },
    )


def attribution_sensitivity():
    path = PRIVATE / "block_attribution_individual_private.npz"
    manifest = json.loads((OUT / "block_attribution_manifest.json").read_text())
    assert (
        manifest["bootstrap"]["seed"] == ATTRIBUTION_SEED
        and manifest["bootstrap"]["reps"] == REPS
    )
    data = np.load(path)
    bootstrap = data["evaluation_aggregate_bootstrap"]
    phi = data["evaluation_phi"]
    disjoint = data["evaluation_disjoint"]
    means = np.r_[phi.mean(axis=0), phi[disjoint].mean(axis=0)]
    assert bootstrap.shape == (REPS, 56) and means.shape == (56,)
    primary = pd.read_csv(OUT / "block_attribution_evaluation.csv")
    assert len(primary) == 56
    expected = []
    for domain in ("whole", "person_disjoint"):
        for key in data["column_keys"]:
            algorithm, hflag, block = str(key).split("::")
            expected.append((domain, algorithm, hflag == "H1", block))
    observed = [
        (r.domain, r.algorithm, bool(r.has_history), r.block)
        for _, r in primary.iterrows()
    ]
    assert expected == observed
    assert np.allclose(primary.shapley_log_loss_reduction, means, atol=1e-12, rtol=0)
    se, q, valid = standardized_maximum(means, bootstrap, two_sided=True)
    assert valid.all()
    assert np.allclose(se, primary.bootstrap_se, atol=1e-12, rtol=0)
    q_primary = float(
        np.quantile(np.max(np.abs(means[None, :] - bootstrap), axis=1), 0.95)
    )
    assert np.isclose(
        q_primary,
        manifest["bootstrap"]["critical_absolute_log_loss"],
        atol=1e-12,
        rtol=0,
    )
    table = primary.copy()
    table["standardized_joint95_low"] = means - q * se
    table["standardized_joint95_high"] = means + q * se
    table["standardized_critical_value"] = q
    table["standardized_joint95_sign"] = np.where(
        table.standardized_joint95_low > 0,
        "positive",
        np.where(table.standardized_joint95_high < 0, "negative", "not_resolved"),
    )
    table["sign_conclusion_changed"] = (
        table.standardized_joint95_sign != table.joint95_sign
    )
    table["sensitivity_status"] = (
        "post-primary retrospective fixed-bootstrap-SE maximum-statistic sensitivity, not bootstrap-t"
    )
    table.to_csv(OUT / "standardized_block_attribution_bounds.csv", index=False)
    family = {
        "effects": 56,
        "bootstrap_reps": REPS,
        "seed": ATTRIBUTION_SEED,
        "standardized_max_abs_critical_value": q,
        "primary_max_abs_absolute_critical_value": q_primary,
        "minimum_se": float(se.min()),
        "maximum_se": float(se.max()),
        "method": "two-sided fixed-bootstrap-SE standardized maximum-absolute band, single 56-effect family",
        "cached_bootstrap_sha256": sha(path),
        "primary_attribution_sha256": sha(OUT / "block_attribution_evaluation.csv"),
    }
    dump(OUT / "standardized_block_attribution_family.json", family)
    return table, family


def main():
    amendment = REV / "protocol_amendment_studentized.md"
    assert amendment.exists()
    policies, policy_families, hashes = policy_sensitivity()
    attribution, attribution_family = attribution_sensitivity()
    changed = policies[
        policies.primary_superior_to_benchmark
        != policies.standardized_superior_to_benchmark
    ]
    ni_primary = policies["primary_noninferior_margin_0.01"]
    ni_standardized = policies["standardized_noninferior_margin_0.01"]
    ni_changed = int((ni_primary != ni_standardized).sum())
    sign_changed = attribution[attribution.sign_conclusion_changed]
    report = {
        "status": "PASS",
        "amendment_sha256": sha(amendment),
        "code_sha256": sha(__file__),
        "policy_input_hashes": hashes,
        "policy_rows": len(policies),
        "policy_families": len(policy_families),
        "attribution_effects": len(attribution),
        "attribution_family": attribution_family,
        "policy_benchmark_superiority_changed_rows": len(changed),
        "attribution_sign_changed_rows": len(sign_changed),
        "policy_full_noninferiority_changed_margin_0.01": ni_changed,
        "policy_full_noninferiority_new_passes_margin_0.01": int(
            (~ni_primary & ni_standardized).sum()
        ),
        "policy_full_noninferiority_lost_passes_margin_0.01": int(
            (ni_primary & ~ni_standardized).sum()
        ),
        "primary_analysis_retained": True,
        "identical_primary_bootstrap_critical_values_reproduced": True,
        "naming": "fixed-bootstrap-SE standardized maximum statistic; not replicate-studentized bootstrap-t",
        "limits": "retrospective sensitivity, fixed predictions; no training, selection, or adaptive-reuse uncertainty correction",
    }
    dump(OUT / "standardized_sensitivity_validation.json", report)
    lines = [
        "# 표준오차 표준화 최대통계량 민감도 결과",
        "",
        "비학생화 주 결과를 확인한 후 추가한 후향적 민감도이다. 기존 주 결과를 유지한다. 반복마다 표준오차를 재추정하는 bootstrap-t가 아니다.",
        "",
        "## 정책 비교",
        "",
        "| 학습계열 | 이력 가용 | 집단 | 기준선 우월 예산 수: 주/민감도 | 이중조건 δ=.01 통과 예산 수: 주/민감도 |",
        "|---|---|---|---:|---:|",
    ]
    for (algo, available, domain), g in policies.groupby(
        ["algorithm", "history_available", "domain"]
    ):
        lines.append(
            f'| {algo} | {available} | {domain} | {int(g.primary_superior_to_benchmark.sum())}/{int(g.standardized_superior_to_benchmark.sum())} | {int(g["primary_dual_pass_margin_0.01"].sum())}/{int(g["standardized_dual_pass_margin_0.01"].sum())} |'
        )
    lines += [
        "",
        "정책 기준선 우월성 판정이 바뀐 행 수: " + str(len(changed)),
        f"전체정보 대비 비열등성(δ=.01) 판정 변경: {ni_changed}행. 새 통과 {int((~ni_primary&ni_standardized).sum())}행, 기존 통과에서 미확인으로 변경 {int((ni_primary&~ni_standardized).sum())}행. 기준선 우월성과 결합한 이중조건 결론은 별도로 확인해야 한다.",
        "",
        "## 정보블록 귀속",
        "",
        f"56개 귀속 중 부호 판정 변경: {len(sign_changed)}개.",
        "",
        "| 집단 | 학습계열 | H 고정 | 블록 | 귀속 | 주 판정 | 표준화 판정 | 표준화 95% 구간 |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for _, r in sign_changed.iterrows():
        lines.append(
            f"| {r.domain} | {r.algorithm} | {r.has_history} | {r.block} | {r.shapley_log_loss_reduction:.6f} | {r.joint95_sign} | {r.standardized_joint95_sign} | [{r.standardized_joint95_low:.6f}, {r.standardized_joint95_high:.6f}] |"
        )
    lines += [
        "",
        "전체56개 효과와 모든248개 예산·집단 행을 CSV에 함께 제공한다. 일부 사양의 통과만을 확정 수집권고로 사용하지 않는다.",
    ]
    (OUT / "standardized_sensitivity_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "policy_superiority_changed": len(changed),
                "attribution_sign_changed": len(sign_changed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
