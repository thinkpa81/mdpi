#!/usr/bin/env python3


from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import os

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"
import numpy as np
import pandas as pd

OPTIONAL = ("D", "B", "M", "P", "C", "A", "K")
ALGORITHMS = ("ridge", "hgb")
LABELS = {
    "D": "인구통계 정보",
    "B": "소비서비스 생활 개선 기대",
    "M": "온라인 쇼핑몰 개인정보 관리 평가",
    "P": "추천서비스 제공자 책무 기대",
    "C": "개인정보 수집 동의 반응",
    "A": "생성형 AI 이용 경험",
    "K": "공통 디지털 역량 11문항",
}
SEED = 2026090717


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subset_name(mask: int, has_history: bool) -> str:
    blocks = (["H"] if has_history else []) + [
        b for j, b in enumerate(OPTIONAL) if mask & (1 << j)
    ]
    return "_".join(blocks) if blocks else "NONE"


def binary_log_loss(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be finite and aligned to outcome rows.")
    if np.any(prediction < 0) or np.any(prediction > 1):
        raise ValueError("Prediction outside [0, 1].")
    p = np.clip(prediction, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def exact_loss_shapley(losses: np.ndarray, n_blocks: int) -> np.ndarray:

    if losses.ndim != 2 or losses.shape[1] != 2**n_blocks:
        raise ValueError("Loss matrix does not contain every ordered subset.")
    phi = np.zeros((len(losses), n_blocks), dtype=float)
    denominator = math.factorial(n_blocks)
    for j in range(n_blocks):
        weight_sum = 0.0
        for mask in range(2**n_blocks):
            if mask & (1 << j):
                continue
            k = mask.bit_count()
            weight = math.factorial(k) * math.factorial(n_blocks - k - 1) / denominator
            phi[:, j] += weight * (losses[:, mask] - losses[:, mask | (1 << j)])
            weight_sum += weight
        assert np.isclose(weight_sum, 1.0, atol=1e-13)
    assert np.allclose(
        phi.sum(axis=1), losses[:, 0] - losses[:, -1], rtol=1e-10, atol=1e-12
    )
    return phi


def validate_known_game() -> None:

    additive = np.array([[1.0, 2.0, -1.0], [-2.0, 0.5, 4.0]])
    pair_bonus = np.array([6.0, -2.0])
    losses = np.repeat(np.array([25.0, 20.0])[:, None], 8, axis=1)
    for mask in range(8):
        for j in range(3):
            if mask & (1 << j):
                losses[:, mask] -= additive[:, j]
        if (mask & 1) and (mask & 2):
            losses[:, mask] -= pair_bonus
    expected = additive.copy()
    expected[:, :2] += pair_bonus[:, None] / 2
    assert np.allclose(exact_loss_shapley(losses, 3), expected, atol=1e-12)


@dataclass
class PhaseAttribution:
    columns: list[dict]
    phi: np.ndarray
    efficiency: list[dict]


def calculate_phase(
    frame: pd.DataFrame, predictions: np.lib.npyio.NpzFile, phase: str
) -> PhaseAttribution:
    y = frame.outcome_any_use.to_numpy(dtype=int)
    all_phi, columns, efficiency = [], [], []
    for algorithm in ALGORITHMS:
        for has_history in (True, False):
            keys = [
                algorithm + "::" + subset_name(mask, has_history) for mask in range(128)
            ]
            losses = np.column_stack(
                [binary_log_loss(y, predictions[key]) for key in keys]
            )
            phi = exact_loss_shapley(losses, 7)

            aggregate_phi = exact_loss_shapley(losses.mean(axis=0, keepdims=True), 7)[0]
            assert np.allclose(phi.mean(axis=0), aggregate_phi, atol=1e-12)
            all_phi.append(phi)
            for block in OPTIONAL:
                columns.append(
                    {
                        "algorithm": algorithm,
                        "has_history": has_history,
                        "block": block,
                        "block_label": LABELS[block],
                        "reference_model": keys[0],
                        "full_model": keys[-1],
                    }
                )
            efficiency.append(
                {
                    "phase": phase,
                    "algorithm": algorithm,
                    "has_history": has_history,
                    "n": len(frame),
                    "reference_model": keys[0],
                    "full_model": keys[-1],
                    "reference_log_loss": float(losses[:, 0].mean()),
                    "full_log_loss": float(losses[:, -1].mean()),
                    "full_loss_improvement": float(
                        (losses[:, 0] - losses[:, -1]).mean()
                    ),
                    "sum_block_attribution": float(phi.mean(axis=0).sum()),
                    "max_person_efficiency_abs_error": float(
                        np.max(np.abs(phi.sum(axis=1) - (losses[:, 0] - losses[:, -1])))
                    ),
                }
            )
    return PhaseAttribution(
        columns=columns, phi=np.column_stack(all_phi), efficiency=efficiency
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exact attribution of held-out predictive performance to seven information blocks.\n\nThis is a retrospective diagnostic of fitted subset-model performance. It is not\nmodel-output SHAP, a causal attribution, or information-theoretic mutual information.\nEvery subset model keeps its existing development-chosen hyperparameters. H is\nfixed present or absent; H itself is not assigned a Shapley value.\n\nExample:\n  python run_block_attribution.py --input-private analysis_private       --output revision_20260907/outputs --private-output revision_20260907/private\n\nOnly aggregate outputs may be distributed. Never publish private-output contents.\n"
    )
    ap.add_argument("--input-private", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("At least 1,000 replicates are required for this diagnostic.")
    args.output.mkdir(parents=True, exist_ok=True)
    args.private_output.mkdir(parents=True, exist_ok=True)
    validate_known_game()
    inputs = {
        name: args.input_private / name
        for name in [
            "development.csv",
            "evaluation.csv",
            "development_oof_predictions.npz",
            "evaluation_predictions_private.npz",
            "evaluation_ids_private.csv",
        ]
    }
    hashes_before = {name: sha256(path) for name, path in inputs.items()}
    dev = pd.read_csv(inputs["development.csv"], dtype={"id": str})
    eva = pd.read_csv(inputs["evaluation.csv"], dtype={"id": str})
    eid = pd.read_csv(inputs["evaluation_ids_private.csv"], dtype={"id": str})
    assert dev.id.is_unique and eva.id.is_unique
    assert len(dev) == 1565 and dev.outcome_any_use.sum() == 880
    assert len(eva) == 1520 and eva.outcome_any_use.sum() == 853
    assert np.array_equal(eva.id.to_numpy(), eid.id.to_numpy())
    assert np.array_equal(eva.outcome_any_use.to_numpy(), eid.outcome.to_numpy())
    disjoint = ~eva.id.isin(set(dev.id)).to_numpy()
    assert disjoint.sum() == 879
    if "seen_development_id" in eva.columns:
        assert np.array_equal(disjoint, eva.seen_development_id.eq(0).to_numpy())
    with np.load(inputs["development_oof_predictions.npz"]) as dev_predictions:
        dev_result = calculate_phase(dev, dev_predictions, "development_oof")
    with np.load(inputs["evaluation_predictions_private.npz"]) as eva_predictions:
        eva_result = calculate_phase(eva, eva_predictions, "evaluation")
    assert dev_result.columns == eva_result.columns

    means = np.concatenate(
        [eva_result.phi.mean(axis=0), eva_result.phi[disjoint].mean(axis=0)]
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_batches = []
    for start in range(0, args.bootstrap_reps, 100):
        count = min(100, args.bootstrap_reps - start)
        weights = rng.multinomial(len(eva), np.full(len(eva), 1 / len(eva)), size=count)
        whole_means = weights @ eva_result.phi / len(eva)
        disjoint_weights = weights[:, disjoint]
        subgroup_n = disjoint_weights.sum(axis=1)
        assert np.all(subgroup_n > 0)
        disjoint_means = (
            disjoint_weights @ eva_result.phi[disjoint] / subgroup_n[:, None]
        )
        bootstrap_batches.append(np.column_stack([whole_means, disjoint_means]))
    bootstrap = np.vstack(bootstrap_batches)
    assert bootstrap.shape == (args.bootstrap_reps, 56)
    critical = float(
        np.quantile(np.max(np.abs(bootstrap - means[None, :]), axis=1), 0.95)
    )
    development_rows = [
        {
            **meta,
            "phase": "development_oof",
            "domain": "development",
            "n": len(dev),
            "shapley_log_loss_reduction": float(dev_result.phi[:, j].mean()),
            "status": "descriptive; OOF folds reused for per-subset tuning, so optimistic",
        }
        for j, meta in enumerate(dev_result.columns)
    ]
    evaluation_rows = []
    for domain_index, (domain, mask) in enumerate(
        [("whole", np.ones(len(eva), dtype=bool)), ("person_disjoint", disjoint)]
    ):
        for j, meta in enumerate(eva_result.columns):
            col = domain_index * 28 + j
            mean = float(means[col])
            lo, hi = np.quantile(bootstrap[:, col], [0.025, 0.975])
            low_joint, high_joint = mean - critical, mean + critical
            dev_mean = development_rows[j]["shapley_log_loss_reduction"]
            evaluation_rows.append(
                {
                    **meta,
                    "phase": "evaluation",
                    "domain": domain,
                    "n": int(mask.sum()),
                    "shapley_log_loss_reduction": mean,
                    "paired_percentile95_low": float(lo),
                    "paired_percentile95_high": float(hi),
                    "joint_basic95_low": low_joint,
                    "joint_basic95_high": high_joint,
                    "joint95_sign": (
                        "positive"
                        if low_joint > 0
                        else ("negative" if high_joint < 0 else "not_resolved")
                    ),
                    "bootstrap_se": float(bootstrap[:, col].std(ddof=1)),
                    "joint_critical_absolute_log_loss": critical,
                    "family_size": 56,
                    "bootstrap_reps": args.bootstrap_reps,
                    "development_oof_attribution": dev_mean,
                    "evaluation_minus_development_attribution": mean - dev_mean,
                    "development_evaluation_point_sign_flip": bool(dev_mean * mean < 0),
                    "status": "retrospective exploratory diagnostic; fixed predictions; selection and training uncertainty excluded",
                }
            )
    dev_table = pd.DataFrame(development_rows)
    eva_table = pd.DataFrame(evaluation_rows)
    dev_table.to_csv(
        args.output / "block_attribution_development_descriptive.csv",
        index=False,
        encoding="utf-8-sig",
    )
    eva_table.to_csv(
        args.output / "block_attribution_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    efficiency = pd.DataFrame(dev_result.efficiency + eva_result.efficiency)

    for j in range(0, 28, 7):
        meta = eva_result.columns[j]
        subgroup_phi = eva_result.phi[disjoint, j : j + 7]
        total = float(subgroup_phi.mean(axis=0).sum())
        with np.load(inputs["evaluation_predictions_private.npz"]) as check_predictions:
            y_sub = eva.loc[disjoint, "outcome_any_use"].to_numpy(dtype=int)
            baseline_loss = binary_log_loss(
                y_sub, check_predictions[meta["reference_model"]][disjoint]
            )
            full_loss = binary_log_loss(
                y_sub, check_predictions[meta["full_model"]][disjoint]
            )
        direct_improvement = baseline_loss - full_loss
        assert np.allclose(subgroup_phi.sum(axis=1), direct_improvement, atol=1e-12)
        efficiency.loc[len(efficiency)] = {
            "phase": "evaluation_person_disjoint",
            "algorithm": meta["algorithm"],
            "has_history": meta["has_history"],
            "n": int(disjoint.sum()),
            "reference_model": meta["reference_model"],
            "full_model": meta["full_model"],
            "reference_log_loss": float(baseline_loss.mean()),
            "full_log_loss": float(full_loss.mean()),
            "full_loss_improvement": float(direct_improvement.mean()),
            "sum_block_attribution": total,
            "max_person_efficiency_abs_error": float(
                np.max(np.abs(subgroup_phi.sum(axis=1) - direct_improvement))
            ),
        }
    efficiency.to_csv(
        args.output / "block_attribution_efficiency_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        args.private_output / "block_attribution_individual_private.npz",
        development_ids=dev.id.to_numpy(dtype=str),
        evaluation_ids=eva.id.to_numpy(dtype=str),
        development_phi=dev_result.phi,
        evaluation_phi=eva_result.phi,
        evaluation_disjoint=disjoint,
        column_keys=np.array(
            [
                f"{m['algorithm']}::H{int(m['has_history'])}::{m['block']}"
                for m in eva_result.columns
            ]
        ),
        evaluation_aggregate_bootstrap=bootstrap,
    )
    assert hashes_before == {name: sha256(path) for name, path in inputs.items()}
    manifest = {
        "analysis": "Exact Shapley attribution of log-loss reduction for existing subset-learning procedures",
        "optional_blocks": list(OPTIONAL),
        "history_conditions": [True, False],
        "algorithms": list(ALGORITHMS),
        "input_sha256": hashes_before,
        "script_sha256": sha256(Path(__file__)),
        "value_function": "v_i(S)=loss_i(algorithm-specific H-only or NONE model)-loss_i(model using S plus fixed H when applicable)",
        "weights": "k! (6-k)! / 7! across all 64 subsets excluding the attributed block",
        "baseline_caution": "H-present reference is learner-specific H-only model, not the P0 benchmark",
        "positive_sign": "decrease in binary natural-log loss averaged over subset contexts",
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "seed": args.seed,
            "joint_family_effects": 56,
            "design": "paired multinomial bootstrap over all evaluation persons; same draws restricted and renormalized for disjoint subgroup",
            "interval": "two-sided centered unstudentized max-absolute basic bootstrap band",
            "critical_absolute_log_loss": critical,
        },
        "validation": {
            "known_additive_pair_interaction_game": True,
            "aggregate_individual_linearity": True,
            "all_person_efficiency_max_abs_error": float(
                efficiency.max_person_efficiency_abs_error.max()
            ),
            "original_input_hashes_unchanged": True,
            "evaluation_id_order_checked": True,
        },
        "limits": [
            "Retrospective exploratory analysis on previously examined evaluation outcomes.",
            "Intervals condition on already fitted predictions and do not refit training, tuning or subset selection.",
            "Per-subset tuned hyperparameters can differ; attribution belongs to learning procedures, not pure data content.",
            "Development OOF predictions were reused for tuning and are only descriptive.",
            "Temporal contrasts conflate sample composition, wave differences, wording changes and learning uncertainty.",
            "Attributions are not causal effects, mutual information, individual true gains or ordinary prediction SHAP.",
            "Joint band covers only the declared 56 effects under this bootstrap approximation, not all manuscript analyses.",
        ],
    }
    (args.output / "block_attribution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = [
        "# 정보블록 성능귀속 추가분석",
        "",
        "7개 선택 정보블록(D·B·M·P·C·A·K)의 128개 부분집합을 모두 이용하여 로그손실 감소에 대한 정확 Shapley 평균귀속을 계산하였다. H는 보유 또는 미보유 조건으로 고정한다. 양수는 다른 블록 조합들에 걸쳐 평균적으로 로그손실을 줄인 기여이며, 인과효과나 예측값에 대한 일반적인 SHAP 설명이 아니다.",
        "",
        f"평가 전체 1,520명과 개인 비중복 879명, 2개 학습기, 2개 H 조건, 7개 블록의 56개 효과를 하나의 동시구간 집합으로 정의하였다. 동일 개인 재표집 가중치를 두 집단에 공유한 {args.bootstrap_reps:,}회 부트스트랩의 95% 양측 max-absolute 기본구간 공통 반폭은 {critical:.6f}이다. 예측은 고정하며 재학습 불확실성을 포함하지 않는다.",
        "",
        "개발 OOF 결과는 부분집합별 튜닝에 같은 폴드를 다시 사용했으므로 낙관적일 수 있다. 개발·평가 간 귀속 차이는 기술적 비교이며 그 원인을 시간 변화로 식별하지 않는다.",
        "",
        "| 평가집단 | 학습기 | H 조건 | K 기여 | K 동시 95% 구간 | 양의 기여 확인 블록 | 음의 기여 확인 블록 |",
        "|---|---|---|---:|---|---|---|",
    ]
    for domain in ("whole", "person_disjoint"):
        for algorithm in ALGORITHMS:
            for history in (True, False):
                sub = eva_table[
                    (eva_table.domain == domain)
                    & (eva_table.algorithm == algorithm)
                    & (eva_table.has_history == history)
                ]
                kr = sub.loc[sub.block == "K"].iloc[0]
                pos = (
                    ", ".join(sub.loc[sub.joint95_sign == "positive", "block"])
                    or "없음"
                )
                neg = (
                    ", ".join(sub.loc[sub.joint95_sign == "negative", "block"])
                    or "없음"
                )
                summary.append(
                    f"| {domain} | {algorithm} | {'보유' if history else '미보유'} | {kr.shapley_log_loss_reduction:.6f} | [{kr.joint_basic95_low:.6f}, {kr.joint_basic95_high:.6f}] | {pos} | {neg} |"
                )
    summary.extend(
        [
            "",
            f"효율성 검증: 개인별 일곱 귀속의 합과 빈 조합 대비 전체 조합의 로그손실 감소 차이 최대 절댓값은 {manifest['validation']['all_person_efficiency_max_abs_error']:.3g}이다. 알려진 가산·상호작용 게임의 정답과도 일치하였다.",
            "",
            "해석 경계: 블록 기여는 모든 조합에 걸친 가중 평균이다. 양의 평균귀속은 해당 블록이 모든 조합에서 도움이 된다는 증명도, 그 블록을 반드시 수집해야 한다는 충분조건도 아니다. H 보유 시 기준은 해당 학습기의 H 단독 모형이며 기존 P0와 동일하지 않다. 입력별 정규화·튜닝 설정의 차이까지 포함한 학습절차의 성능을 분해한다.",
        ]
    )
    (args.output / "block_attribution_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(
        eva_table.loc[
            eva_table.block.eq("K"),
            [
                "domain",
                "algorithm",
                "has_history",
                "shapley_log_loss_reduction",
                "joint_basic95_low",
                "joint_basic95_high",
            ],
        ].to_string(index=False)
    )
    print(
        f'PASS: {len(eva_table)} evaluation effects; exact efficiency error {manifest["validation"]["all_person_efficiency_max_abs_error"]:.3g}; joint half-width {critical:.6f}'
    )


if __name__ == "__main__":
    main()
