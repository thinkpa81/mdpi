#!/usr/bin/env python3

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import json
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs/alternative_outcomes"


def main() -> None:
    d = pd.read_csv(OUT / "all_targets_policy_metrics.csv")
    assert len(d) == 496
    group_keys = ["target", "domain", "algorithm", "availability"]
    summary = []
    for names, part in d.groupby(group_keys, sort=True):
        winning = part[part.joint_superior_to_baseline]
        summary.append(
            {
                **dict(zip(group_keys, names)),
                "policies": int(len(part)),
                "n": int(part.n.iloc[0]),
                "events": int(part.events.iloc[0]),
                "event_rate": float(part.event_rate.iloc[0]),
                "superiority_policy_count": int(len(winning)),
                "superiority_unique_model_count": int(winning.source_model.nunique()),
                "superiority_budgets": winning.budget.astype(int).tolist(),
                "superiority_models": winning.source_model.unique().tolist(),
                "min_log_loss": float(part.log_loss.min()),
                "max_log_loss": float(part.log_loss.max()),
                "min_roc_auc": float(part.roc_auc.min()),
                "max_roc_auc": float(part.roc_auc.max()),
                "min_delta_vs_baseline": float(part.delta_vs_baseline.min()),
                "min_joint_upper95_vs_baseline": float(
                    part.joint_upper95_vs_baseline.min()
                ),
                "family_unique_contrasts": int(part.family_unique_contrasts.iloc[0]),
                "joint_critical_value": float(part.joint_critical_value.iloc[0]),
            }
        )
    frame = pd.DataFrame(summary)
    frame.to_csv(OUT / "family_superiority_summary.csv", index=False)
    (OUT / "family_superiority_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    winners = d[d.joint_superior_to_baseline]
    winners.to_csv(OUT / "all_policies_with_joint_superiority.csv", index=False)
    report = [
        "# 대체 이진 코딩 민감도 분석 결과",
        "",
        "두 결과변수×2개 모형×2개 이용이력 가용성×31개 예산×2개 평가집단을 모두 보고하였다(496행). 대표예산은 결과 확인 전에 정한 0·1·4·11·30문항이다.",
        "",
        "## 핵심 결과",
        "",
        "월 1회 이상 코딩에서는 전체 평가표본에서 세 정책의 기준모형 대비 단측 95% 공동상한이 0 미만이었다. 해당 정책은 대표 예산 밖에 있으므로 아래에 별도로 보고한다. 비중복 879명에서는 같은 정책의 공동우월성이 확인되지 않았다. 주 1회 이상 코딩에서는 전체와 비중복 집단 모두 모든 정책에서 공동우월성이 확인되지 않았다.",
        "",
        "| 결과변수 | 집단 | 모형 | 이용이력 가용성 | 예산 | 선택조합 | 기준모형 대비 로그손실차 | 단측 95% 공동상한 |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for _, r in winners.iterrows():
        report.append(
            f"| {r.target} | {r.domain} | {r.algorithm} | {r.availability} | {int(r.budget)} | {r.selected_subset} | {r.delta_vs_baseline:.6f} | {r.joint_upper95_vs_baseline:.6f} |"
        )
    report.extend(
        [
            "",
            "## 해석 및 보고 범위",
            "",
            "이 결과는 원 코딩의 무우월성 결론을 모든 빈도 임계값에 일반화할 수 없음을 보여준다. 동시에 전체표본의 일부 결과가 비중복 표본에서 재현되지 않았으므로, 추가 정보수집을 일률적으로 정당화하는 근거로 해석하지 않는다. '차이가 없다' 또는 '추가 정보가 무용하다'는 결론도 지지하지 않는다.",
            "",
            "대체 코딩은 응답 임계값에 대한 민감도 분석이다. 2025년 문항 지시문과 예시의 변경 효과를 시간 변화와 분리하거나 측정동일성을 입증하지 않는다. 기존 H=이전 연도 소비 부문 서비스 이용 여부를 고정했으므로 이용이력 정의 변경에 대한 분석도 아니다.",
            "",
            "개발 OOF 성능은 이번 정책 선택에 사용되었으며 반복 중첩 교차검증을 다시 수행한 결과가 아니다. 2,000회 부트스트랩은 고정된 모형 예측에 조건부이며, 비교집합은 결과변수·모형·이용이력 가용성·평가집단별로 구분한다. 비교집합 간 통합 다중비교 보장 및 선택·재적합 불확실성을 주장하지 않는다.",
            "",
            "전체 31개 예산: all_targets_policy_metrics.csv. 대표예산: all_targets_representative_metrics.csv. 참조모형: all_targets_reference_metrics.csv. 비교집합별 우월성 집계: family_superiority_summary.csv/json. 실제 실행 버전 및 해시: completion_manifest.json 및 결과변수별 completion_manifest.json.",
        ]
    )
    (OUT / "manuscript_interpretation.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
