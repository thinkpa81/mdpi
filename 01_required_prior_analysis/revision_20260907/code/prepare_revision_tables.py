#!/usr/bin/env python3

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
from pathlib import Path
import json, shutil, os

R = Path(__file__).resolve().parents[1]
BASE = R.parent
O = R / "outputs"
T = R / "tables"


def tab(name, rows, widths):
    assert sum(widths) == 9360 and all((len(row) == len(widths) for row in rows))
    (T / (name + ".json")).write_text(
        json.dumps({"rows": rows, "widths": widths}, ensure_ascii=False, indent=2)
    )


def f(x):
    return f"{float(x):.4f}"


def signed(x):
    return f"{float(x):+.4f}"


def combo(x):
    return str(x).replace("_", "+")


def label(a):
    return "릿지" if a == "ridge" else "HGB"


def main() -> None:
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[k] = "1"
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T.mkdir(exist_ok=True)
    for name in (
        "main_blocks",
        "main_sample",
        "supp_mapping",
        "supp_digital",
        "supp_quality",
        "main_selected_inference",
        "supp_K_contrasts",
        "main_shortlist",
        "supp_disjoint_candidates",
        "main_sensitivity",
        "supp_weighted",
    ):
        src = (BASE / "tables" / (name + ".json")).read_text()
        if name not in (
            "main_blocks",
            "main_sample",
            "supp_mapping",
            "supp_digital",
            "supp_quality",
        ):
            src = src.replace("미보유", "제외").replace("보유", "고정 포함")
        (T / (name + ".json")).write_text(src)
    tab(
        "design",
        [
            ["단계", "자료·범위", "평가 목적"],
            [
                "정보 후보",
                "H 가용: 256조합+P0; H 미가용: 128조합",
                "가용한 H도 사용하지 않을 수 있음",
            ],
            [
                "문항수 제약",
                "추가 원문항 0–30; 예산당 최저 개발손실",
                "31예산×2가용성×2학습기=124정책",
            ],
            [
                "중첩 검증",
                "외부5겹×2반복, 각 내부5겹",
                "전처리·튜닝·조합 선택을 포함한 개발 성능",
            ],
            [
                "최종 선택",
                "전체 개발1,565명의 5겹 OOF",
                "중첩 결과 확인 전 예산별 조합 동결",
            ],
            [
                "시간순 평가",
                "후속1,520명; 비중복879명",
                "전체정보 및 단순 기준과의 손실차",
            ],
            [
                "정책 구간",
                "2,000회 개인 부트스트랩",
                "학습기·가용성·집단별 고유 대비 공동상한",
            ],
            [
                "블록 귀속",
                "7블록, 2학습기, H고정/제외",
                "56개 효과 공동구간; 비용·인과효과와 구분",
            ],
        ],
        [1500, 4050, 3810],
    )
    p = pd.read_csv(O / "budget_policy_loss_boundaries.csv")
    n = pd.read_csv(O / "nested_budget_summary.csv")
    d = pd.read_csv(O / "block_attribution_development_descriptive.csv")
    s = pd.read_csv(O / "block_attribution_evaluation.csv")
    for avail in (True, False):
        rows = [
            [
                "학습기",
                "예산",
                "선택 조합\n(실제 문항)",
                "개발\nOOF",
                "중첩\n검증",
                "후속\n평가",
            ]
        ]
        for a in ("ridge", "hgb"):
            for b in (0, 1, 4, 11, 30):
                r = p[
                    (p.algorithm == a)
                    & (p.history_available == avail)
                    & (p.domain == "whole")
                    & (p.budget == b)
                ].iloc[0]
                nr = n[n.policy_id == r.policy_id].iloc[0]
                rows.append(
                    [
                        label(a),
                        str(b),
                        combo(r.chosen_subset) + f" ({int(r.selected_items)})",
                        f(r.development_oof_log_loss),
                        f(nr.nested_log_loss_mean),
                        f(r.log_loss),
                    ]
                )
        tab(
            "budget_" + ("available" if avail else "unavailable"),
            rows,
            [750, 650, 3160, 1600, 1600, 1600],
        )
    rows = [
        [
            "학습기·H 가용성",
            "예산30\n다른 조합 수",
            "최빈 조합\n선택 횟수",
            "개발 최종조합\n선택 횟수",
        ]
    ]
    for a in ("ridge", "hgb"):
        for av in ("history_available", "history_unavailable"):
            nr = n[(n.algorithm == a) & (n.availability == av) & (n.budget == 30)].iloc[
                0
            ]
            rows.append(
                [
                    label(a)
                    + "·"
                    + ("가용" if av == "history_available" else "미가용"),
                    str(nr.selection_unique_subsets),
                    combo(nr.selection_modal_subset)
                    + f" ({int(nr.selection_modal_count_out_of10)}/10)",
                    str(nr.selection_matches_final_count_out_of10) + "/10",
                ]
            )
    tab("selection_stability", rows, [2200, 1600, 3560, 2000])
    rows = [
        [
            "학습기·H",
            "예산",
            "전체정보 대비\n상한",
            "기준 대비\n차이",
            "기준 대비\n상한",
        ]
    ]
    for a in ("ridge", "hgb"):
        for av in (True, False):
            for b in (1, 11, 30):
                r = p[
                    (p.algorithm == a)
                    & (p.history_available == av)
                    & (p.domain == "whole")
                    & (p.budget == b)
                ].iloc[0]
                rows.append(
                    [
                        label(a) + "·" + ("가용" if av else "미가용"),
                        str(b),
                        signed(r.upper95_vs_full),
                        signed(r.delta_vs_benchmark),
                        signed(r.upper95_vs_benchmark),
                    ]
                )
    tab("policy_bounds", rows, [2360, 800, 2100, 2050, 2050])
    for a in ("ridge", "hgb"):
        for av in (True, False):
            g = p[
                (p.algorithm == a) & (p.history_available == av) & (p.domain == "whole")
            ].sort_values("budget")
            rows = [
                [
                    "예산",
                    "선택 조합\n(실제 문항)",
                    "중첩\n손실",
                    "후속\n손실",
                    "전체 대비\n상한",
                    "기준 대비\n상한",
                ]
            ]
            for _, r in g.iterrows():
                nr = n[n.policy_id == r.policy_id].iloc[0]
                rows.append(
                    [
                        str(r.budget),
                        combo(r.chosen_subset) + f" ({int(r.selected_items)})",
                        f(nr.nested_log_loss_mean),
                        f(r.log_loss),
                        signed(r.upper95_vs_full),
                        signed(r.upper95_vs_benchmark),
                    ]
                )
            tab(f"supp_budget_{a}_{int(av)}", rows, [600, 2800, 1490, 1490, 1490, 1490])
    inf = pd.read_csv(O / "loss_boundary_inference_families.csv")
    rows = [
        ["학습기·H", "표본", "정책 대비수", "정책 임계값", "전체후보 수", "쌍대 임계값"]
    ]
    for _, r in inf.iterrows():
        rows.append(
            [
                label(r.algorithm)
                + "·"
                + ("가용" if r.history_available else "미가용"),
                "전체" if r.domain == "whole" else "비중복",
                str(r.policy_unique_nonzero_contrasts),
                f(r.policy_critical_value),
                str(r.candidates),
                f(r.pairwise_critical_value),
            ]
        )
    tab("inference_families", rows, [2200, 1100, 1400, 1600, 1460, 1600])
    print("ATTRIBUTION_COLUMNS", list(d.columns), list(s.columns))
    rows = [["학습기·H조건", "블록", "개발\n귀속", "후속\n귀속", "후속 공동95%구간"]]
    for a in ("ridge", "hgb"):
        for h in (True, False):
            for block in ("D", "A", "K"):
                r = s[
                    (s.algorithm == a)
                    & (s.has_history == h)
                    & (s.domain == "whole")
                    & (s.block == block)
                ].iloc[0]
                rows.append(
                    [
                        label(a) + "·" + ("고정" if h else "제외"),
                        block,
                        signed(r.development_oof_attribution),
                        signed(r.shapley_log_loss_reduction),
                        f"[{signed(r.joint_basic95_low)}, {signed(r.joint_basic95_high)}]",
                    ]
                )
    tab("attribution_key", rows, [1900, 600, 1700, 1700, 3460])
    for domain in ("whole", "person_disjoint"):
        rows = [["학습기·H조건", "블록", "귀속", "공동95%구간", "해석"]]
        for _, r in s[s.domain == domain].iterrows():
            sig = {
                "positive": "양의 기여",
                "negative": "음의 기여",
                "not_confirmed": "미확인",
                "uncertain": "미확인",
            }.get(r.joint95_sign, "미확인")
            rows.append(
                [
                    label(r.algorithm) + "·" + ("고정" if r.has_history else "제외"),
                    r.block,
                    signed(r.shapley_log_loss_reduction),
                    f"[{signed(r.joint_basic95_low)}, {signed(r.joint_basic95_high)}]",
                    sig,
                ]
            )
        tab("supp_attribution_" + domain, rows, [1700, 700, 1550, 3610, 1800])
    tab(
        "decision_implications",
        [
            ["의사결정", "이번 자료에서 확인한 내용", "적용 범위"],
            [
                "추가질문 없는 출발점",
                "H 가용 시 P0, 미가용 시 무정보 예측을 비교기준으로 설정",
                "추가수집 불필요의 증명이 아니라 개선 근거를 확인할 출발점",
            ],
            [
                "적은 문항의 후보",
                "1문항 B는 두 가용성·학습기에서 선택되며 후속 손실 점추정치가 기준보다 낮음",
                "공동 우월성은 미확인; 즉시 적용 기준으로 확정하지 않음",
            ],
            [
                "11문항 K 후보",
                "평가 후 K 또는 H+K가 유망했지만 예산11 개발선택은 다른 조합",
                "후속 검증 후보. 11문항 최소성과 각 문항 필요성 미입증",
            ],
            [
                "수집 확대",
                "큰 예산의 개발·중첩 성능과 후속 평가가 일치하는지 확인",
                "개발 성능 개선만으로 D·A 등을 기본수집 항목으로 확대하지 않음",
            ],
            [
                "최소 문항수 결정",
                "예산별 손실차 상한과 단순 기준 우월성을 함께 제시",
                "이번 자료에서는 기준선 우월성까지 충족한 확정 예산 없음",
            ],
        ],
        [1650, 3910, 3800],
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 200,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    for ia, a in enumerate(("ridge", "hgb")):
        for ih, av in enumerate((True, False)):
            ax = axs[ia, ih]
            g = p[
                (p.algorithm == a) & (p.history_available == av) & (p.domain == "whole")
            ].sort_values("budget")
            ng = n[
                (n.algorithm == a)
                & (
                    n.availability
                    == ("history_available" if av else "history_unavailable")
                )
            ].sort_values("budget")
            ax.plot(
                g.budget,
                g.development_oof_log_loss,
                color="#577590",
                linestyle=":",
                label="Development OOF",
            )
            ax.plot(
                ng.budget,
                ng.nested_log_loss_mean,
                color="#43aa8b",
                linestyle="--",
                label="Nested validation",
            )
            ax.plot(
                g.budget,
                g.log_loss,
                color="#c8553d",
                linewidth=2,
                label="Temporal evaluation",
            )
            baseline = float(g.iloc[0].log_loss - g.iloc[0].delta_vs_benchmark)
            ax.axhline(
                baseline, color="#555555", linewidth=1, label="Temporal benchmark"
            )
            ax.set_title(
                a.upper()
                + " / "
                + ("History available" if av else "History unavailable"),
                fontsize=11,
            )
            ax.set_xlim(0, 30)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
            ax.grid(axis="y", alpha=0.18)
            ax.set_ylim(0.6, 0.77)
            if ia == 1:
                ax.set_xlabel("Additional raw-item budget")
            if ih == 0:
                ax.set_ylabel("Log loss (lower is better)")
    handles, labs = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labs,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(O / "figure_budget_validation.png", bbox_inches="tight")
    fig.savefig(O / "figure_budget_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Tables and budget figure ready")
    ss = pd.read_csv(O / "standardized_block_attribution_bounds.csv")
    rows = [["평가·학습기·H", "K 귀속", "주 공동95%구간", "표준화 공동95%구간"]]
    for _, r in ss[ss.block == "K"].iterrows():
        rows.append(
            [
                ("전체" if r.domain == "whole" else "비중복")
                + "·"
                + label(r.algorithm)
                + "·"
                + ("고정" if r.has_history else "제외"),
                signed(r.shapley_log_loss_reduction),
                f"[{signed(r.joint_basic95_low)}, {signed(r.joint_basic95_high)}]",
                f"[{signed(r.standardized_joint95_low)}, {signed(r.standardized_joint95_high)}]",
            ]
        )
    tab("standardized_K", rows, [2260, 1300, 2900, 2900])
    sp = pd.read_csv(O / "standardized_policy_bounds.csv")
    rows = [["평가·학습기·H", "1문항 후보", "주 기준상한", "표준화 기준상한"]]
    for _, r in sp[sp.budget == 1].iterrows():
        rows.append(
            [
                ("전체" if r.domain == "whole" else "비중복")
                + "·"
                + label(r.algorithm)
                + "·"
                + ("가용" if r.history_available else "미가용"),
                combo(r.chosen_subset),
                signed(r.primary_upper95_vs_benchmark),
                signed(r.standardized_upper95_vs_benchmark),
            ]
        )
    tab("standardized_policy", rows, [2760, 1600, 2500, 2500])
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True, sharey=True)
    blocks = ["D", "B", "M", "P", "C", "A", "K"]
    yy = np.arange(7)
    for ia, a in enumerate(("ridge", "hgb")):
        for ih, h in enumerate((True, False)):
            ax = axs[ia, ih]
            dd = (
                d[(d.algorithm == a) & (d.has_history == h)]
                .set_index("block")
                .loc[blocks]
            )
            ss = (
                s[(s.algorithm == a) & (s.has_history == h) & (s.domain == "whole")]
                .set_index("block")
                .loc[blocks]
            )
            ax.scatter(
                dd.shapley_log_loss_reduction,
                yy - 0.1,
                marker="s",
                s=25,
                color="#577590",
                label="Development OOF (descriptive)",
            )
            ax.errorbar(
                ss.shapley_log_loss_reduction,
                yy + 0.1,
                xerr=ss.joint_critical_absolute_log_loss,
                fmt="o",
                ms=4,
                color="#c8553d",
                elinewidth=1,
                capsize=2,
                label="Temporal / joint 95% interval",
            )
            ax.axvline(0, color="#777777", linewidth=0.8)
            ax.set_yticks(yy, blocks)
            ax.invert_yaxis()
            ax.set_title(
                a.upper() + " / " + ("H fixed" if h else "H excluded"), fontsize=11
            )
            ax.grid(axis="x", alpha=0.15)
            ax.set_xlim(-0.08, 0.055)
            if ia == 1:
                ax.set_xlabel("Attributed log-loss reduction")
    hh, ll = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        hh, ll, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(O / "figure_block_attribution.png", bbox_inches="tight")
    fig.savefig(O / "figure_block_attribution.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
