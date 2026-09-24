from pathlib import Path
import pandas as pd, json

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "analysis_outputs"
T = ROOT / "tables"


def save(name, rows, widths):
    (T / (name + ".json")).write_text(
        json.dumps({"rows": rows, "widths": widths}, ensure_ascii=False, indent=2)
    )


def f(x, n=4):
    return "—" if pd.isna(x) else f"{x:.{n}f}"


def model(s):
    return (
        s.replace("ridge_fixed::", "공통 릿지 ")
        .replace("ridge::", "릿지 ")
        .replace("hgb::", "HGB ")
        .replace("_", "+")
    )


def csvtable(
    source, name, cols, labels, widths, fmtcols=(), filterfn=None, maxrows=None
):
    df = pd.read_csv(source)
    df = filterfn(df) if filterfn else df
    if maxrows:
        df = df.head(maxrows)
    rows = [labels]
    for _, r in df.iterrows():
        rows.append(
            [
                (
                    f(r[c], 5)
                    if c in fmtcols
                    else str(r[c]).replace("True", "보유").replace("False", "미보유")
                )
                for c in cols
            ]
        )
    save(name, rows, widths)


import re


def main() -> None:
    T.mkdir(exist_ok=True)
    save(
        "main_blocks",
        [
            ["블록", "측정·이론상 역할", "원문항수", "운영상 구분 근거"],
            ["H", "이전 소비서비스 이용", "1", "과거 행동 기준; 습관 척도 아님 [8]"],
            [
                "D",
                "성별·연령대·학력·지역",
                "4",
                "이용환경의 맥락; 단일 잠재변수 아님 [2]",
            ],
            ["B", "소비서비스 기대혜택", "1", "프라이버시 계산의 혜택 측면 [1]"],
            [
                "M",
                "온라인 쇼핑몰 개인정보 관리평가",
                "1",
                "제공자에 대한 평가; 검증된 신뢰 척도 아님 [1,6]",
            ],
            [
                "P",
                "AI 추천서비스 제공자의 보호책무 기대",
                "11",
                "제도적 보장·통제에 대한 기대 [2,3,21]",
            ],
            [
                "C",
                "서비스 약관·개인정보 동의전략",
                "1",
                "정보 제공 관련 보고행동; P와 별도 [3,6]",
            ],
            ["A", "생성형 AI 경험", "1", "인접 AI 서비스 경험 [7,19,20]"],
            [
                "K",
                "자기보고 디지털 역량의 공통 문항",
                "11",
                "디지털 활용능력; 선행 척도와 동일시하지 않음 [9,10]",
            ],
            [
                "합계",
                "H 포함 8블록 / H 제외 7블록",
                "31 / 30",
                "H 보유 시 추가 수집 최대 30문항",
            ],
        ],
        [650, 3090, 900, 4720],
    )
    save(
        "main_sample",
        [
            ["구분", "개발 2023→2024", "평가 2024→2025"],
            ["표본 / 양성", "1,565 / 880", "1,520 / 853"],
            ["양성률", "56.23%", "56.12%"],
            ["이전 비이용자: 후속 비이용 / 이용", "344 / 255", "299 / 200"],
            ["이전 이용자: 후속 비이용 / 이용", "341 / 625", "368 / 653"],
            ["평가표본 중 개발 ID와 중복", "—", "641명(42.17%)"],
            ["평가표본 중 개발 ID와 비중복", "—", "879명(양성 471)"],
            ["유효 종단면 가중치 보유", "1,565", "1,369"],
        ],
        [3600, 2880, 2880],
    )
    save(
        "main_models",
        [
            ["요소", "실행 사양"],
            ["릿지 로지스틱", "L2, lbfgs, C=0.001·0.01·0.1·1·10, 최대 2,000회"],
            ["공통 사양 민감도", "모든 256개 조합에 C=0.1을 동일 적용"],
            [
                "HGB",
                "최대 잎 3·7, 학습률 0.05, 100회, 최소 잎 표본20, L2=1, 조기종료 없음",
            ],
            ["개발 폴드", "동일한 층화 5겹, seed=20260907, 참여자 ID 비중복"],
            ["전처리", "수치 중앙값 대치·표준화; 범주 최빈값 대치·원핫"],
            ["조합 선택", "개발 최소 로그손실의 1SE 이내에서 추가 원문항수 최소"],
            ["평가·추정", "시간순 평가; 고정예측 2,000회; 참여자 합집합 재적합 500회"],
        ],
        [2200, 7160],
    )
    E = pd.read_csv(O / "evaluation_all_subsets.csv").set_index("model")
    S = pd.read_csv(O / "selected_portfolios_evaluation.csv")
    S = S[(S.rule == "one_se") & (S.domain == "extended_K")]
    rows = [["상황 / 모형", "조합", "추가 문항", "로그손실", "AUC", "Brier"]]
    for _, r in S.iterrows():
        for k, label in [(r.model, "선택"), (r.reference, "전체")]:
            q = E.loc[k]
            rows.append(
                [
                    ("H보유" if r.has_history else "H미보유")
                    + " / "
                    + ("릿지" if r.algorithm == "ridge" else "HGB")
                    + " "
                    + label,
                    q.subset.replace("_", "+"),
                    str(int(q.additional_items)),
                    f(q.log_loss),
                    f(q.roc_auc),
                    f(q.brier),
                ]
            )
    for k in ["P0", "prevalence"]:
        q = E.loc[k]
        rows.append(
            [
                "행동기준" if k == "P0" else "무정보 기준",
                k,
                "0",
                f(q.log_loss),
                f(q.roc_auc),
                f(q.brier),
            ]
        )
    save("main_selected_performance", rows, [2100, 2300, 1000, 1320, 1320, 1320])
    I = pd.read_csv(O / "frozen_selected_four_family_inference.csv")
    I = I[I.domain == "extended_K"]
    rows = [
        [
            "개발선택 조합",
            "Δ 로그손실",
            "고정예측 공동상한",
            "재적합 공동상한",
            "δ=.01 판정 고정 / 재적합",
        ]
    ]
    for _, r in I.iterrows():
        rows.append(
            [
                model(r.candidate),
                f(r.delta_log_loss, 6),
                f(r.selected_family_joint_upper95_basic, 6),
                f(r.union_refit_joint_upper95_basic, 6),
                ("확인" if r["fixed_prediction_joint_NI_0.01"] else "미확인")
                + " / 미확인",
            ]
        )
    save("main_selected_inference", rows, [2500, 1400, 1700, 1700, 2060])
    Q = pd.read_csv(O / "post_evaluation_exploratory_shortlist.csv")
    Q = Q[Q.subset.isin(["K", "H_K"])]
    rows = [
        [
            "사양 / 조합",
            "추가 문항",
            "로그손실",
            "전체 대비 공동상한",
            "기준 대비 공동상한",
        ]
    ]
    for _, r in Q.iterrows():
        rows.append(
            [
                model(r.model),
                str(int(r.collection_items)),
                f(r.log_loss, 6),
                f(r.upper95_vs_full_joint, 6),
                f(r.upper95_vs_benchmark_joint, 6),
            ]
        )
    save("main_shortlist", rows, [2500, 1000, 1560, 2100, 2200])
    J = pd.read_csv(O / "itemized_full_model_sensitivity.csv")
    rows = [["구분", "H 보유 로그손실", "H 미보유 로그손실"]]
    for alg in ["ridge", "hgb"]:
        v = J[(J.algorithm == alg) & (J.domain == "extended_K")]
        rows.append(
            [
                ("릿지" if alg == "ridge" else "HGB") + " 원문항 전체",
                f(v[v.has_history].iloc[0].log_loss, 6),
                f(v[~v.has_history].iloc[0].log_loss, 6),
            ]
        )
    for key, label in [
        ("H_D", "기존 M1: H+D"),
        ("H_D_B", "기존 M2: H+D+B"),
        ("H_D_B_M_P_C_A", "기존 M5: K 제외 전체"),
        ("H_D_B_K", "기존 M6: H+D+B+K"),
    ]:
        rows.append(
            [label + " / C=.1", f(E.loc["ridge_fixed::" + key].log_loss, 6), "—"]
        )
    save("main_sensitivity", rows, [4160, 2600, 2600])
    save(
        "main_decision",
        [
            ["단계", "판단내용", "기록·산출물"],
            [
                "1. 목적·보유정보",
                "예측 사건, 고객군, H의 실제 이용 가능성 정의",
                "목적서·정보목록",
            ],
            [
                "2. 범주·후보",
                "이론상 역할과 실제 문항을 연결하고 후보공간 고정",
                "블록사전·조합 전체목록",
            ],
            [
                "3. 개발선택",
                "개발자료에서 조합·학습사양·간소화 규칙 선택",
                "동결 설정·개발예측·해시",
            ],
            [
                "4. 후속 평가",
                "전체 정보 대비 허용손실 및 기준선 대비 개선 확인",
                "쌍대 손실차·공동상한·표본수",
            ],
            [
                "5. 보류·후속검증",
                "불확실성이 크거나 평가 후 발견한 후보는 독립검증",
                "검증대상·성공/보류 기준",
            ],
            [
                "6. 수집 검토",
                "측정비용·민감도·목적 적합성·보유기간을 실제 확인",
                "수집승인 근거·주기적 재평가",
            ],
        ],
        [1700, 4200, 3460],
    )
    for has_h, label in [(True, "history"), (False, "nohistory")]:
        ids = (
            E[(E.algorithm == "ridge") & (E.has_history == has_h)]
            .sort_values(["raw_items", "subset"])
            .index
        )
        rows = [["정보조합", "원문항수", "튜닝 릿지 LL", "C=.1 LL", "HGB LL"]]
        for k in ids:
            q = E.loc[k]
            key = q.subset
            rows.append(
                [
                    key.replace("_", "+"),
                    str(int(q.raw_items)),
                    f(q.log_loss, 5),
                    f(E.loc["ridge_fixed::" + key].log_loss, 5),
                    f(E.loc["hgb::" + key].log_loss, 5),
                ]
            )
        save("supp_all_" + label, rows, [2900, 1000, 1820, 1820, 1820])
    csvtable(
        O / "extended_vs_original_full_paired_contrasts.csv",
        "supp_K_contrasts",
        ["candidate_extended", "delta_log_loss", "joint_upper95_basic"],
        ["K 포함 전체모형", "K 포함−제외 ΔLL", "6개 대비 공동상한"],
        [5100, 2130, 2130],
        ("delta_log_loss", "joint_upper95_basic"),
    )
    save(
        "supp_mapping",
        [
            ["측정", "2023", "2024", "2025"],
            ["ID", "ID", "id", "id"],
            ["이용 H / 결과 Y", "q1_8", "Q1_8", "q1_8"],
            ["선정: 구매 주저", "q15_6", "Q15_6", "q14_6"],
            ["기대혜택 B", "q3_8", "Q3_8", "q3_8"],
            ["관리평가 M", "q20_7", "q20_7", "q18_7"],
            ["보호책무 P", "q18_1–11", "q18_1–11", "q11_1–11"],
            ["동의전략 C", "q21", "q21", "q12"],
            ["생성형 AI A", "q23", "Q24", "q20"],
            ["디지털 역량 K", "q33_[공통번호]", "Q31_[공통번호]", "q29_[공통번호]"],
            ["성별·연령·학력·지역", "SQ1; Age_group; DQ2_1; Area1", "동일", "동일"],
            ["종단면 가중치", "WEIGHTw2_B2", "WEIGHTw3_B2", "WEIGHTw4_B2"],
        ],
        [2550, 2270, 2270, 2270],
    )
    v = pd.read_csv(ROOT / "audit/digital_item_wording_audit.csv")
    v = v[v.included_in_common11]
    rows = [["문항번호", "기준 문구(2024)", "대응 확인"]]
    for _, r in v.iterrows():
        rows.append(
            [
                str(r.item_position),
                re.sub("^Q31\\. 디지털 역량 수준 \\d+\\)\\s*", "", r.wording_2024),
                (
                    "정규화 후 동일"
                    if r.exact_after_question_number_whitespace_normalization
                    else "경미한 문구 차이; 의미 대응"
                ),
            ]
        )
    save("supp_digital", rows, [950, 6560, 1850])
    qual = json.loads((ROOT / "audit/scale_quality.json").read_text())
    save(
        "supp_quality",
        [["연도", "전체 n", "P α", "K α", "P·K 완전응답"]]
        + [
            [
                str(q["year"]),
                str(q["n"]),
                f(q["safeguard_alpha"]),
                f(q["digital_alpha"]),
                str(q["digital_complete_n"]),
            ]
            for q in qual
        ],
        [1500, 1900, 1700, 1700, 2560],
    )
    csvtable(
        O / "selected_subgroup_paired_intervals.csv",
        "supp_disjoint_selected",
        [
            "model",
            "delta_log_loss",
            "paired_percentile_95_low",
            "paired_percentile_95_high",
        ],
        ["개발선택 후보", "ΔLL", "주변95% 하한", "주변95% 상한"],
        [3900, 1820, 1820, 1820],
        ("delta_log_loss", "paired_percentile_95_low", "paired_percentile_95_high"),
        lambda d: d[
            (d.subgroup == "development_disjoint_id") & (d.domain == "extended_K")
        ],
    )
    csvtable(
        O / "selected_weighted_sensitivity.csv",
        "supp_weighted",
        ["model", "n", "log_loss", "roc_auc"],
        ["모형", "가중치 유효 n", "가중 LL", "가중 AUC"],
        [4260, 1500, 1800, 1800],
        ("log_loss", "roc_auc"),
        lambda d: d[
            d.model.isin(list(S.model) + list(S.reference) + ["P0", "prevalence"])
        ],
    )
    save(
        "supp_uncertainty_families",
        [
            ["분석", "공동 비교수", "재표집·해석"],
            ["개발선택 확장4", "4", "고정예측2,000·합집합재적합500; 주 보고"],
            ["개발선택 기존4", "4", "고정예측2,000; 별도 민감도"],
            ["전수조합 전체모형 대비", "확장127 / 기존63", "학습사양×이력×범위별 별도"],
            ["전수조합 이중기준", "확장255 / 기존127", "전체 대비+P0/무정보 대비 통합"],
            [
                "개인 비중복 전수감사",
                "확장255 / 기존127",
                "879명에서 별도 분석; 전체와 공동보장 없음",
            ],
            ["원문항 전체모형 민감도", "8", "평균값 표현에 대한 대조"],
            ["K 추가 전체모형 대조", "6", "두 튜닝계열+공통C×이력상황"],
            ["하위집단", "각 주변구간", "집단전체 다중비교 보장 없음"],
        ],
        [3000, 1700, 4660],
    )
    csvtable(
        O / "disjoint_all_subset_dual_reference_exploratory.csv",
        "supp_disjoint_candidates",
        ["model", "log_loss", "delta_loss_vs_benchmark", "upper95_vs_benchmark_joint"],
        ["사후 최소후보", "879명 LL", "기준 대비 ΔLL", "255대비 공동상한"],
        [3500, 1760, 1950, 2150],
        ("log_loss", "delta_loss_vs_benchmark", "upper95_vs_benchmark_joint"),
        lambda d: d[d.subset.isin(["K", "H_K"]) & (d.domain == "extended_K")],
    )
    print("tables prepared", len(list(T.glob("*.json"))))


if __name__ == "__main__":
    main()
