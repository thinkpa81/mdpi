from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )

import argparse
import json
from pathlib import Path

import pandas as pd

CODES = (1, 2, 3, 4, 5, 9)
THRESHOLDS = {"any_use": (1, 2, 3, 4), "monthly_use": (1, 2, 3), "regular_use": (1, 2)}
SERVICE_KEYS = {
    "leisure": ("q1_7", "여가 활용"),
    "consumer": ("q1_8", "소비"),
    "work": ("q1_9", "업무"),
}


def read_dictionary(raw_dir: Path, year: int):
    variables = pd.read_csv(raw_dir / f"{year}_variables.csv")
    values = pd.read_csv(raw_dir / f"{year}_values.csv")
    values["변수명"] = values["변수명"].ffill()
    variables["변수명"] = variables["변수명"].str.strip()
    id_rows = variables.loc[variables["변수명"].str.lower().eq("id")]
    assert len(id_rows) == 1 and id_rows.iloc[0]["변수설명"].strip().upper() == "ID"
    metadata = {"year": year, "id_field": id_rows.iloc[0]["변수명"], "services": {}}
    for key, (expected_name, expected_label) in SERVICE_KEYS.items():
        rows = variables.loc[variables["변수명"].str.lower().eq(expected_name)]
        assert len(rows) == 1
        name, label = str(rows.iloc[0]["변수명"]), str(rows.iloc[0]["변수설명"])
        assert "지능형 서비스 이용 빈도" in label and label.endswith(expected_label)
        lv = values.loc[values["변수명"].eq(name) & values["Value"].notna()]
        assert set(lv["Value"].astype(int)) == set(CODES) and len(lv) == len(CODES)
        metadata["services"][key] = {
            "field": name,
            "description": label,
            "service": expected_label,
            "value_labels": {
                str(int(row.Value)): str(row.Label).strip() for row in lv.itertuples()
            },
        }
    return metadata


def validate_id(df, label):
    assert df["id"].notna().all(), f"Missing identifier in {label}"
    assert not df["id"].duplicated().any(), f"Duplicate identifier in {label}"


def main(base: Path):
    private = base / "private"
    raw_dir = private / "raw"
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    metadata = {year: read_dictionary(raw_dir, year) for year in (2023, 2024, 2025)}
    label_sets = [
        m["services"][key]["value_labels"]
        for m in metadata.values()
        for key in SERVICE_KEYS
    ]
    assert all(x == label_sets[0] for x in label_sets), "Frequency value labels differ"
    development = pd.read_csv(private / "development.csv", dtype={"id": str})
    evaluation = pd.read_csv(private / "evaluation.csv", dtype={"id": str})
    for label, frame, expected_n, expected_year in (
        ("development", development, 1565, 2024),
        ("evaluation", evaluation, 1520, 2025),
    ):
        validate_id(frame, label)
        assert len(frame) == expected_n
        assert frame["outcome_year"].eq(expected_year).all()
        assert frame["outcome_frequency"].isin(CODES).all()
        for key, positive in THRESHOLDS.items():
            assert (
                frame[f"outcome_{key}"]
                .eq(frame["outcome_frequency"].isin(positive).astype(int))
                .all()
            )
    in_development = evaluation["id"].isin(development["id"])
    assert in_development.sum() == 641
    assert evaluation["seen_development_id"].eq(in_development.astype(int)).all()
    disjoint = evaluation.loc[~in_development].copy()
    assert len(disjoint) == 879
    overlap_ids = evaluation.loc[in_development, ["id"]].copy()
    raw = {}
    for year in (2024, 2025):
        md = metadata[year]
        frame = pd.read_csv(
            raw_dir / f"{year}_numeric.csv", dtype={md["id_field"]: str}
        )
        frame = frame.rename(columns={md["id_field"]: "id"})
        validate_id(frame, str(year))
        for spec in md["services"].values():
            assert spec["field"] in frame
        raw[year] = frame
        source = development if year == 2024 else evaluation
        field = md["services"]["consumer"]["field"]
        joined = source[["id", "outcome_frequency"]].merge(
            frame[["id", field]], how="left", on="id", validate="one_to_one"
        )
        assert (
            joined[field].notna().all()
            and joined["outcome_frequency"].eq(joined[field]).all()
        )

    overlap = development[["id", "outcome_frequency"]].merge(
        evaluation.loc[in_development, ["id", "prior_frequency"]],
        on="id",
        validate="one_to_one",
    )
    assert (
        len(overlap) == 641
        and overlap["outcome_frequency"].eq(overlap["prior_frequency"]).all()
    )

    distribution_rows = []
    group_rate_rows = []
    groups = (
        ("development", "개발표본", 2024, development),
        ("evaluation_full", "전체 평가표본", 2025, evaluation),
        ("evaluation_disjoint", "참여자 비중복 평가표본", 2025, disjoint),
    )
    for key, label, year, frame in groups:
        for code in CODES:
            count = int(frame["outcome_frequency"].eq(code).sum())
            distribution_rows.append(
                {
                    "sample": key,
                    "sample_label": label,
                    "outcome_year": year,
                    "frequency_code": code,
                    "frequency_label": label_sets[0][str(code)],
                    "n": len(frame),
                    "count": count,
                    "percent": 100 * count / len(frame),
                }
            )
        for threshold, positive in THRESHOLDS.items():
            count = int(frame["outcome_frequency"].isin(positive).sum())
            group_rate_rows.append(
                {
                    "sample": key,
                    "sample_label": label,
                    "outcome_year": year,
                    "threshold": threshold,
                    "positive_codes": ",".join(map(str, positive)),
                    "n": len(frame),
                    "positive_count": count,
                    "percent": 100 * count / len(frame),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        out / "outcome_frequency_distributions.csv", index=False
    )
    pd.DataFrame(group_rate_rows).to_csv(
        out / "outcome_threshold_sample_rates.csv", index=False
    )

    paired_rows = []
    for key, (_, service_label) in SERVICE_KEYS.items():
        matched = overlap_ids.copy()
        for year in (2024, 2025):
            field = metadata[year]["services"][key]["field"]
            matched = matched.merge(
                raw[year][["id", field]].rename(columns={field: f"frequency_{year}"}),
                on="id",
                how="left",
                validate="one_to_one",
            )
            assert matched[f"frequency_{year}"].isin(CODES).all()
        assert len(matched) == 641
        for threshold, positive in THRESHOLDS.items():
            y24 = matched["frequency_2024"].isin(positive)
            y25 = matched["frequency_2025"].isin(positive)
            c24, c25 = int(y24.sum()), int(y25.sum())
            n00, n01 = int((~y24 & ~y25).sum()), int((~y24 & y25).sum())
            n10, n11 = int((y24 & ~y25).sum()), int((y24 & y25).sum())
            assert n00 + n01 + n10 + n11 == 641
            assert c25 - c24 == n01 - n10
            paired_rows.append(
                {
                    "service": key,
                    "service_label": service_label,
                    "threshold": threshold,
                    "positive_codes": ",".join(map(str, positive)),
                    "n_paired": 641,
                    "positive_2024": c24,
                    "percent_2024": 100 * c24 / 641,
                    "positive_2025": c25,
                    "percent_2025": 100 * c25 / 641,
                    "change_percentage_points": 100 * (c25 - c24) / 641,
                    "n_0_to_0": n00,
                    "n_0_to_1": n01,
                    "n_1_to_0": n10,
                    "n_1_to_1": n11,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_service_threshold_rates.csv", index=False)
    (out / "measurement_diagnostic_dictionary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checks = {
        "development_n": 1565,
        "evaluation_n": 1520,
        "evaluation_disjoint_n": 879,
        "overlap_n": 641,
        "identifier_uniqueness": True,
        "overlap_flag_matches_id_intersection": True,
        "consumer_frequency_matches_raw_2024_and_2025": True,
        "overlap_2024_development_outcome_equals_evaluation_history": True,
        "frequency_value_labels_identical_2023_2024_2025": True,
        "all_thresholds_match_harmonized_binary_outcomes": True,
        "missing_selected_frequencies": 0,
        "rates_weighted": False,
        "pandas_version": pd.__version__,
        "participant_rows_exported": False,
        "analysis_status": "post_hoc_descriptive_measurement_diagnostic",
    }
    (out / "measurement_diagnostic_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(
        paired[
            [
                "service_label",
                "threshold",
                "positive_2024",
                "positive_2025",
                "change_percentage_points",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    main(parser.parse_args().base_dir)
