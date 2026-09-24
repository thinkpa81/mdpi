from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

DIGITAL_INDICES = (1, 3, 4, 6, 7, 8, 9, 10, 13, 14, 20)
SPECS = {
    2023: dict(
        id="ID",
        weight="WEIGHTw2_B2",
        use="q1_8",
        benefit="q3_8",
        hesitation="q15_6",
        management="q20_7",
        safeguards="q18_",
        consent="q21",
        genai="q23",
        digital="q33_",
    ),
    2024: dict(
        id="id",
        weight="WEIGHTw3_B2",
        use="Q1_8",
        benefit="Q3_8",
        hesitation="Q15_6",
        management="q20_7",
        safeguards="q18_",
        consent="q21",
        genai="Q24",
        digital="Q31_",
    ),
    2025: dict(
        id="id",
        weight="WEIGHTw4_B2",
        use="q1_8",
        benefit="q3_8",
        hesitation="q14_6",
        management="q18_7",
        safeguards="q11_",
        consent="q12",
        genai="q20",
        digital="q29_",
    ),
}
GROUPS = {
    "H": dict(
        label="과거 소비 부문 지능형 서비스 이용",
        numeric=["prior_any_use"],
        categorical=[],
        acquisition_items=1,
    ),
    "D": dict(
        label="인구통계 정보",
        numeric=[],
        categorical=["gender", "age_group", "education", "region"],
        acquisition_items=4,
    ),
    "B": dict(
        label="소비 부문 지능형 서비스의 생활 개선 기대",
        numeric=["benefit_expectation"],
        categorical=[],
        acquisition_items=1,
    ),
    "M": dict(
        label="온라인 쇼핑몰의 개인정보 관리 평가",
        numeric=["data_management"],
        categorical=[],
        acquisition_items=1,
    ),
    "P": dict(
        label="추천서비스 제공자 책무 기대",
        numeric=["safeguard_expectation"],
        categorical=[],
        acquisition_items=11,
    ),
    "C": dict(
        label="개인정보 수집 절차에 대한 동의 반응",
        numeric=[],
        categorical=["consent_strategy"],
        acquisition_items=1,
    ),
    "A": dict(
        label="생성형 AI 이용 경험",
        numeric=["genai_experience"],
        categorical=[],
        acquisition_items=1,
    ),
    "K": dict(
        label="디지털 역량 공통 11문항",
        numeric=["digital_competence_11"],
        categorical=[],
        acquisition_items=11,
    ),
}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_source(p):
    p = Path(p)
    if p.suffix.lower() == ".xlsx":
        data = pd.read_excel(p, sheet_name="Numeric")
    elif p.suffix.lower() == ".csv":
        data = pd.read_csv(p, low_memory=False)
    else:

        data = pd.read_csv(
            io.StringIO(p.read_text(encoding="utf-8").split("\f")[0]), low_memory=False
        )
    return data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed:")]


def allowed(x, codes):
    v = pd.to_numeric(x, errors="coerce")
    return v.where(v.isin(codes))


def alpha(frame):
    x = frame.dropna()
    k = x.shape[1]
    total = x.sum(axis=1).var(ddof=1)
    return (
        float(k / (k - 1) * (1 - x.var(ddof=1).sum() / total))
        if len(x) > 1 and total > 0
        else None
    )


def harmonize(raw, year):
    s = SPECS[year]
    idcol = raw[s["id"]]
    assert (
        idcol.notna().all() and idcol.is_unique
    ), f"{year}: incomplete or duplicate ID"
    c = pd.DataFrame({"id": idcol.astype(str), "year": year})
    for target, source, codes in [
        ("gender", "SQ1", [1, 2]),
        ("age_group", "Age_group", range(1, 8)),
        ("education", "DQ2_1", range(1, 9)),
        ("region", "Area1", range(1, 18)),
        ("use_frequency", s["use"], [1, 2, 3, 4, 5, 9]),
        ("benefit_expectation", s["benefit"], [1, 2, 3, 4]),
        ("hesitation_frequency", s["hesitation"], [1, 2, 3, 4, 9]),
        ("data_management", s["management"], [1, 2, 3, 4, 5]),
        ("consent_strategy", s["consent"], [1, 2, 3, 4]),
    ]:
        c[target] = allowed(raw[source], codes)
    c["genai_experience"] = allowed(raw[s["genai"]], [1, 2]).map({1: 1, 2: 0})
    c["any_use"] = (
        c.use_frequency.isin([1, 2, 3, 4]).astype(float).where(c.use_frequency.notna())
    )
    c["regular_use"] = (
        c.use_frequency.isin([1, 2]).astype(float).where(c.use_frequency.notna())
    )
    c["monthly_use"] = (
        c.use_frequency.isin([1, 2, 3]).astype(float).where(c.use_frequency.notna())
    )
    c["long_weight"] = pd.to_numeric(raw[s["weight"]], errors="coerce")
    c["cross_weight"] = pd.to_numeric(raw["wgt_b"], errors="coerce")
    sf = []
    dk = []
    for i in range(1, 12):
        col = f"safeguard_{i}"
        sf.append(col)
        c[col] = allowed(raw[s["safeguards"] + str(i)], range(1, 6))
    for i in DIGITAL_INDICES:
        col = f"digital_{i}"
        dk.append(col)
        c[col] = allowed(raw[s["digital"] + str(i)], range(1, 6))
    c["safeguard_expectation"] = (
        c[sf].mean(axis=1).where(c[sf].notna().sum(axis=1) >= 8)
    )
    c["digital_competence_11"] = (
        c[dk].mean(axis=1).where(c[dk].notna().sum(axis=1) >= 9)
    )
    quality = {
        "year": year,
        "n": len(c),
        "unique_ids": c.id.nunique(),
        "safeguard_alpha": alpha(c[sf]),
        "digital_alpha": alpha(c[dk]),
        "safeguard_complete_n": int(c[sf].notna().all(axis=1).sum()),
        "digital_complete_n": int(c[dk].notna().all(axis=1).sum()),
        "missing_n": c.drop(columns=["id"]).isna().sum().to_dict(),
    }
    return c, quality


def transition(baseline, outcome, baseline_year):
    fields = (
        [
            "id",
            "gender",
            "age_group",
            "education",
            "region",
            "use_frequency",
            "any_use",
            "regular_use",
            "monthly_use",
            "benefit_expectation",
            "data_management",
            "safeguard_expectation",
            "consent_strategy",
            "genai_experience",
            "digital_competence_11",
            "hesitation_frequency",
            "cross_weight",
        ]
        + [f"safeguard_{i}" for i in range(1, 12)]
        + [f"digital_{i}" for i in DIGITAL_INDICES]
    )
    b = baseline.loc[baseline.hesitation_frequency.isin([1, 2]), fields].rename(
        columns={
            "use_frequency": "prior_frequency",
            "any_use": "prior_any_use",
            "regular_use": "prior_regular_use",
            "monthly_use": "prior_monthly_use",
            "cross_weight": "baseline_cross_weight",
        }
    )
    o = outcome[
        [
            "id",
            "use_frequency",
            "any_use",
            "regular_use",
            "monthly_use",
            "long_weight",
            "cross_weight",
        ]
    ].rename(
        columns={
            "use_frequency": "outcome_frequency",
            "any_use": "outcome_any_use",
            "regular_use": "outcome_regular_use",
            "monthly_use": "outcome_monthly_use",
            "long_weight": "outcome_long_weight",
            "cross_weight": "outcome_cross_weight",
        }
    )
    x = (
        b.merge(o, on="id", how="inner", validate="one_to_one")
        .dropna(subset=["outcome_any_use"])
        .reset_index(drop=True)
    )
    x["baseline_year"] = baseline_year
    x["outcome_year"] = baseline_year + 1
    return x


def compare_frames(a, b, idcol):
    assert a[idcol].is_unique and b[idcol].is_unique
    aa = a.set_index(idcol).sort_index()
    bb = b.set_index(idcol).sort_index()
    common = list(bb.columns)
    assert aa.index.equals(bb.index), "ID sets differ"
    assert set(common).issubset(aa.columns), "comparison columns absent"
    aa = aa[common]
    exact_mismatch = 0
    tolerant_mismatch = 0
    max_numeric_diff = 0.0
    categorical_mismatch = 0
    for col in common:
        x = aa[col]
        y = bb[col]
        equal = (x == y) | (x.isna() & y.isna())
        exact_mismatch += int((~equal).sum())
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xx = x.to_numpy(dtype=float)
            yy = y.to_numpy(dtype=float)
            tolerant_mismatch += int(
                (~np.isclose(xx, yy, rtol=1e-12, atol=1e-12, equal_nan=True)).sum()
            )
            dif = np.abs(xx - yy)
            if np.isfinite(dif).any():
                max_numeric_diff = max(max_numeric_diff, float(np.nanmax(dif)))
        else:
            categorical_mismatch += int((~equal).sum())
            tolerant_mismatch += int((~equal).sum())
    return dict(
        rows=len(aa),
        columns=len(common) + 1,
        exact_cell_mismatches=exact_mismatch,
        tolerance_cell_mismatches=tolerant_mismatch,
        categorical_mismatches=categorical_mismatch,
        max_numeric_abs_difference=max_numeric_diff,
    )


def locate(directory, year):
    directory = Path(directory)
    candidates = [
        directory / f"{year}_numeric.csv",
        directory / f"{year}_extracted.txt",
    ] + list(directory.glob(f"{year}_*data*.xlsx"))
    return next(p for p in candidates if p.exists())


def main():
    ap = argparse.ArgumentParser(
        description="Reconstruct private analysis inputs from authorized KISDI 2023--2025 data.\n\nAll feature columns come from t; only outcome and descriptive survey weights\ncome from t+1. Never distribute row-level outputs as a public supplement.\nAccepts official Numeric-sheet Excel files, numeric CSV files, or lossless\nmulti-sheet text extracts. This script does not fit or select a model.\n"
    )
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--private-output", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--verify-extract-dir", type=Path)
    ap.add_argument("--verify-view-dir", type=Path)
    ap.add_argument(
        "--current-source-proof",
        type=Path,
        help="Optional current Drive Numeric-sheet export hash evidence",
    )
    args = ap.parse_args()
    args.private_output.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_policy": "Authorized files read only; row-level outputs private; no claim of byte identity with un-fetched Drive originals.",
        "waves": [],
        "comparison_tolerance": "rtol=atol=1e-12 only for numeric workbook serialization differences",
    }
    waves = {}
    qualities = []
    live = (
        {
            item["year"]: item
            for item in json.loads(args.current_source_proof.read_text())
        }
        if args.current_source_proof
        else {}
    )
    for year in SPECS:
        source = locate(args.input_dir, year)
        raw = read_source(source)
        entry = {
            "year": year,
            "source_name": source.name,
            "source_bytes": source.stat().st_size,
            "source_sha256": digest(source),
            "source_rows": len(raw),
            "source_columns": len(raw.columns),
        }
        if year in live:
            entry["current_drive_numeric_sha256"] = live[year]["sha256"]
            entry["current_drive_numeric_exact_byte_match"] = (
                digest(source) == live[year]["sha256"]
            )
            assert entry[
                "current_drive_numeric_exact_byte_match"
            ], "Current source Numeric-sheet export differs"
        if args.verify_extract_dir:
            other = locate(args.verify_extract_dir, year)
            entry["independent_text_comparison"] = compare_frames(
                raw, read_source(other), SPECS[year]["id"]
            )
            entry["text_sha256"] = digest(other)
            assert entry["independent_text_comparison"]["exact_cell_mismatches"] == 0
        if args.verify_view_dir:
            other = locate(args.verify_view_dir, year)
            entry["prior_view_comparison"] = compare_frames(
                raw, read_source(other), SPECS[year]["id"]
            )
            entry["view_sha256"] = digest(other)
            assert entry["prior_view_comparison"]["tolerance_cell_mismatches"] == 0
        waves[year], q = harmonize(raw, year)
        qualities.append(q)
        manifest["waves"].append(entry)
    dev = transition(waves[2023], waves[2024], 2023)
    eva = transition(waves[2024], waves[2025], 2024)
    overlap = set(dev.id) & set(eva.id)
    eva["seen_development_id"] = eva.id.isin(overlap).astype(int)
    for name, frame in [("development", dev), ("evaluation", eva)]:
        path = args.private_output / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        manifest[name] = {
            "n": len(frame),
            "events": int(frame.outcome_any_use.sum()),
            "prevalence": float(frame.outcome_any_use.mean()),
            "unique_ids": frame.id.nunique(),
            "output_sha256": digest(path),
            "valid_longitudinal_weight_n": int(frame.outcome_long_weight.notna().sum()),
            "model_columns_missing": frame[
                [v for g in GROUPS.values() for v in g["numeric"] + g["categorical"]]
            ]
            .isna()
            .sum()
            .to_dict(),
        }
    manifest["overlap"] = {
        "shared_ids": len(overlap),
        "evaluation_share": len(overlap) / len(eva),
        "evaluation_person_disjoint_n": int((eva.seen_development_id == 0).sum()),
        "evaluation_person_disjoint_events": int(
            eva.loc[eva.seen_development_id == 0, "outcome_any_use"].sum()
        ),
    }
    manifest["common_all_waves_ids"] = len(
        set(waves[2023].id) & set(waves[2024].id) & set(waves[2025].id)
    )
    manifest["source_weight_rule"] = (
        "outcome_long_weight is descriptive/evaluation-only and is never a model feature; 2025 missing weights are not filled with invented weights"
    )
    manifest["digital_rule"] = {
        "indices": list(DIGITAL_INDICES),
        "minimum_valid": 9,
        "scale": "self-reported competence 1-5; equal item mean; no latent invariance claim",
    }
    manifest["script_sha256"] = digest(__file__)
    if args.current_source_proof:
        manifest["current_source_proof_sha256"] = digest(args.current_source_proof)
        manifest["source_policy"] = (
            "Authorized source files read only; current Drive Numeric-sheet text export hash matches numeric CSV byte for byte. Original XLSX binary byte identity is not claimed. Row-level outputs private."
        )
    (args.report_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.report_dir / "feature_groups.json").write_text(
        json.dumps(GROUPS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.report_dir / "scale_quality.json").write_text(
        json.dumps(qualities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tables = []
    for name, f in [("development", dev), ("evaluation", eva)]:
        tab = (
            f.groupby(["prior_any_use", "outcome_any_use"])
            .size()
            .rename("n")
            .reset_index()
        )
        tab.insert(0, "sample", name)
        tables.append(tab)
    pd.concat(tables).to_csv(
        args.report_dir / "outcome_transition_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        json.dumps(
            {
                "development": manifest["development"],
                "evaluation": manifest["evaluation"],
                "overlap": manifest["overlap"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
