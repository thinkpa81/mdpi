from __future__ import annotations

if not __debug__:
    raise RuntimeError(
        "Run without -O or -OO so that research validation checks remain enabled."
    )
import argparse
import io
import json
from pathlib import Path
import re
import pandas as pd
from harmonize_data import SPECS, DIGITAL_INDICES, digest


def read_dictionary(path):
    parts = path.read_text(encoding="utf-8").split("\f")
    if len(parts) != 4:
        raise ValueError("Expected four extracted workbook sheets")
    v = pd.read_csv(io.StringIO(parts[2])).iloc[:, 1:]
    v.columns = ["variable", "description"]
    z = pd.read_csv(io.StringIO(parts[3])).iloc[:, 1:]
    z.columns = ["variable", "value", "label"]
    z.variable = z.variable.ffill()
    return v, z


def normalize_wording(x):
    return re.sub(r"\s+", "", re.sub(r"^Q\d+\.\s*", "", x))


def main():
    ap = argparse.ArgumentParser(
        description="Audit embedded questionnaire labels without exposing respondent records."
    )
    ap.add_argument("--extract-dir", type=Path, required=True)
    ap.add_argument("--report-dir", type=Path, required=True)
    ap.add_argument("--questionnaire-dir", type=Path)
    args = ap.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    all_vars = []
    all_values = []
    dictionaries = {}
    quality = []
    for year, spec in SPECS.items():
        p = args.extract_dir / f"{year}_extracted.txt"
        v, z = read_dictionary(p)
        dictionaries[year] = v.set_index("variable").description.to_dict()
        source_cols = (
            [
                spec["id"],
                "wgt_b",
                spec["weight"],
                "SQ1",
                "Age_group",
                "Area1",
                "DQ2_1",
                spec["use"],
                spec["benefit"],
                spec["hesitation"],
                spec["management"],
                spec["consent"],
                spec["genai"],
            ]
            + [spec["safeguards"] + str(i) for i in range(1, 12)]
            + [spec["digital"] + str(i) for i in DIGITAL_INDICES]
        )
        vv = v[v.variable.isin(source_cols)].copy()
        vv.insert(0, "year", year)
        all_vars.append(vv)
        zz = z[z.variable.isin(source_cols) & z.value.notna()].copy()
        zz.insert(0, "year", year)
        all_values.append(zz)
        timing = v[
            v.variable.str.contains(
                "date|timestamp|interview_date", case=False, regex=True
            )
            | v.description.str.contains(
                "조사일|면접일|응답일|시작일|종료일|타임스탬프", regex=True
            )
        ]
        commerce = v[v.description.str.contains("소비|쇼핑|구매", na=False)].copy()
        commerce.insert(0, "year", year)
        commerce.to_csv(
            args.report_dir / f"commerce_label_search_{year}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        entry = dict(
            year=year,
            variables=len(v),
            selected_variables=len(vv),
            respondent_timestamp_fields=timing.variable.tolist(),
            extract_sha256=digest(p),
        )
        if args.questionnaire_dir:
            current_dictionary = args.questionnaire_dir / f"{year}_dictionary.csv"
            if current_dictionary.exists():
                live = current_dictionary.read_text(encoding="utf-8")
                old = p.read_text(encoding="utf-8").split("\f")
                entry["current_drive_dictionary_sha256"] = digest(current_dictionary)
                entry["current_drive_variable_dictionary_exact_text_contained"] = (
                    old[2] in live
                )
                entry["current_drive_value_dictionary_exact_text_contained"] = (
                    old[3] in live
                )
                assert (
                    old[2] in live and old[3] in live
                ), "Current Drive dictionary differs from parsed source"
        quality.append(entry)
    rows = []
    for i in range(1, 25):
        row = {"item_position": i, "included_in_common11": i in DIGITAL_INDICES}
        texts = []
        for year in SPECS:
            source = SPECS[year]["digital"] + str(i)
            desc = dictionaries[year][source]
            row[f"wording_{year}"] = desc
            texts.append(normalize_wording(desc))
        row["exact_after_question_number_whitespace_normalization"] = (
            len(set(texts)) == 1
        )
        row["interpretation"] = (
            "near-equivalent wording; minor grammatical variation in remedy phrase"
            if i == 9
            else (
                "same wording after question-number/whitespace normalization"
                if row["exact_after_question_number_whitespace_normalization"]
                else "substantive wording change; excluded"
            )
        )
        rows.append(row)
    pd.concat(all_vars).to_csv(
        args.report_dir / "selected_variable_dictionary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(all_values).to_csv(
        args.report_dir / "selected_value_labels.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows).to_csv(
        args.report_dir / "digital_item_wording_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    questionnaires = []
    if args.questionnaire_dir:
        for year in SPECS:
            p = args.questionnaire_dir / f"{year}_questionnaire.txt"
            t = p.read_text(encoding="utf-8")
            questionnaires.append(
                dict(
                    year=year,
                    source_name=p.name,
                    sha256=digest(p),
                    recent_year_Q1=bool(re.search(r"Q1\..*?최근 1년간", t, re.S)),
                    commerce_examples="소비(쇼핑 어드바이저, 온라인 쇼핑 등)" in t,
                    interview_date_form_field="조사일시" in t,
                    expanded_consumer_service_explanation="8) 지능형(인공지능) 소비서비스"
                    in t,
                )
            )
    (args.report_dir / "measurement_manifest.json").write_text(
        json.dumps(
            {
                "source_scope": "embedded variable/value labels plus fetched full-questionnaire text extracts when provided; no individual fieldwork dates provided in analysis data",
                "waves": quality,
                "questionnaires": questionnaires,
                "common11_exact_wording_count": sum(
                    x["included_in_common11"]
                    and x["exact_after_question_number_whitespace_normalization"]
                    for x in rows
                ),
                "common11_near_equivalent_count": 1,
                "limitations": [
                    "No transaction amount, order conversion, retention or intervention outcome is measured by q1_8.",
                    "2025 adds a detailed consumer-service description and commercial examples; common item codes do not establish stable interpretation across waves.",
                    "Single-item temporal response stability is not internal-consistency reliability.",
                    "Equal-weight competence and provider-duty means are operational indices; alpha alone does not establish construct validity or longitudinal measurement invariance.",
                    "Information-block choices cover this declared feature universe, not all possible customer information.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote measurement audit and selected variable/value labels.")


if __name__ == "__main__":
    main()
