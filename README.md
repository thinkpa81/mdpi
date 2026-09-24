# JTAER research reproducibility materials

**Customer Data Collection Volume and Incremental Predictive Value for AI-Enabled Consumer Service Use: A Temporal Evaluation**

This repository contains the analysis code, aggregate results, input schema without observations, Figure 1, and final Figures 2 and 3 supporting the manuscript and supplementary materials. The analysis uses the KISDI Panel Survey of Intelligent Information Society Users, 2023–2025. The development sample contains 1,565 participants; temporal evaluation contains 1,520 participants, including 879 who do not overlap with development.

The publication cleanup of 24 September 2026 preserves the statistical code and reported aggregate estimates. Duplicate table caches and internal code-maintenance reports were archived outside this repository. Historical selection receipts, analysis manifests, validation records, and code-adaptation hashes remain as provenance. Their dates and hashes refer to their original runs.

## Data access and publication scope

Only aggregate research outputs and metadata are distributed. Source microdata, participant-level derived data, identifiers, individual predictions, participant-specific resampling multiplicities, fold assignments, and fitted model objects are excluded. Access to source data must be obtained separately from the provider under its applicable terms. The input schema does not provide source-data access or synthetic observations.

The source Word documents and the private `removed/` archive are not part of this repository. The archive is excluded by `.gitignore`. Do not upload it through a browser or API: those interfaces do not enforce `.gitignore` automatically.

## Repository map

| Path | Purpose | Manuscript support |
|---|---|---|
| `01_required_prior_analysis/` | Harmonization, candidate selection, nested validation, item budgets, block attribution, inference, historical aggregate results | Main Tables 1–5; Supplementary S1–S21 and related analyses |
| `00_current_revision_20260909/` | Precision, ridge-family sensitivity, dependence diagnostics | Main Table 4; Supplementary S22–S26 |
| `analysis_20260916/` | Auxiliary metrics, original representative-policy refit records, monthly/weekly outcome sensitivity, measurement diagnostics | Supplementary S28–S37 |
| `figures_final/` | Figure 1 extracted from the manuscript; full-precision aggregate inputs, plotting code, final PNG/SVG figures, and source audit for Figures 2 and 3 | Figures 1–3 |
| `INPUT_SCHEMA_NO_OBSERVATIONS.json` | Required input fields without any participant records | Supplementary S27 |
| `ENVIRONMENT_RECORDED.json` and `ADAPTATION_LOG.json` | Recorded environments and code provenance | Reproducibility documentation |
| `PACKAGE_INVENTORY.json` and `MANIFEST_SHA256.json` | Current file inventory and SHA-256 integrity | This release |
| `PUBLICATION_CLEANUP.json` | Archived-file list, duplicate-to-retained-file map, scope of this cleanup | This release |
| `README_분석재현안내.md` | Detailed Korean execution instructions and scientific qualifications | Complete execution order |

Table 6 and Figure 1 describe the proposed operational procedure and analysis design; they are not fitted statistical outputs. Figure 1 is provided as an exact PNG extraction from the final manuscript; its source is recorded in `figures_final/README.md`. Historical Korean table caches are retained only where they are unique or required by the table-generation code. Identical removed copies can be reconstructed from the retained paths listed in `PUBLICATION_CLEANUP.json`.

## Verify the package without private data

```bash
python verify_package.py --strict
```

This checks released-file hashes, Python syntax, and JSON parsing. It does not refit models or reproduce participant-level bootstraps. `.git/`, local Python environments and caches, and `removed/` are ignored by the integrity check.

## Recheck aggregates and reproduce Figures 2 and 3

The recorded analysis used Python 3.12.13/3.12.14. See `ENVIRONMENT_RECORDED.json` for the exact package records; the operating system, BLAS, and fonts are not fully locked.

```bash
python -m pip install -r 01_required_prior_analysis/code/requirements_analysis.txt
python analysis_20260916/code/extract_validate_metrics.py --output-dir /path/outside/repository/aggregate_checks
python figures_final/plot_figures.py --output-dir /path/outside/repository/reproduced_figures
```

An optional audit against separately held Word documents requires `python-docx`:

```bash
python -m pip install python-docx==1.2.0
python figures_final/audit_against_original_sources.py --main-docx /path/to/main.docx --supp-docx /path/to/supp.docx --output-dir /path/outside/repository/figure_audit
```

Use a working copy when regenerating historical outputs. Some historical validation scripts compare the exact code hashes recorded at the original execution; code formatting and portability changes have altered current file hashes. Preserve those checks and follow the execution order in the Korean guide to create a new consistent set of records. Do not run analysis scripts with `python -O` or `python -OO`.

## Full analysis with authorized inputs

Prepare the authorized `2023_numeric.csv`, `2024_numeric.csv`, and `2025_numeric.csv` under the ignored `analysis_20260916/private/raw/` directory, using the capitalization and columns in the schema. Measurement diagnostics additionally require the provider's variable and value dictionaries. Follow Sections 3 and 5 of `README_분석재현안내.md`; the aggregate outputs alone are insufficient for full model training.

## Interpretation and verification limits

Development out-of-fold scores used for selection are distinct from nested-validation estimates and temporal evaluation. Monthly and weekly outcome analyses are post hoc sensitivity analyses, not a new nested-validation experiment. The monthly full-sample analysis contains three jointly superior policies; the monthly non-overlapping and weekly analyses do not establish joint superiority. The changed 2025 wording and temporal change cannot be causally separated by these results.

The publication cleanup rechecked released-file integrity, aggregate metrics, figure inputs, and selected manuscript/supplementary values. It did not rerun the complete original selection, nested validation, or participant-level resampling pipeline. Historical PASS records describe their recorded runs. See `RELEASE_VALIDATION.json` for the checks actually performed for this release.

The manuscript names an earlier ZIP package dated 22 September 2026. This repository is the curated release dated 24 September 2026; its current manifest is authoritative for this version. It must not be described as byte-identical to the earlier ZIP. Record the repository commit identifier when citing the released code.

No new software license or rights to redistribute provider microdata are granted by this cleanup.
