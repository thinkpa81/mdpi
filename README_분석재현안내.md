# MDPI 분석코드 및 재현지원자료 — 2026년 9월 24일 공개 정리본

## 0. 현재 배포본 확인과 빠른 실행

이 폴더는 주 분석, 보충분석, 검증 이력 및 최종 영문 그림 재현자료를 함께 제공한다. 최소 실행파일만 모은 구성은 아니다. 실행 코드가 입력으로 사용하지 않는 동일 표 JSON 사본 37개와 내부 정비보고 2개는 removed로 이동하였다. 구성 변경 전 목록·해시 2개도 removed에 보관하고 현행 파일로 다시 작성하였다. 표 생성 코드가 읽는 원본과 역사적 분석 계보는 유지하였다. 이동 파일과 보존 원본의 대응은 PUBLICATION_CLEANUP.json을 따른다. removed는 GitHub 공개 대상에서 제외한다.

2026년 9월 23일 정비는 실행부 분리, 기본 경로·누락 입력 안내, 오류 시 임시 설정 복원, 검증 생략 방지 및 코드 서식 통일을 포함한다. 설명 주석과 독스트링은 추가하지 않았다. 아래 1~5절의 원자료 분석 수치와 실행환경은 2026년 9월 16일의 기록이며, 이번 코드 정비에서 전체 모형을 재학습한 결과가 아니다.

| 구성 | 용도 | 보존 이유 |
|---|---|---|
| `01_required_prior_analysis` | 전수 정보조합, 중첩검증, 예산정책, 정보블록 기여도와 기존 보충분석 | 후속 분석이 참조하는 코드·집계·고정 선택 기록 |
| `00_current_revision_20260909` | 고정 예측 정밀도·릿지 민감도 분석 | 본문 및 보충자료의 추가 추론 근거 |
| `analysis_20260916` | 대체 결과변수, 보조지표, 측정 진단 | 보충 S28~S37의 근거와 재현 코드 |
| `figures_final` | 최신 영문 Figure 2·3의 입력·렌더러·도판·출처 검증 | 최종 논문에 제시한 그림을 집계자료로 재생성 |
| 루트 JSON·Markdown·`verify_package.py` | 환경·입력 스키마·변경 이력·배포 무결성 | 실행 범위, 파일 계보 및 검증 결과 확인 |

폴더 최상위에서 다음 명령으로 개인단위 원자료 없이 배포파일을 검사할 수 있다. Python 표준 라이브러리만 사용하며 파일을 수정하지 않는다.

```bash
python verify_package.py
```

필요 라이브러리를 설치한 뒤 최신 그림을 별도 출력 경로에 재생성한다.

```bash
python -m pip install -r 01_required_prior_analysis/code/requirements_analysis.txt
python figures_final/plot_figures.py --output-dir regenerated_figures
```

`figures_final/data/plot_data.json`의 488개 값은 기존 집계 CSV와 일치한다. 영문 논문·보충자료를 별도로 제공하면 다음과 같이 문서 표와 원 집계값도 대조할 수 있다. 이 선택적 감사에는 `python-docx`가 필요하다.

```bash
python figures_final/audit_against_original_sources.py --main-docx /path/to/main.docx --supp-docx /path/to/supp.docx --output-dir regenerated_audit
```

저장된 집계 보조지표의 수치 검산은 다음 명령으로 수행한다. 지정 출력 폴더를 사용하므로 배포 결과를 덮어쓰지 않는다.

```bash
python analysis_20260916/code/extract_validate_metrics.py --output-dir rechecked_metrics
```

`verify_package.py`의 PASS는 현재 배포 파일의 해시·구문 검증이다. `validate_budget_policy.py` 등의 과거 실행 검사와 구별한다. 후자는 승인 원자료·개인 예측 캐시와 동일 실행 시점의 고정 기록이 있어야 한다. 현재 정비 코드와 과거 동결 해시의 차이를 없애기 위해 역사 기록을 수정하거나 검증 조건을 제거하지 않았다. 전체 분석을 새로 실행할 때는 작업 사본에서 새 코드로 생성한 고정 기록을 사용한다. 검증이 생략되는 `python -O` 또는 `python -OO`로 분석 코드를 실행하지 않는다.

## 1. 2026년 9월 16일 분석 수행 기록

2023~2025년 승인 원자료에서 분석자료를 다시 생성하였다. 개발 1,565명 및 후속 평가 1,520명의 파생 CSV 해시는 기존 기록과 일치하였다. 기존 대표 20개 정책을 고정 조합·고정 초모수로 다시 적합하고 전체 1,520명·참여자 비중복 879명의 보조지표 40행을 재계산하였다. 로그손실·ROC AUC·AP는 저장값과 동일했고 Brier·보정절편·기울기의 최대 차이는 1.11×10^-16이었다.

월 1회 이상(원 응답 1·2·3)과 주 1회 이상(1·2)을 각각 결과변수로 하는 사후 민감도 분석을 새로 실행하였다. 각 결과변수에서 기존 256개 정보조합과 동일 초모수 격자를 사용하여 층화 5겹 개발 교차검증을 수행하였다. 개발 선택 이후 31개 예산정책을 고정하고 두 평가집단에서 평가하였다. 2개 결과변수의 전체 정책 평가 496행과 대표예산 0·1·4·11·30의 80행을 모두 제공한다. 각 비교집합에서 고정 예측에 대한 부트스트랩 2,000회를 수행하였다.

월간 코딩의 전체 평가표본에서만 3개 정책(릿지 예산 15의 가용·미가용, HGB 가용 예산 16)이 기준모형 대비 공동 우월성을 보였다. 비중복 평가와 주간 코딩에서는 공동 우월성이 확인되지 않았다. 대표예산 밖의 세 결과도 별도 CSV와 요약에 공개하였다. 모든 코딩에서 무우월성이라는 결론을 내리지 않는다.

소비 부문 원응답 6개 범주의 분포, 세 표본의 임계값별 이용률, 동일 참여자 641명의 소비·여가 활용·업무 문항 비교도 원자료와 제공기관 변수·값 사전에 대조하여 산출하였다. 인접 서비스 역시 공통 지시문 변경을 공유하므로 영향을 받지 않은 대조군으로 해석하지 않는다.

기존 주 분석의 반복 중첩 교차검증과 모든 원 정책·한계기여의 전체 파이프라인은 이번에 재실행하지 않았다. 새 대체코딩의 개발 OOF는 선택용 결과이며 중첩 검증 결과가 아니다. 사후 민감도 분석은 측정문구 변경 효과와 시간 변화의 인과적 분리 또는 측정동일성의 입증이 아니다. H는 기존 이전 이용 여부를 유지하였다.

## 2. 제공·제외 자료와 환경

기존 통계·집계표 코드 17개와 모든 공개 집계 결과를 보존하고, 이번 추가분석 코드·프로토콜·집계표·검증 결과를 포함한다. 과거 DOCX·패키지 작성용 코드 8개는 제외하였다. 최종 Word 문서는 이 저장소의 코드로 자동 재생성되는 산출물이 아니다. 원자료, 참여자별 파생자료, 식별자 값, 개인 예측값, 개별 폴드, 모형 객체와 중복 실행 체크포인트는 포함하지 않는다.

실제 추가분석 환경은 Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, joblib 1.5.3, threadpoolctl 3.6.0이다. 기존 실행 Python은 3.12.13이며 나머지 주요 패키지는 같다. 운영체제·BLAS를 동결한 컨테이너 환경은 아니다. 설치환경을 다시 구성하는 전체 설치 실험까지 수행한 것은 아니다.

```bash
python -m pip install -r 01_required_prior_analysis/code/requirements_analysis.txt
```

Excel Numeric 시트를 직접 읽을 경우 openpyxl 3.1.5가 추가로 필요하다. 선택적 Word 표 대조만 python-docx 1.2.0을 사용한다. 통계 실행에는 Word 문서나 문서생성 패키지가 필요하지 않다.

## 3. 이번 보완분 재현 순서

저장소 작업 사본의 최상위를 현재 작업 디렉터리로 사용한다. 원 집계 결과를 보존하기 위해 분석은 작업 사본에서 실행한다. 승인받은 2023_numeric.csv, 2024_numeric.csv, 2025_numeric.csv를 analysis_20260916/private/raw/에 준비한다. 자료는 제공기관의 이용조건에 따라 별도로 확보해야 한다. 입력 열과 값의 정의는 INPUT_SCHEMA_NO_OBSERVATIONS.json 및 harmonize_data.py를 따른다.

```bash
python 01_required_prior_analysis/code/harmonize_data.py --input-dir analysis_20260916/private/raw --private-output analysis_20260916/private --report-dir analysis_20260916/outputs/harmonization
python analysis_20260916/code/extract_validate_metrics.py
python analysis_20260916/code/refit_original_representatives.py
python analysis_20260916/code/run_alternative_outcomes.py --jobs 6 --target all --source-root 01_required_prior_analysis
python analysis_20260916/code/summarize_alternative_outcomes.py
python analysis_20260916/code/audit_alternative_outcomes.py
python analysis_20260916/code/measurement_descriptive_diagnostics.py --base-dir analysis_20260916
```

마지막 기술진단 명령에는 Numeric CSV 외에 제공기관 변수·값 사전을 CSV로 저장한 2023_variables.csv·2023_values.csv, 2024_variables.csv·2024_values.csv, 2025_variables.csv·2025_values.csv가 같은 private/raw/ 폴더에 필요하다. 변수사전은 ‘변수명·변수설명’, 값사전은 ‘변수명·Value·Label’ 열을 갖는다. 실제 개인자료가 없는 사전 메타데이터의 확인 결과는 measurement_diagnostic_dictionary.json에 제공한다.

`--source-root`를 생략하면 현재 패키지의 `01_required_prior_analysis`를 사용한다. 다른 원분석 경로를 사용할 때만 명시적으로 지정한다. 실행 전에 필수 원자료·고정정책 파일 존재 여부를 검사한다. 모든 입력·출력은 코드 위치 또는 지정 경로 기준이다. 병렬 작업 수는 장비에 맞추어 조정한다. 한 작업 내부의 수치 연산 스레드는 1개로 제한한다.

extract_validate_metrics.py와 refit_original_representatives.py는 패키지 이동을 지원하도록 경로·입출력 인자를 일반화하였다. 계산식, 원모형 함수, 선택정책, 평가집단 및 비교 허용오차는 실제 실행 코드와 같다. 전자는 원자료 없이 저장된 집계 결과를 검산한다. 후자는 원자료에서 생성한 파생자료를 이용해 기존 대표정책만 재적합한다. 전체 원 분석의 선택과정을 재수행하는 명령이 아니다. 추가로 `--package-root`, `--output-dir` 등을 지정할 수 있다. 검증표의 재작성은 기존 집계값을 변경하는 정책 선택이 아니다.

## 4. 결과 파일과 해석

- analysis_20260916/alternative_outcome_protocol.md: 추가분석의 첫 적합 전에 작성한 사후 분석 사양.
- analysis_20260916/outputs/auxiliary_metrics/: 원 결과변수 대표정책 40행과 표 삽입용 JSON.
- analysis_20260916/outputs/original_representative_policy_refit_*.csv/json: 원자료에서 재적합한 대표정책의 지표·차이·환경 및 해시.
- analysis_20260916/outputs/alternative_outcomes/all_targets_policy_metrics.csv: 496행 전체 정책 평가·공동추론.
- 같은 폴더 all_targets_representative_metrics.csv: 80행 대표정책 보조지표.
- 같은 폴더 family_superiority_summary.csv/json 및 all_policies_with_joint_superiority.csv: 16개 비교집합과 우월 정책의 전체 공개.
- 같은 폴더 monthly/·regular/: 각 결과변수의 개발 선택, 모든 초모수 격자, 고정정책, 참조모형, 공동추론집합, 완료·해시 기록.
- analysis_20260916/outputs/outcome_frequency_distributions.csv·outcome_threshold_sample_rates.csv·paired_service_threshold_rates.csv: 6개 원응답 범주 분포, 임계값별 이용률, 동일인 인접 서비스 비교의 집계. measurement_diagnostic_*.json/md에는 사전 대조·검증·해석 범위를 기록한다.
- 01_required_prior_analysis/ 및 00_current_revision_20260909/: 기존 분석코드·집계 및 당시 검증 기록. 이곳의 PASS는 당시 기록이며 이번 전체 재실행의 증거가 아니다.

AP의 무정보 기준은 해당 집단의 양성률이다. 상이한 이진 코딩 간 AP·Brier·로그손실의 절대값으로 모형 우열을 직접 판단하지 않는다. NONE의 상수 예측은 보정 기울기가 식별되지 않는다. 보정 기울기는 C=1e8을 사용한 절편 포함 로지스틱 보정 진단이며, 평가자료로 원 예측모형을 재보정한 값이 아니다. 기존 엄격 수렴 검산값은 진단용으로 분리하였다.

공동추론은 결과변수·모형·이용이력·평가집단별 비교집합 안에서 적용한다. 비교집합 간 통합 오류율 제어나 재선택·재적합 불확실성을 포함하지 않는다. 같은 참여자와 예측을 공유하는 결과들을 독립 실험으로 취급하지 않는다.

## 5. 기존 주 분석 전체 재현이 필요한 경우

이번 보완분만 재현하려면 3절이면 충분하다. 기존 전수분석·반복 중첩 검증·부트스트랩까지 재생성하려면 아래 의존 순서를 사용한다. 이 경로는 계산량이 크며 이번 최종 보완에서 전체 재실행하지 않았다. 기존 기록과 같은 원자료와 직렬화가 필요하다. 해시·표본수 검사를 무조건 제거하지 않는다.

```bash
cd 01_required_prior_analysis
python code/harmonize_data.py --input-dir ../analysis_20260916/private/raw --private-output analysis_private --report-dir audit
python code/run_exhaustive.py develop --jobs 6
python code/run_exhaustive.py evaluate --jobs 6 --bootstrap 2000
python code/run_exhaustive.py refit --jobs 6 --refits 500
python code/run_itemized_sensitivity.py
python code/run_exploratory_shortlist.py
python code/run_exploratory_shortlist.py --disjoint
python code/run_selected_inference.py
python code/run_additional_contrasts.py
python code/validate_analysis.py
python revision_20260907/code/run_budget_policy.py develop --jobs 6
python revision_20260907/code/run_budget_policy.py evaluate --jobs 6
python revision_20260907/code/validate_budget_policy.py
python revision_20260907/code/run_block_attribution.py --input-private analysis_private --output revision_20260907/outputs --private-output revision_20260907/private --bootstrap-reps 2000 --seed 2026090717
python revision_20260907/code/run_decision_boundaries.py --policy revision_20260907/outputs/development_budget_policies.csv --bootstrap 2000 --seed 20260909
python revision_20260907/code/run_standardized_sensitivity.py
cd ..
python 00_current_revision_20260909/code/run_precision_analysis.py
python 00_current_revision_20260909/code/run_ridge_sensitivity.py
```

S16·S17의 후보 255개 대비 집합·시드 20260994는 주 예산정책 비교집합·시드 20260909와 구별한다. 상세 내용은 S16_S17_확인결과.md를 따른다. 평가 시점 재사용과 사후 사양의 제한은 원고와 함께 보고한다.

## 6. 무결성 및 공개 단위

MANIFEST_SHA256.json은 자기 자신과 removed를 제외한 현재 공개 파일의 크기와 SHA-256을 기록한다. PACKAGE_INVENTORY.json은 현재 공개 파일 목록과 CSV별 열·행·공개 단위를 기록한다. 이전 목록의 원본은 removed에 보존하였다. 원자료 없는 스키마는 관측치가 없는 입력 설명이다. 원자료 권한 또는 실행용 가상 관측치를 제공하지 않는다. 모든 CSV는 정책·모형·문항·범주·표본별 집계로 검토하였다.

코드 해시는 해당 시점의 실제 파일에 결부된다. ADAPTATION_LOG.json은 배포 코드의 경로·실행구조·서식 정비 및 변경 전후 해시를 기록한다. 내부 CODE_REVIEW_20260923 보고서 원본은 removed에 보관하였다. 주석 제거와 서식 정비로도 파일 해시는 달라진다. 과거 분석·선택 고정 기록의 원천 코드 해시와 실행 결과는 덮어쓰지 않았다. 과거 저장 결과와 현재 코드의 해시가 동일하다고 주장하지 않는다.

## 7. 2026년 9월 24일 공개 정리 범위

분석·통계 Python 코드는 변경하지 않았다. 배포 검사 코드 verify_package.py에 removed 제외만 추가하였다. README 및 S16·S17 안내의 현존 파일 경로를 갱신하고, 공개 목록·해시를 다시 작성하였다. 과거 검증·동결 기록은 당시 실행 근거로 보존하였다. 이번 검증은 공개 파일·집계값·문서 대응 확인이며 전체 모델 재학습을 수행한 것이 아니다. 상세 결과는 RELEASE_VALIDATION.json을 따른다.

현재 GitHub 정리본은 2026년 9월 24일 공개 정리본이다. 논문 제출 시 실제 공개 저장소와 커밋 식별자를 자료 이용 가능성 설명에 맞추어 기재한다.
