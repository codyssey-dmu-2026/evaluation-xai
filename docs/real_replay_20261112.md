# 실제 데이터·PPO·SHAP 연결 결과 — 2026-11-12

전달된 `rl-engine_data_snapshot.zip`의 피처·수익률·설정 파일을 읽어 실제 PPO를 재실행했다.
**4개 테스트 연도에서 대표 모델의 일별 결과가 기존 평가 CSV와 허용 오차 안에서 일치했고, 대표 의사결정 4건의 SHAP을 생성했다.**
60개 모델 전체 또는 모든 날짜의 설명을 검증했다는 의미는 아니다.

## 받은 파일과 검사 범위

- 위치: 프로젝트 상위 `data/processed/`.
- `features_window1~4.parquet`: 구간별 48개 피처. 1,273 / 1,531 / 1,791 / 2,050행.
- `returns.parquet`: 12개 자산의 정규화 전 로그수익률, 2,243행.
- `walk_forward_manifest.json`: 구간, 변수 순서, winsorize 경계·평균·표준편차.
- `SHA256.json`: 6개 입력 파일의 SHA-256 앞 16자리. 모두 일치했고, 출력에는 전체 SHA-256을 별도 기록했다. 동봉된 해시는 전달 중 변경 여부를 확인하는 자료이며 원 출처의 진위를 독립 인증하지 않는다.
- 각 구간의 수익률을 저장된 경계로 clip하고 저장된 평균·표준편차로 변환한 값이 모델 입력의 수익률 피처와 일치했다. 재학습하거나 scaler를 다시 fit하지 않았다.
- **`prices.parquet`는 없다.** 원본 가격과 수익률의 대조는 미실행이다. 가격을 임의로 복원하지 않았으며, 이 상태는 `input_verification`에 남긴다.

이전에 가격을 필수로 요구하던 로더에 명시적인 replay 모드를 추가했다. `check-data` 기본 동작은 여전히 가격을 요구하며, `--replay-only`일 때만 가격 없는 입력을 허용한다. 실제 PPO 환경은 피처·로그수익률을 직접 사용하므로 이번 재현에는 가격 파일이 필요하지 않다.

## 실제 재실행 결과

대표 모델은 기존 기본 보상 `mdd_l1.0`, seed 42로 고정했다. 성과가 좋은 모델을 골라 제시한 것이 아니다.
설명 날짜는 각 연도의 6월 중순 거래일이다.

| Window | 테스트 기간 | 대조한 일별 행 | 설명 날짜 | SPY 배경 평균 비중 | SPY 정책 비중 |
| --- | --- | ---: | --- | ---: | ---: |
| 1 | 2022 | 259 | 2022-06-15 | 7.8651% | 6.8153% |
| 2 | 2023 | 257 | 2023-06-15 | 7.9876% | 5.3552% |
| 3 | 2024 | 259 | 2024-06-14 | 9.0726% | 8.7350% |
| 4 | 2025 | 258 | 2025-06-16 | 6.7912% | 8.8215% |

모델의 288차원 관측, 12차원 행동, 자산 순서·기간·학습 스텝을 검사했다. `rl-engine/src/environment.py`의 실제 환경과 `action_to_weights` 함수를 그대로 사용했다.

기존 `models/eval_test_daily.csv`와 날짜, 순수익률, 계좌 가치, 낙폭, 회전율, 최대 목표 비중, SPY 벤치마크 수익률을 대조했다. 4개 실행 모두 `atol=1e-8, rtol=1e-6` 기준을 통과했다. 정확한 오차는 각 결과의 `reconciliation`에 저장한다. 이는 **기존 실행을 재현했다는 확인**이며, 기존 실행의 혼합 통화·체결 규칙 등이 과제 최종 기준에 맞는다는 판정은 아니다.

## SHAP 설명을 읽는 방법

2022-06-15 SPY 예시:

- 비교 기준인 학습 배경의 평균 SPY 비중: **7.8651%**.
- 해당 날짜 모델이 결정한 SPY 비중: **6.8153%**.
- 차이 약 **−1.0498%p**를 입력 그룹별 기여도로 나눈다.
- AAPL 과거 수익률 그룹: 약 **−1.2398%p**.
- NVDA 과거 수익률 그룹: 약 **−1.1333%p**.
- SK하이닉스(000660) 과거 수익률 그룹: 약 **+1.1132%p**.
- 나머지 그룹을 모두 합쳐야 최종 차이가 된다. 위 세 항목만의 합이 최종 차이는 아니다.

이는 '해당 입력이 선택한 배경과 비교해 모델의 SPY 출력 비중에 어떻게 기여했는가'라는 설명이다. 특정 종목의 수익률이 SPY 가격 변화를 일으켰다는 인과관계나 매수·매도의 타당성을 증명하지 않는다. 배경 평균 비중도 해당 날짜 직전 보유 비중이 아니다.

배경은 각 모델의 학습 구간에서 실행한 정책 관측 32개다. 테스트 구간을 배경에 넣지 않았다. 각 종목의 과거 20일 수익률은 한 그룹, 현재 보유 비중 12개는 하나의 공동 그룹으로 설명한다. 총 49개 그룹의 Permutation SHAP을 사용한다.

4건의 `기준값 + 기여도 합 = 정책 비중` 최대 오차는 약 **4.29×10⁻⁸**로 허용 기준 1×10⁻³ 이하이다. Summary Plot은 각 실행에서 선택한 **한 결정**의 기여도 크기를 나타내므로 전체 기간의 중요도로 일반화하지 않는다. Force Plot은 기여도의 방향도 보여준다. 배경·seed 변경에 따른 안정성 검사는 남아 있다.

## 실행 방법

저장소 루트에서 가상환경을 활성화하고 아래 명령을 실행한다. 출력 폴더는 매번 비어 있어야 한다.

```bash
python -m pip install -r requirements-rl.txt
python -m pip install -e . --no-deps

python -m evaluation_xai check-data \
  --processed-dir ../data/processed --window 1 --replay-only

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m evaluation_xai explain-ppo \
  --model ../models/ppo_test_w1_mdd_l1.0_s42.zip \
  --processed-dir ../data/processed \
  --rl-source-dir ../rl-engine/src \
  --target-asset SPY --decision-date 2022-06-15 \
  --reference-daily ../models/eval_test_daily.csv \
  --output artifacts/my-real-w1
```

`--reference-daily`를 지정하면 선택한 모델의 테스트 기간 전체를 대조한 후 설명을 게시한다. 날짜나 값이 다르면 실패한다. 생략 시 대조하지 않았다는 상태를 명시한다.

이번에 생성한 결과는 `artifacts/real-20261112-w1`부터 `w4`에 있다. 원본 입력·모델과 생성 결과는 Git에 포함하지 않는다.

```text
artifacts/real-20261112-w1/
  evaluation.json          # 설명, 입력 검증, CSV 대조, 파일 해시
  replayed_daily.parquet   # 해당 모델의 테스트 구간 전체 재실행 결과
  explanation_inputs.npz   # 설명 대상 관측 1개와 학습 배경 32개
  plots/shap_summary.png
  plots/shap_force.html
```

API 조회:

```bash
EVALUATION_ARTIFACT_DIR=artifacts/real-20261112-w1 \
  python -m uvicorn evaluation_xai.api:app --port 8000
```

`POST /explain` 요청:

```json
{"run_id":"ppo_test_w1_mdd_l1p0_s42","decision_id":"d20220615","target_asset":"SPY"}
```

그림 경로는 `/explanations/ppo_test_w1_mdd_l1p0_s42/d20220615/SPY/plots/summary`와 `/force`다. 실제 모델 설명이므로 합성 opt-in은 필요 없다.

이 폴더는 **설명 전용 결과**다. 기존 64개 실행 성과 묶음과 자동 병합하지 않으며, 플랫폼 성과 화면에 필요한 전체 지표 묶음으로 취급하지 않는다. 프론트·Gateway의 연결은 별도다.

## 검증과 남은 작업

- 로컬 pytest **98개 통과**. Black·flake8 검사 통과.
- 4개 실제 결과 각각 `/explain`, Summary PNG, Force HTML 응답 200 확인. Summary 이미지 육안 확인. 브라우저에서 Force Plot 동작·플랫폼 전체 통합은 미검증.
- 관측/배경 생성과 SHAP은 오프라인 작업이다. 최초 실행 전체 약 7.76초, SHAP 계산만 약 2.82초였다. 이후 실행은 캐시/초기화 영향을 받으므로 이 수치로 요청당 5초 보장을 주장하지 않는다.
- 원본 가격 대조, 혼합 시장 FX·시각 검증, RF, 동일 조건 MVO 비교, 실제 ANOVA 2·3, 모든 결정 설명과 배경 안정성, RAG·Safe-Guard 변경 설명은 남아 있다.
- 기존 RL 평가와 맞추기 위해 재현 실행에서도 Safe-Guard 조기 종료는 껐다. 과제 기준·기존 Evaluation 기준과의 차이는 [11월 11일 연동 기록](team_integration_20261111.md)을 함께 참고한다.

교육용이며 실제 투자 조언에 사용할 수 없다. 백테스트 성과는 미래 수익을 보장하지 않는다.
