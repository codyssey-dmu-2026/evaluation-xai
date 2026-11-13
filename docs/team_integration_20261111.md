# 팀 코드 연동 현황 — 2026-11-11

> 아래는 11월 11일 기록이다. 11월 12일에 학습 피처·수익률을 받아
> [대표 PPO 4개 구간의 실제 재현·SHAP 검증](real_replay_20261112.md)을 완료했다.

## 결론

평가 코드를 합성 데모 전용에서 팀 데이터·결과·모델을 받는 구조로 확장했다.
**실제 일별 결과 CSV 분석은 실행했다. 실제 시장 입력을 사용한 PPO 재평가·SHAP은 아직 실행하지 않았다.**
모델 파일 60개는 사용자가 전달했으며, 대표 모델 CPU 로딩을 확인했다.
학습 당시 `data/processed` 자료는 아직 없다. 임의 입력으로 실제 의사결정 설명을 만들지 않는다.

## 확인한 코드 버전

| 저장소 | 확인 커밋 | 전달 상태 |
| --- | --- | --- |
| data-pipeline | `755af516259adebf93a01c5b1d61ea32918e245e` | 수집·전처리 코드, `data/processed/`는 Git 제외 |
| rl-engine | `05579d2c845e959fd7948f8cd5a826ca0a3b0599` | 일별/요약 CSV, 모델 메타 JSON. ZIP은 별도 수령 |
| agentic-research | `c395f1a341fe46677edeed49cc114b958efe6fc6` | 공용 RiskTag 사용, 시점별 provider 제공 |
| robo-advisor-platform | `db622de94037479d09c43693bfff7f5e04f5e30b` | Gateway·6개 화면. 현재 bootstrap은 synthetic demo |

학습 완료와 학습 자료의 Git 업로드는 다른 문제다. `.gitignore`에서 가격·피처와 모델 ZIP을 제외하고 있으므로 공개 저장소만 clone해도 모두 생기는 구조가 아니다. 수집 뉴스는 이 저장소에 복사하거나 재배포하지 않았다.

## 보완 항목별 변경

| 요청한 보완 | 변경 | 남은 검증 |
| --- | --- | --- |
| 실데이터 로더·CLI | `data_loader.py`, `check-data`, `import-rl` 추가 | 학습 당시 processed 자료로 실제 검사 |
| 실제 PPO SHAP | `explain-ppo`: 실제 환경의 관측·action_to_weights 재사용, 학습 구간 rollout 배경 | 입력 자료 수령 후 실제 설명 산출·안정성·성능 측정 |
| toy 설명 오해 | synthetic `/explain` 기본 409. `allow_synthetic=true`로 명시한 경우만 반환, 설명 범위 표시 | 플랫폼에서 데모 전용 표시 및 옵트인 UI 연결 |
| SHAP 이미지 HTTP | 결정·자산별 PNG/HTML 조회 경로, 경로 이탈 차단, HTML sandbox 헤더 | Gateway 프록시·화면 임베드 연결 |
| ANOVA 부분 실패 | 검정별 결과 보존. 하나가 부족해도 다른 검정 실행, 전체 완료와 Holm 상태는 별도 | 동일 조건 MVO·검증된 국면 표본 확보 |
| 오래된 handoff | 현재 returns는 정규화 전 로그수익률임을 정정 | 최신 자료 전달 시 확인 |
| 위험 태그·시각 | 결정에 aware UTC 시각·risk_tag_ids 기록, 미래/만료/없는 태그 차단 | 실제 교차시장 시간표와 정책의 태그 사용 연결 |

기본 참조 백테스트의 UTC 시각은 **합성 일별 순서 표시**다. 거래소 마감 시각을 알아낸 것이 아니다. 실제 위험 태그를 기록하려면 호출자가 시간대 포함 `execution_schedule`을 제공해야 한다. 정책이 실제로 사용했다고 보고한 `Allocation.risk_tag_ids`만 기록한다. 빈 목록은 RAG 연동 완료를 의미하지 않는다.

## 사용 방법

저장소 루트에서 실행한다. 기존 실행 결과를 덮어쓰지 않으므로 매번 새 출력 폴더를 사용한다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

# 1. 공개 RL 일별 결과 분석: 모델·원시 가격 없이 가능
python -m evaluation_xai import-rl \
  --models-dir ../rl-engine/models \
  --output artifacts/team-results

# 2. 원본 학습 입력을 받았을 때 검증
python -m evaluation_xai check-data \
  --processed-dir ../data-pipeline/data/processed --window 1

# 3. 월별 전략 표본으로 3종 ANOVA를 각각 실행
python -m evaluation_xai anova \
  --samples artifacts/team-results/stat_samples.parquet \
  --output artifacts/mission-anova.json

# 4. 실제 모델 + 원본 학습 입력이 둘 다 있을 때만 설명
python -m pip install -r requirements-rl.txt
python -m evaluation_xai explain-ppo \
  --model ../models/ppo_test_w1_mdd_l1.0_s42.zip \
  --processed-dir ../data-pipeline/data/processed \
  --rl-source-dir ../rl-engine/src \
  --target-asset SPY \
  --output artifacts/ppo-explanation
```

`explain-ppo`는 신뢰하는 팀 코드와 체크포인트에만 사용한다. SB3/PyTorch 모델 로드는 직렬화된 Python 객체를 실행할 수 있다. 모델 파일은 Git에 넣지 않는다.

입력 없이 명령을 실행하면 실패한다. 자동 다운로드·랜덤 피처·가짜 모델로 대체하지 않는다.

## 현재 파일 형식

- `prices.parquet`: 정규화 전 수정가격과 시장 개장 플래그.
- `returns.parquet`: **정규화 전 로그수익률**, 열은 `{ticker}_return`.
- `features_window1~4.parquet`: 각 학습 구간으로 정규화한 피처와 개장 플래그.
- `walk_forward_manifest.json`: 상대경로, 반개구간 `[train_start, train_end)`, `[train_end, test_end)`, 컬럼 순서, 평균·표준편차·winsorize 경계.

로더는 컬럼·날짜·수치·순서·원시 가격 대비 로그수익률 일치·학습/테스트 분리를 검사한다. 정규화를 다시 fit하지 않는다. 원본 수집 시점·환율·거래소 달력·실제 scaler fit 과정의 정확성을 검사만으로 증명하지는 못한다.

`anova` 입력은 `month,strategy,seed,return`과 선택적 `regime`이다. 전략명은 `PPO-simple`, `PPO-sharpe`, `PPO-mdd_l1.0`, `MVO`, `EqualWeight`. PPO seed는 42/43/44, 기준선 seed는 null이다. 국면은 월 시작 전에 정한 `up/down/sideways` 하나를 모든 전략에 공통으로 적용해야 한다.

세 분석은 독립적으로 수행한다. MVO나 국면이 없으면 검정 2/3만 미완료다. 세 검정이 모두 유효할 때만 5개 효과에 대한 Holm 보정을 계산한다. 계산 불가 p값을 1이나 0으로 위장하지 않는다.

## 이번에 실행한 실제 CSV 분석

- 4개 테스트 구간 × (5개 PPO 변형 × 3seed + 동일가중 1개) = **64개 실행**.
- 각 실행의 계좌 가치·수익률·낙폭 일치 검사와 공통 날짜·필수 seed 검사.
- 12개 지표 구조를 제공하되 RF가 없으므로 Sharpe·Sortino·Alpha·Beta는 null과 이유 반환. 나머지 8개 지표 계산.
- 보상 검정 1은 simple/sharpe/mdd λ=1.0의 같은 월 seed 평균, 48개월/집단.
- `F=0.1965616760`, `p=0.8217753923`, `η²=0.0027803569`: 비유의, Tukey 실행 조건 미충족.
- mdd λ=1.0은 기존 기본값으로 고정했다. 테스트 결과가 가장 좋은 λ를 선택한 것이 아니다. λ=0.5/5.0 결과도 원본 실행별로 보존한다.
- PPO-mdd λ=1.0과 일간 동일가중의 평균 월수익률 차이에 대해 1/3/6개월 블록 부트스트랩 실행.
- MVO가 없으므로 전략별·국면별 미션 검정은 미완료. 예시 MVO나 합성 수익률을 섞지 않았다.

위 숫자는 팀이 공개한 CSV에 대한 **탐색적 분석**이다. 원자료로 재실행한 검증, 과제 성능 기준 통과, 미래 성과의 증거가 아니다. 파일 해시를 생성 결과에 보존한다.

## 기존 기준과 달라 최종 판정을 보류하는 이유

| 항목 | 기존 Evaluation v1.0 | 현재 팀 결과 |
| --- | --- | --- |
| 학습 기간 | 매번 직전 4년 | 2018년부터 확장: 4/5/6/7년 |
| seed | 11/22/33 | 42/43/44 |
| 평가 Safe-Guard | 15% 초과 후 청산 | 꺼짐, 초과 여부만 기록 |
| 체결 | 다음 종가 체결 | 비중 결정 후 다음 날 수익 적용 |
| 동일가중 | 월간 리밸런싱 | 매일 리밸런싱 |
| 보상 | 로그수익률·누적 최대낙폭 기준안 | 단순수익률·현재낙폭, 추가 종료 페널티 |
| 시장 | 미국 ETF/USD 제안 | 미국8+한국4, 공통통화·FX 검증 미완료 |
| benchmark/RF | S&P 500 가격지수·일별 RF | SPY ETF 대용, RF 자료 없음 |

현 기준에 맞는 재실험 또는 기준 변경 합의를 하고 버전을 남겨야 한다. 숫자를 맞추려고 과거 기준을 조용히 바꾸지 않는다. 수집부터 다시 실행하면 학습 당시 공급본과 달라질 수 있으므로 모델 담당자와 사용한 스냅샷을 먼저 맞춘다.

## HTTP·플랫폼 전달

```bash
EVALUATION_ARTIFACT_DIR=artifacts/team-results \
  python -m uvicorn evaluation_xai.api:app --port 8000
```

- `/backtest`는 기존 `runs/metrics` 구조를 유지한다. import 결과는 `data_mode=imported_unverified`, `status=provisional`이다.
- 현재 플랫폼이 strategy/fold로만 묶으므로 seed를 strategy 표시에도 포함해 마지막 seed만 보이는 문제를 피했다. 원래 전략명은 `strategy_family`에 남긴다.
- `ledger.parquet` 시계열은 제공하지만 보유 비중 CSV는 없으므로 `weights.parquet`를 만들어내지 않는다.
- `/explain`은 합성 결과일 때 기본 409. 데모 의도가 명확할 때만 요청에 `allow_synthetic: true`를 추가한다.
- 그림: `GET /explanations/{run_id}/{decision_id}/{target_asset}/plots/summary` 및 `/force`. 합성 그림은 `?allow_synthetic=true` 필요.
- 플랫폼은 새 데이터 모드 경고, 개별 검정 `mission_anova` 결과, 그림 경로 프록시/표시를 연결해야 한다. 현재 플랫폼의 `synthetic_baseline_comparison`에 실제 통계 결과를 위장해서 넣지 않았다.
- 전체 대시보드와 5초 SLA의 실제 통합 시험은 아직 하지 않았다.

## 다음에 받을 자료

1. 모델 학습 당시의 `data/processed` 전체와 수집/전처리 버전.
2. 혼합 시장 기준통화·환율·거래 시각 및 휴장일 정책.
3. 같은 실행 조건의 MVO 결과, RF와 테스트 이전 국면 계산용 벤치마크.
4. risk tag/규칙/결정 ID 연결 기록과 학습 곡선.

## 로컬 검증 기록

- Python 3.13: pytest **90개 통과**, Black·flake8·`git diff --check` 통과.
- 실제 공개 CSV: `artifacts/team-20261111-verified/`에 64개 실행 분석 저장.
- 같은 월별 표본으로 `anova` CLI 실행: 보상 검정 `ok`, 전략·국면 검정 `incomplete`를 각각 보존.
- 대표 모델 `ppo_test_w1_mdd_l1.0_s42.zip` CPU 로딩: 관측 288차원, 행동 12차원, 저장된 학습 스텝 100,352 확인.
- `explain-ppo`는 원본 manifest 부재 시 종료 코드 2로 중단하며 설명 결과를 만들지 않음.
- 세 ANOVA가 모두 실행되는 경로와 SHAP 계산·그림 HTTP는 테스트용 자료로 검증. 실제 PPO 설명·플랫폼 연결·Docker 실행 검증은 포함하지 않음.
- 라이브러리의 사용 중단 예정 경고가 있으나 테스트 실패는 없음. 원격 CI 결과와는 별개인 로컬 기록임.

교육용이며 실제 투자 조언에 사용할 수 없다. 백테스트 성과는 미래 수익을 보장하지 않는다.
