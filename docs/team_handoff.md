> 2026-11-15 추가: `compare-rl` 월간 비교군·PPO 재실행·실제 ANOVA 묶음을 제공한다.
> 프로필·원본 대조 불일치·파일 경로·남은 입력은 [최신 전달 기록](rl_comparison_20261115.md)을 참조한다.
> 기존 import 결과·SHAP 폴더와 자동 병합되지 않으며, 과제 최종 판정은 보류다.

# 팀 전달 자료와 연결 계약

[평가 기준 요약](evaluation_criteria.md) · [상세 기준서](evaluation_specification.md) · [구현 상태와 차이](implementation_status.md)

이 문서는 현재 코드에 연결할 입력 중심이다. 상세 기준서 9절은 최종 목표 계약이므로 파일명·필드가 모두 같지는 않다. 연동 시 구현 상태 문서의 차이표를 먼저 확인한다.

2026-11-13 입력 규칙 변경: 명시적 실행표의 시각은 지정한 `session_timezone`에서 직전/현재 가격 세션 날짜와 일치해야 한다. 월별 통계는 `YYYY-MM` 또는 월별 Period만 허용하고 PPO의 모든 seed를 요구한다. 국면 값 오류는 국면 검정에만 기록한다. 자세한 규칙은 [리뷰 수정 기록](review_fixes_20261113.md)을 참조한다.

## 현재 Data Pipeline 확인 결과

2026-11-11 `codyssey-dmu-2026/data-pipeline` commit
`755af516259adebf93a01c5b1d61ea32918e245e` 기준. 다른 팀원의 코드는 수정하지 않았다.
실행 명령과 파일별 현재 규격은 [최신 연동 현황](team_integration_20261111.md)을 참조한다.

1. 미국8+한국4자산. 이전 미국ETF10개 제안과 다르므로 임의로 변경하지 않는다.
2. USD·KRW 가격을 그대로 합산하면 안 된다. 기준통화·당시 환율·의사결정 시각·휴장일 정책 확정 필요.
3. 현재 `returns.parquet`는 **정규화 전 로그수익률**이다. `features_window1~4.parquet`에 구간별 정규화 피처를 저장한다. 계좌 성과에는 비용 차감 후 포트폴리오 수익률을 사용한다.
4. 현재 구간별 scaler와 변수순서는 `walk_forward_manifest.json`에 저장한다. 학습 기간은 2018년부터 확장되므로 기존 4년 롤링 기준과 다르다. 단일 `features.parquet`는 이전 구간 평가에 그대로 사용하지 않는다.
5. 시장 달력 합집합·forward fill·개장 플래그를 사용한다. 실제 달력 완전성·정보 가용시각과 혼합 시장 체결 규칙은 별도 확인이 필요하다.
6. 벤치마크·RF·환율·가용시각이 아직 평가 계약으로 제공되지 않았다. 0/미래값으로 자동 대체하지 않는다.

`MarketDataManifest`는 정책 명시를 요구하지만 통화변환의 경제적 정확성까지 증명하지는 못한다.

## Data Pipeline → Evaluation

아래 `MarketDataManifest` 표는 참조 백테스터의 공통통화 입력 계약이다.
현재 팀의 processed 파일을 읽는 `load_processed_window`는 별도 로더이며,
파일 형식 검사를 통과해도 이 표의 통화·시각 검증까지 완료되는 것은 아니다.

| 자료 | 내용 |
| --- | --- |
| prices.parquet | 날짜 인덱스·종목 열, 정규화 전 양수 수정주가, 동일 기준통화 |
| manifest.json | dataset_id, data_mode, asset_order, source, price_type=adjusted_close, normalized=false, base_currency, converted_to_base_currency=true, calendar_policy, availability_policy, missing_value_policy |
| benchmark/RF | 평가일과 정확히 일치하는 일별 단순수익률. 로그값은 expm1 변환 후 명시 |
| feature schema | 종목/변수 순서·shape·정규화 전후 구분·산출/가용시각 |
| fold preprocessing | train_start/end, scaler, fit 기간, 데이터 버전/해시 |

실데이터는 `validate_market_input(prices, MarketDataManifest(...))` 선행.
날짜·shape 검사와 별개로 point-in-time 정확성을 원본 기록에서 확인한다.

## RL Engine → Evaluation

| 자료 | 내용 |
| --- | --- |
| train_factory(train_prices, fold) | 구간마다 새 모델 학습/전처리, Policy 반환. 외부 테스트 자료 참조 금지 |
| model bundle | 체크포인트·SB3/torch버전·seed·보상/λ·학습스텝·자산순서·관측shape·scaler |
| predict_weights(batch) | 결정론적 추론→환경과 동일 비중 변환→[batch,asset] |
| ledger.parquet | 날짜, equity_start/end, net_return_simple/log, commission, slippage. 리셋/외부입출금 금지 |
| decisions | run/decision ID, observation, decision_at, execution_at, weight_policy/final, guard_reason, risk_tag_ids |
| SHAP background | 해당 fold 학습 rollout32개 관측과 날짜 |

`sb3_weight_predictor`에 model·observation_shape·action_to_weights·필요시 preprocess 전달.
전처리 두 번 적용 금지. 신뢰하는 팀 체크포인트를 명시적으로 제공한 경우만 로드한다.
2026-11-12 모델 60개와 피처·수익률·전처리 manifest를 별도 수령했다.
대표 모델 4개 재현·SHAP 결과는 [실제 연결 기록](real_replay_20261112.md)을 참조한다.
가격 파일은 없어 가격 대조가 필요한 계약과 replay-only 검증을 구분한다.
**Episode Reward는 투자수익률이 아니다.** 학습 종료 뒤에도 평가 원장은 현금 유지까지 필요하다.

## Agentic RAG → Evaluation

RiskTag: tag_id, asset_id, risk_type, severity/confidence(0~1), published_at,
available_at, generated_at, expires_at(시간대 포함), source_url, tagger_version, mode.

live에서 생성 전 가용 불가. historical_replay는 실제 생성시각을 숨기지 않고 표기한다.
과거 가용시각을 기입하는 것만으로 LLM 미래지식 누수가 해결되지는 않는다.
태그 반영 전후 비중·규칙 로그 연결은 엔진/RAG 통합 작업이다.

## Evaluation → 프론트

- GET /health: 상태와 실제 모델 미로드 표시.
- GET /backtest: 전체 결과, `?run_id=mvo_WF2022`로 개별 조회.
- POST /explain 합성 데모 예제: `{"run_id":"toy_policy","decision_id":"toy0","target_asset":"SYN00","allow_synthetic":true}`. 실제 설명으로 표시하지 않는다.
- 그림: GET `/explanations/{run_id}/{decision_id}/{target_asset}/plots/summary` 또는 `/force`. 합성은 `?allow_synthetic=true` 필요. Gateway 연결은 별도 작업이다.
- data_mode/금융 면책 문구 항상 표시. null은0으로 바꾸지 말고 이유 표시.
- 409=설명 미생성,404=없는ID,503=결과 미준비. Streamlit은 HTTP만 사용.

## 회의에서 우선 결정할 것

1. 혼합12자산 유지 여부와 기준통화·환율·공통 평가시각.
2. 전처리 전 가격/수익률 및 fold별 scaler 규격.
3. PPO action→weight 함수와 보호규칙 전후 기록.
4. 벤치마크/RF 출처, 대표 보상전략 및 성능 판정기간.

교육용이며 실제 투자 조언에 사용할 수 없다. 백테스트 성과는 미래 수익을 보장하지 않는다.
