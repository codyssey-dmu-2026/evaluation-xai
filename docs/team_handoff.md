# 팀 전달 자료와 연결 계약

## 현재 Data Pipeline 확인 결과

2026-09-26 `codyssey-dmu-2026/data-pipeline` commit
`fd63b9795c24f3239e00c553af6a552d91ac1897` 기준. 다른 팀원의 코드는 수정하지 않았다.

1. 미국8+한국4자산. 이전 미국ETF10개 제안과 다르므로 임의로 변경하지 않는다.
2. USD·KRW 가격을 그대로 합산하면 안 된다. 기준통화·당시 환율·의사결정 시각·휴장일 정책 확정 필요.
3. 현재 `returns.parquet`는 **정규화된 수익률**. 성과에 사용하지 않는다. 정규화 전 수정주가와 계좌 순수익률을 별도 전달받는다.
4. 현재 normalize는 단일 종료일 이전 전체 자료로 fit. WF마다 정확한4년 학습 구간에 fit한 scaler와 변수순서를 저장해야 한다.
5. 날짜 교집합은 시장별 휴장 차이를 숨길 수 있다. 원본/통합 달력과 결측 사유 필요.
6. 벤치마크·RF·환율·가용시각이 아직 평가 계약으로 제공되지 않았다. 0/미래값으로 자동 대체하지 않는다.

`MarketDataManifest`는 정책 명시를 요구하지만 통화변환의 경제적 정확성까지 증명하지는 못한다.

## Data Pipeline → Evaluation

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
전처리 두 번 적용 금지. torch/SB3 강제 설치나 모델 임의 역직렬화는 하지 않는다.
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
- POST /explain 예제: `{"run_id":"toy_policy","decision_id":"toy0","target_asset":"SYN00"}`.
- data_mode/금융 면책 문구 항상 표시. null은0으로 바꾸지 말고 이유 표시.
- 409=설명 미생성,404=없는ID,503=결과 미준비. Streamlit은 HTTP만 사용.

## 회의에서 우선 결정할 것

1. 혼합12자산 유지 여부와 기준통화·환율·공통 평가시각.
2. 전처리 전 가격/수익률 및 fold별 scaler 규격.
3. PPO action→weight 함수와 보호규칙 전후 기록.
4. 벤치마크/RF 출처, 대표 보상전략 및 성능 판정기간.

교육용이며 실제 투자 조언에 사용할 수 없다. 백테스트 성과는 미래 수익을 보장하지 않는다.
