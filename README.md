# Evaluation & XAI

에이전틱 RAG + 강화학습 로보어드바이저의 **평가·설명 가능성 모듈**.
정제 데이터와 PPO 모델을 기다리지 않고 계산·검증·연동 코드를 합성 데이터로 먼저 개발한다.

> 데모는 가상 가격과 예시 정책을 사용한다. 실제 투자 성과, PPO 학습 완료,
> 과제 성능 기준 충족을 의미하지 않는다. 수집·전처리와 PPO 학습은 다른 팀원의 역할이다.

## 바로 실행

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m evaluation_xai demo --output artifacts/demo --with-shap
python -m uvicorn evaluation_xai.api:app --host 127.0.0.1 --port 8000
```

Swagger: http://127.0.0.1:8000/docs

재실행 시 새로운 빈 `--output` 폴더를 지정한다. 기존 결과는 덮어쓰지 않는다.
다른 폴더를 API에서 읽으려면 `EVALUATION_ARTIFACT_DIR` 환경변수를 지정한다.
Matplotlib 캐시 권한 경고가 있으면 `MPLCONFIGDIR=/tmp/evaluation-mpl`을 지정한다.

## 구현 범위

| 기능 | 현재 상태 |
| --- | --- |
| 12개 성과지표 | 원장 검사 후 계산. 계산 불가 값은 0 대신 null과 사유 반환 |
| 동일가중·MVO | 월간 리밸런싱, MVO 252일 공분산·비중 상한 40%·공매도 금지 |
| 참조 백테스터 | 비용·다음 종가 체결·MDD 보호·종료까지 현금 기록 |
| Walk-Forward | 4년 학습/1년 테스트 일정, 구간마다 새 학습 함수 호출 |
| 통계 | 월별/seed 집계, One-/Two-way ANOVA, η², 조건부 Tukey, Holm, 블록 부트스트랩 |
| SHAP | 그룹별 기여도 계산, Summary·Force Plot, SB3 형태 모델 연결 어댑터 |
| 평가 API | /health, /backtest, /explain: 사전 계산 결과 조회 |
| 팀 전달 규격 | 가격·계좌 원장·모델·risk tag, 데이터 파이프라인 연동 주의점 |

아직 미완료: 실제 PPO 학습/평가, 실데이터 3종 ANOVA, 실제 모델 SHAP,
신규 의사결정 5초 이내 설명, 전체 대시보드, 20페이지 보고서, 보너스 통합.
`toy_policy`는 PPO가 아니며 MVO 성과의 설명으로 연결하지 않는다.

## 생성 결과

```text
artifacts/demo/
├── evaluation.json       # 합성 여부·지표·통계·설명·미완료 상태
├── equal_weight_WF2022/  # WF2023~2025도 동일
│   ├── ledger.parquet
│   ├── weights.parquet
│   └── decisions.json
├── mvo_WF2022/
└── plots/
    ├── shap_summary.png
    └── shap_force.html
```

## 연결 구조

```text
Data Pipeline → 가격·벤치마크·RF·통화/시간 메타데이터
RL Engine     → 모델·전처리·행동 변환·결정/계좌 기록
Agentic RAG   → 가용시각·출처가 있는 risk tag와 규칙 변경 로그
                           ↓
Evaluation & XAI → 성과·통계·정책 SHAP·보호규칙 변경 설명
                           ↓
FastAPI 결과 조회 → Streamlit (HTTP만 사용)
```

API는 모델을 직접 로드하지 않아 `real_model_loaded=false`이다.
알려진 결정의 설명이 없으면 `/explain`은 409, 모르는 결정은 404,
결과 파일이 없으면 503. 가짜 설명으로 빈 결과를 채우지 않는다.
`/optimize`, `/research`는 전체 서비스의 다른 담당 영역이다.

## 개발·검증

```bash
python -m black --check src tests
python -m flake8 src tests
python -m pytest --cov=evaluation_xai
```

테스트: 수익률/자산가치 일치, 거래비용, 미래 데이터 접근 방지, MDD 종료 후 손실 기록,
누락 seed/기준전략 복제 차단, SHAP 가산성, 날짜/비중/API 입력 검사.
GitHub Actions에는 Python 3.12/3.13 테스트를 설정했다.
`requirements.txt`는 검증 환경의 고정 버전, `pyproject.toml`은 호환 범위이다.
실제 RL의 torch/SB3 버전은 모델 담당자와 맞춘 뒤 추가한다.
`.env`, 모델, 원본 데이터, 생성 결과, 가상환경은 Git에서 제외한다.

## Docker — 평가 API만

```bash
docker compose build
docker compose run --rm evaluation-api python -m evaluation_xai demo --output artifacts/docker-demo --with-shap
docker compose up
```

이 구성은 평가 서비스만 실행한다. 전체 API+Streamlit Compose는 팀 통합 시 합친다.
Docker 검증 여부는 PR 검증 기록을 확인한다.

## 읽을 문서

- [평가 규칙](docs/evaluation_protocol.md)
- [팀원에게 받을 자료와 연동 주의점](docs/team_handoff.md)
- [남은 개발 순서](docs/roadmap.md)

참고: [과제](https://nimble-ceder-40b.notion.site/38_-RAG-1bd17efd202c831fb96381f11fb17539),
[SHAP](https://shap.readthedocs.io/), [statsmodels](https://www.statsmodels.org/),
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
