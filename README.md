# Evaluation & XAI

에이전틱 RAG + 강화학습 로보어드바이저의 **평가·설명 가능성 모듈**.
팀의 일별 평가 CSV를 분석하고, 학습 입력과 PPO 모델을 받아 성과·통계·SHAP을 계산한다.
현재 공개 CSV 64개 실행을 분석했고, 전달된 학습 입력으로 대표 PPO 4개 구간의 재현과 SHAP 설명을 확인했다.

> 데모는 가상 가격과 예시 정책을 사용한다. 실제 투자 성과, PPO 학습 완료,
> 과제 성능 기준 충족을 의미하지 않는다. 수집·전처리와 PPO 학습은 다른 팀원의 역할이다.

## 평가 기준부터 읽기

- [리뷰 수정·입력 규칙 (2026-11-13)](docs/review_fixes_20261113.md): 거래 시각·월·seed 검증, 국면 오류 격리. 로컬 테스트 121개 통과.
- **[실제 PPO·SHAP 결과와 실행 방법 (2026-11-12)](docs/real_replay_20261112.md)**: 받은 데이터 검사, 대표 4개 실행 재현, 설명 그림·API.
- [팀 연동 현황 (2026-11-11)](docs/team_integration_20261111.md): CSV 분석과 기존 기준의 차이.
- **[평가 기준 요약](docs/evaluation_criteria.md)**: 과제 필수조건, 담당자가 정한 기본값, 목표 판정 방법.
- **[상세 기준서](docs/evaluation_specification.md)**: 12개 지표 계산식, Walk-Forward, 거래·통계·SHAP 기준과 목표 전달 규격.
- **[현재 구현 상태와 차이](docs/implementation_status.md)**: 기준서 중 구현된 항목과 실자료 연동·검증이 남은 항목.
- [전체 문서 안내](docs/README.md) · [팀원에게 받을 자료](docs/team_handoff.md)

기존 미국 ETF 10개 연구안은 팀 최종 자산 구성의 확정본이 아니다.
혼합 시장 사용 시 자산·기준통화·환율·달력·벤치마크/RF를 먼저 합의한다.

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
| 팀 자료 입력 | `import-rl`로 일별 CSV·메타데이터 검사, `check-data`로 processed 파일 검사 |
| 12개 성과지표 | 원장 검사 후 계산. 계산 불가 값은 0 대신 null과 사유 반환 |
| 동일가중·MVO | 월간 리밸런싱, MVO 252일 공분산·비중 상한 40%·공매도 금지 |
| 참조 백테스터 | 비용·다음 종가 체결·MDD 보호·종료까지 현금 기록 |
| Walk-Forward | 4년 학습/1년 테스트 일정, 구간마다 새 학습 함수 호출 |
| 통계 | `anova`로 3종 개별 실행. 자료가 부족한 검정만 미완료, 가능한 결과 보존 |
| SHAP | `explain-ppo`로 실제 환경 재사용. 4개 구간 대표 모델 재실행과 설명 4건 검증 |
| 평가 API | /health, /backtest, /explain 및 PNG/HTML 조회. 합성 설명은 명시적 opt-in 필요 |
| 팀 전달 규격 | 가격·계좌 원장·모델·risk tag, 데이터 파이프라인 연동 주의점 |

아직 미완료: 60개 PPO 전체 재평가·원본 가격/FX 대조, 실데이터 ANOVA 검정 2·3, 모든 결정 SHAP,
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
실제 PPO 설명용 의존성은 `requirements-rl.txt`로 별도 설치한다.
전달된 대표 모델 4개의 일별 결과를 재실행해 팀 CSV와 대조했다. 나머지 모델·프로토콜 검증은 남아 있다.
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

- [평가 문서 안내](docs/README.md)
- [평가 기준 요약](docs/evaluation_criteria.md)
- [상세 기준서](docs/evaluation_specification.md)
- [현재 구현 상태와 차이](docs/implementation_status.md)
- [팀원에게 받을 자료와 연동 주의점](docs/team_handoff.md)
- [남은 개발 순서](docs/roadmap.md)

참고: [과제](https://nimble-ceder-40b.notion.site/38_-RAG-1bd17efd202c831fb96381f11fb17539),
[SHAP](https://shap.readthedocs.io/), [statsmodels](https://www.statsmodels.org/),
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
