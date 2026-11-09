# deep

내 프로그램에서 독립 실행하는 6인·100BB 노리밋 홀덤 AI를 준비하는 프로젝트입니다.

**작업을 이어갈 때는 [현재 상태·설치·학습 재개 안내](docs/continue-work.md)를 먼저 읽으세요.** 코드와 실험 요약은 Git에, 큰 학습 체크포인트는 별도 백업에 저장합니다.

- [실행 계획](PLAN.md)
- [공개 후보 실사와 실행 결과](docs/robopoker-assessment.md)
- [추가 후보와 상용 모델 제공 경로](docs/engine-candidates.md)
- [NoRegrets 로컬 학습·실행 검증](docs/noregrets-assessment.md)
- [내 프로그램용 JSON API와 예제](docs/json-api.md)
- [직접 학습한 1M 모델과 평가 결과](docs/selfplay-training.md)
- [논문·코드에 근거한 학습 개선 우선순위](docs/training-improvement-review.md)
- [6인 Deep CFR 공개 모델 검증과 프로젝트 연결 방법](docs/fullhouse-integration.md)
- [Sol 구현 인계 계획과 별도 검토 쟁점](docs/sol-handoff.md)
- [S1~S5 구현 결과와 Fullhouse 실행 명령](docs/sol-implementation-results.md)
- [R1~R3 연구 판정과 구현 결과 감사](docs/research-decisions-r1-r3.md)
- [R1 수정·평균 전략 oracle·정확 CPU 재개·새 모델 평가](docs/r1-corrections-results.md)
- [R1 CPU 3-seed 학습 곡선과 가중치 집중도](docs/r1-learning-curve.md)
- [평균 전략 수집 방식 비교와 고정 데이터 fitting 진단](docs/r1-sampler-comparison.md)
- [중간 규모 CPU 학습 실행·상태·자동 평가](docs/fullhouse-medium-training.md)
- [600회 학습 완료·30,000핸드 평가 결과](docs/fullhouse-medium-results.md)
- [성능 후퇴 원인 조사·564,000핸드 재평가·학습 엔진 오류](docs/fullhouse-regression-diagnosis.md)
- [엔진 수정·평균 전략 안정화·새 버퍼 3-seed 재학습](docs/fullhouse-v3-training.md)
- [V3 재학습 완료·288,000핸드 평가·실행 가능한 새 모델](docs/fullhouse-v3-results.md)
- [10일 목표·클라우드 서버 후보와 참고 예산](docs/pluribus-cloud-budget.md)

현재는 기존 공개 엔진을 재사용해 6인·100BB 자기대국을 직접 학습하는 단계입니다. 상용 문의 경로는 중단했습니다. 프로급 실력은 아직 검증되지 않았습니다.

현재 환경에는 100만 회 트리 순회를 학습한 연구용 모델이 저장되어 있습니다. JSONL 호출은 `python scripts/run_poker.py`로 시작합니다. 학습은 CPU 기반 CFR이며 신경망 모델이 아닙니다.

Fullhouse 신경망 backend도 연결했습니다: `python scripts/run_poker.py --backend fullhouse`. 공개 advantage 모델을 사용합니다. 짧은 올인 규칙 오류와 평균 전략 적합도 문제를 수정한 뒤, **[새 버퍼 3-seed×120회 재학습과 288,000핸드 평가](docs/fullhouse-v3-results.md)**를 완료했습니다. 기존 평균 전략 대비 개선됐으나, 60→120회 증가의 일관된 이득은 미확인입니다. 같은 학습의 advantage 정책이 더 유망한 결과를 보였습니다. 기존 실행과 기본 모델은 보존하며 프로급 실력은 미검증입니다.

```bash
bash scripts/fetch_robopoker.sh
python scripts/verify_robopoker.py
```

Rust 1.90.0 설치와 재현 환경은 실사 문서에 설명되어 있습니다. 외부 소스와 빌드 결과는 `.cache/`, 실행 로그는 `artifacts/`에 저장하며 Git에는 포함하지 않습니다.

별도의 NoRegrets 후보에서 6인·100BB 학습·저장·재개·대전의 작은 기능 검사를 실행하려면:

```bash
bash scripts/try_noregrets.sh
```

이 검사는 220회 반복의 극소규모 모델을 만들며 실력을 측정하거나 강한 모델을 제공하는 명령이 아닙니다. 실제 자기대국 예비 학습은 `configs/selfplay-pilot.json`과 `scripts/train_selfplay.py`를 사용합니다.

검증 모델을 로컬 프로그램에서 호출하기:

```bash
python scripts/build_json_adapter.py
python examples/query_poker.py artifacts/noregrets-smoke/run-NK70RShx/resumed.bin
```
