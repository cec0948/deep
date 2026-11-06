# S1~S5 구현 및 실행 결과

2026-11-06. `sol-handoff.md`의 **기능 원형 구현과 제한된 검증**을 진행했다. 인계 명세의 모든 완료 조건을 충족한 상태는 아니다. 후속 [R1~R3 감사](research-decisions-r1-r3.md)에서 학습 목표값 변형, 실행 경로·검증 범위 문제를 확인했다. 대규모 학습 전 R1 수정이 필요하다.

## 구현

| 단계 | 결과 | 파일 |
| --- | --- | --- |
| S1 | 고정 소스 fetch/검증, 모델·preflop table 해시 검사, inference/training 의존성 설정 | `scripts/prepare_fullhouse.py`, `scripts/setup_fullhouse.sh`, `configs/fullhouse-*` |
| S2 | 모델 없는 공개 상태 replay, 규칙 엔진 기반 검증, 평가용 reset/step | `native/noregrets-json/bridge.rs`, `scripts/build_json_adapter.py` |
| S3 | Fullhouse JSONL backend, 직접 신경망 호출, 합법 분포·금액 변환·출처, 기존 실행 호환 | `scripts/fullhouse_runtime.py`, `scripts/run_poker.py` |
| S4 | 하나의 NoRegrets 규칙 엔진에서 양방향 및 동일 정책 pilot, 원시 deal 로그, deal-block CI | `scripts/evaluate_fullhouse.py` |
| S5 | 제한된 CPU 새 학습, 버퍼 저장, 새 프로세스에서 상태 보존 재개, 평균 전략 export/조회 확인 | `scripts/train_fullhouse_pilot.py` |

원본 두 checkout은 clean 상태다. 공개 가중치·외부 엔진은 우리 Git 소스로 복사하지 않았다. 기존 NoRegrets 현재 모델 등록도 유지했다.

## 실행

기존 환경에서:

```bash
cd /workspace/deep
python scripts/run_poker.py --backend fullhouse
```

JSON 요청 예:

```json
{"hero":3,"button":0,"hole":["As","Kd"],"board":[],"actions":[],"seed":42}
```

실제 응답: call 확률 약 0.07844, raise-to 225 약 0.46750, raise-to 350 약 0.45407. seed 42에서 raise-to 350. 이는 공개 모델이 실행된 증거이며 해당 행동의 우수성 검증은 아니다. `model_type=advantage`, `model_iterations=null`, `strength_verified=false`다.

새 checkout의 준비 순서:

```bash
bash scripts/try_noregrets.sh
python scripts/build_json_adapter.py
bash scripts/setup_fullhouse.sh
# 작은 직접 학습 환경까지 설치하려면:
bash scripts/setup_fullhouse.sh --training
```

NoRegrets 현재 1M 모델은 Git에 포함되지 않는다. 새 머신의 Fullhouse backend는 이 파일 없이 실행할 수 있지만, 동일 모델 상대 pilot에는 기존 `artifacts/selfplay/current.json`과 그 파일이 필요하다. upstream fetch·Rust 설치 상세는 기존 NoRegrets 문서를 따른다.

## 검증

- 우리 통합 검사 **14개 통과**: 기존 NoRegrets 7개, Fullhouse 7개. Fullhouse는 street별 조회, 동일 상태의 deal marker 유무에 따른 특징·확률 비교, placeholder 독립성, 버튼 회전, 콜 금액, short all-in의 레이즈 비재개, 잘못된 입력과 JSONL 복구, 모델 누락을 포함한다. 독립적인 두 엔진의 전체 동등성 검사는 아니다.
- Fullhouse upstream 학습 검사 **10개 통과**: `test_strategy_weights.py`, `test_numba_parity.py`. Python/Numba 경로·평가·reach weight 등의 검사이며 학습기 전체의 이론적 정확성을 증명하지 않는다.
- 원본의 규칙·사이드팟 검사는 앞선 엔진 검사 기록을 재사용했다. 이번 short all-in 상태 회귀는 별도로 추가했다. 모든 가능한 규칙 조합을 새로 전수 검증한 것은 아니다.
- CLI `python scripts/run_poker.py --backend fullhouse` 실제 요청/응답 성공. 인자가 없는 기존 NoRegrets 경로는 유지했다.
- 로그: `.cache/fullhouse-integration-tests.log`, `.cache/fullhouse-training-tests.log`, `.cache/bridge-build.log`.

재현:

```bash
OPENBLAS_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python -m unittest discover -s tests -v
```

## 공통 엔진 대전 pilot

최종 수정 후 기록: `artifacts/fullhouse-evaluation/20261106T013048502614Z/`.
각 구성은 100개 deal × 6 focal 좌석 = 600핸드. 총 1,800핸드이며 독립 1,800deal이 아니다. 두 정책은 같은 seed의 실제 deck으로 평가했고, 정책 행동 RNG는 별도로 사용했다.

| 1개 focal 좌석 vs 나머지 5개 좌석 | BB/100 | 명목상 95% CI 반폭 |
| --- | ---: | ---: |
| Fullhouse vs NoRegrets 1M | +152.89 | ±261.19 |
| NoRegrets 1M vs Fullhouse | -636.83 | ±354.99 |
| NoRegrets 1M vs 동일 모델 | +186.15 | ±538.27 |

첫 구성의 CI는 0을 포함한다. 역방향은 다른 상대 구성이며 두 결과가 대칭일 필요가 없다. 같은 모델 구성은 0과 일치하는 sanity 결과다. 약한 연구용 1M 기준 상대·100deal의 탐색적 비교이므로 Fullhouse의 일반적인 우세나 프로급 실력은 결론내리지 않는다.

총 실행 약 25.1초, 평가기+엔진 관측 RSS peak 약 1.38 GiB. Fullhouse 결정 p50 약 0.334ms, p95 약 0.909ms. 이 지연은 Python에서 정책 계산 및 금액 변환까지이며 전체 사용자 API 요청시간은 아니다. `hands.jsonl`에 모든 좌석 수익, `results.json`에 street/잔존 인원별 fallback·지연·해시를 기록했다. 비교 중 칩 보존 및 합법 액션을 확인했다.

콜 기록을 street total로 잘못 해석했던 초기 pilot은 참고용으로만 보존하며, 위 최종 경로를 사용한다. 정상 Fullhouse 로그는 call=추가 납부액, raise=street raise-to다.

## 작은 새 학습과 재개

기록: `artifacts/fullhouse-training/20261106T012907404298Z/`.
설정: 6인·고정 100BB, CPU workers=1, 8 traversals/iteration, buffer 2,000/player, strategy buffer 2,000, batch 64, 두 네트워크 학습 각각 2 steps, exploration decay horizon=12.

경로 정정: workers=1이므로 실제 학습 데이터 수집은 Python `run_traversals_fast`였다. Numba는 초기화/워밍업 및 별도 parity 검사가 수행됐고, 해당 pilot의 순회 경로는 아니었다.

1. 총 6 iterations 후 full buffer checkpoint 저장.
2. 별도 디렉터리의 새 프로세스에서 총 12 iterations까지 resume.
3. 체크포인트 내 모델·6개 버퍼·평균 전략 버퍼·유한 수치 확인.
4. 6→12 가중치가 실제 달라지고, 모든 플레이어 버퍼가 유지·증가함을 확인.
5. 새로 export한 `12/iter_00012.npz`를 `Fullhouse`로 로드하여 **average_strategy** 추론 성공.

버퍼 수: 6회 `[91,68,57,231,167,132]` → 12회 `[190,93,119,367,202,165]`. 평균 전략 버퍼 275→439. 관측 RSS 약 0.48/0.44 GiB. 첫 실행은 JIT 등 포함 약 34.9초, 재개 프로세스 약 5.0초. 실패한 SciPy 누락 시도는 별도 디렉터리에 남겼고 의존성을 고쳐 새 캠페인으로 완료했다.

이번 재개는 **모델·버퍼 보존 검증**이다. 원본 RNG가 다시 시작하므로 uninterrupted와 정확히 같은 결과를 보장하지 않는다. 정확 재개를 주장하려면 RNG 상태·worker 상태 복원을 추가해야 한다. checkpoint publication은 각 단계가 성공한 후 우리 manifest에 기록하며, 원본 trainer의 `latest.pt` 직접 덮어쓰기는 원자적 저장으로 보강되지 않았다.

이 wrapper는 CPU 기능 검사 전용으로 GPU cache 경로를 사용하지 않는다. 4070용 학습 설정과 실제 여유 VRAM 기반 cache limit은 미검증이며 대규모 GPU 실행 전 처리할 항목이다. 기존 18GB 기준을 그대로 GPU 실행에 사용하지 않는다.

## 별도 연구 판단 R1~R3

- **R1:** 후속 감사에서 샘플별 regret centering에 의해 목표 전략 비율이 2/3·1/3에서 0.9·0.1로 바뀌는 실제 함수 반례를 확인했다. 배치 의존 시간 가중치, exploration과 평균 전략 표본화도 표준 Deep CFR 동등성을 보장하지 않는다. 공개 모델의 기능 검증과 별개로 새 대규모 학습 전 수정·검증이 필요하다.
- **R2:** 위 pilot은 엔진·연결 검증과 비용 측정까지다. 주력 모델 자동 승격은 수행하지 않았다. 강도 판단은 독립 seed/held-out 상대/더 큰 paired-deal 표본이 필요하다.
- **R3:** 자체 코드 MIT에 더해 vendored 원본 `uzlez/fullhouse-engine@a80b269`도 MIT임을 후속 감사에서 확인했다. 재사용 시 해당 고지를 보존한다. data/ 가중치 재배포 범위는 여전히 명확히 확인되지 않았다.
