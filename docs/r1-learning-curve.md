# R1 CPU 학습량 확대 실험

2026-11-06. 6인·100BB, CPU workers=1 기준선의 제한 실험이다. 프로급 강도 판정이나 GPU 성능 검증을 의미하지 않는다.

후속 결과: [수집 방식 비교와 고정 replay fitting 진단](r1-sampler-comparison.md).

## 재현과 기록

```bash
cd /workspace/deep
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/run_r1_learning_curve.py
# 위에서 출력한 폴더로 분석을 생성한다.
python scripts/summarize_r1_learning_curve.py RUN_DIRECTORY
```

실행 폴더: `artifacts/r1-learning-curve/20261106T015348055280Z/`. `results.json`에 명령·실행 시간·소스 해시·평가 경로를 기록한다. `analysis.json`은 가중치 집중도와 평가 요약이다. 각 seed 폴더의 `diagnostics.jsonl`은 반복별 원시 계측, `train.log`는 학습 로그다. 체크포인트와 평균 전략 npz를 25·50·75·100회에 저장한다.

학습 seed=42/43/44 각각 100회, traversal=32/회, 플레이어별 regret 버퍼 20,000, 평균 전략 버퍼 20,000, batch=128, 매 반복 advantage/strategy 각각 optimizer 16 steps다. 신경망은 upstream처럼 매 반복 초기화하여 replay로 회귀한다. 이 작은 fitting 예산이 충분하다고 입증한 실험은 아니다. 과거 12회 pilot보다 traversal/버퍼/optimizer 예산도 증가했으므로 두 실험의 차이를 iteration 수 하나의 효과로 해석하지 않는다. 25→50→100 비교는 각 seed의 동일 실행에서 추출한다.

CPU reference의 알고리즘은 유지하고, 표본 수집과 replay의 가중치를 계측했다. horizon=100은 시작부터 고정이며 실행당 600초/학습 프로세스 RSS 20GiB 상한을 둔다. 평가 실행은 각각 180초 제한, 평가기 내부에서 엔진 포함 RSS 20GiB를 확인한다. 실행 시간 상한 초과 시 실패하고 자동으로 비용을 확대하지 않는다.

## 가중치 진단의 의미

수집 단계에서는 reservoir 투입 전의 모든 방문(가중치 0 포함)을 계측한다. replay 단계에서는 실제 저장된 표본의 `iteration/horizon × importance weight`를 계측한다. 중앙값·95/99백분위·최대·최대 단일 표본 점유율·상위 1% 점유율·`(sum w)^2/sum(w²)` ESS를 기록한다. 최대값으로 rescale하여 ESS 계산의 overflow를 피하고, 비정상 가중치/비유한 loss는 실패시킨다.

ESS는 가중치 집중도를 나타내는 기술 통계다. 같은 궤적의 history는 상관되어 있으므로 독립 표본 수나 신뢰구간의 유효 표본 수로 해석하면 안 된다. 버퍼가 가득 차면 원본 로그의 `strat_samples`는 크기 증가분이어서 0이 될 수 있다. 실제 수집량은 `collection.count`와 `reservoir_total_seen`을 본다.

## 평가 설계와 해석 범위

공개 Fullhouse advantage 모델, 이전 R1 12회 평균 전략 모델, seed 3개 × 25·50·100회 평균 전략 모델을 각각 **동일 NoRegrets 1M 상대 5명**과 평가한다. 각 후보 100개 독립 deal × focal 6좌석=600핸드, 총 11후보=6,600핸드다. deal seed=77110600이며 학습 RNG와 분리했다. 후보는 자기 패와 공개 상태만 받는다.

같은 deal/좌석/action seed를 사용하지만 정책에 따라 경로가 달라지므로 모든 난수 소모가 정확히 대응하는 것은 아니다. 각 deal에서 6좌석 손익을 평균한 뒤, 후보와 공개 모델의 차이에 명목 95% paired deal-block CI를 계산한다. BB=100칩이므로 여기서는 mean chips/hand의 수치가 BB/100과 같다.

이는 하나의 약한 비교 상대에 대한 작은 표본 실험이다. 다수 후보 중 유리한 결과를 고르면 선택 편향이 생긴다. 승격을 검토할 때는 새로운 held-out deal set과 추가 상대가 필요하다. 시간에 따라 바뀌는 가중 replay loss의 크기를 실력 지표로 사용하지 않는다.

## 검사

계측 단위 검사(균등 가중치 ESS=4, 단일 양의 가중치 ESS=1, 0/비유한 처리)를 추가하여 전체 **20개 통과**. 외부 eval7의 pyparsing deprecation warning 7개는 남아 있다. 기록 `.cache/learning-curve-tests.log`.

계측을 추가한 소스로 `scripts/run_r1_pilot.py`를 다시 실행했다. 연속/중단 후 재개/같은 seed 반복의 모델·버퍼·RNG 동일성과 실제 parameter update가 재확인됐다. 기록 `artifacts/r1-training/20261106T015415259557Z/results.json`.

## 학습과 가중치 결과

| Seed | 학습 시간 | 관측 프로세스 RSS | 최종 버퍼 ESS / 20,000 | 상위 1% 가중치 점유율 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 155.0초 | 429.0 MiB | 37.74 | 89.26% |
| 43 | 166.9초 | 412.4 MiB | 18.00 | 90.90% |
| 44 | 170.2초 | 429.2 MiB | 11.95 | 88.19% |

세 seed 모두 100회를 완료했다. 학습 과정의 비유한 loss/weight 실패는 없었다. 최종 replay 가중치의 ESS는 약 12~38, 상위 1% 점유율은 88~91%다. 샘플 수를 늘려도 소수의 표본에 학습이 지배되는 구조적 비효율이 드러났다. 높은 중요도 분산은 실험으로 확인한 문제이며, 가중치를 임의 clip하여 결과를 바꾸지는 않았다.

위 시간은 이 환경의 실행 기록이며 독립적인 속도 벤치마크는 아니다. 일부 검사는 첫 학습과 동시에 실행했다.

## 최종 대전 결과

각 수치는 BB/100 ± 명목 95% deal-block CI 반폭이다. 마지막 열은 공개 모델 대비 동일 deal의 paired 차이다.

| 후보 | 손익 | 공개 모델 대비 차이 |
| --- | ---: | ---: |
| public | -41.82 ±115.57 | +0.00 ±0.00 |
| r1-12 | -1002.45 ±767.09 | -960.63 ±783.84 |
| seed-42-iter-25 | -887.99 ±782.65 | -846.16 ±795.21 |
| seed-42-iter-50 | -764.20 ±803.56 | -722.38 ±805.63 |
| seed-42-iter-100 | -708.04 ±799.57 | -666.22 ±798.59 |
| seed-43-iter-25 | -634.40 ±761.72 | -592.58 ±773.39 |
| seed-43-iter-50 | -630.77 ±765.06 | -588.95 ±777.92 |
| seed-43-iter-100 | -1024.65 ±864.72 | -982.83 ±885.48 |
| seed-44-iter-25 | -504.75 ±732.33 | -462.92 ±752.37 |
| seed-44-iter-50 | -731.87 ±780.10 | -690.05 ±789.02 |
| seed-44-iter-100 | -704.53 ±797.00 | -662.71 ±798.32 |

11개 평가 모두 완료됐다. 원시 기록을 다시 읽어 6,600핸드의 좌석/deal 유일성, 칩 보존, focal 손익과 요약 수치 일치를 확인했다. 평가 실행 시간 합계는 약 179초다. 저장한 12개 npz는 모두 유한한 tensor 값을 가지며 해시도 서로 다르다(`model-checks.json`). 공개 모델/기본 registry는 교체하지 않았다.

| Seed | 100회 − 25회 손익 변화 |
| --- | ---: |
| 42 | +179.95 ±457.81 |
| 43 | -390.25 ±572.88 |
| 44 | -199.79 ±492.98 |

세 within-seed 구간 모두 0을 포함하며 방향도 일치하지 않는다. **이번 학습 예산에서 반복 수 증가에 따른 일관된 강도 개선을 확인하지 못했다.** seed 43의 최종 후보는 공개 모델 대비 명목 paired CI에서도 불리하지만, 이를 모든 조건에 대한 일반적 성능 판정으로 확대하지 않는다. 작은 fitting 예산, 학습 표본 수, 높은 중요도 분산의 기여도를 이번 실험 하나로 분리하지는 못한다.

## 다음 우선순위와 별도 판단

1. 장시간/GPU 학습 확대 전에 평균 전략 표본 수집의 분산을 줄인다. 현재 own-reach 목표와 작은 게임 oracle을 유지하면서 대안 collector의 기대값·표본 효율을 검증한다. 임의 clipping은 목표를 바꾸므로 별도 편향 평가 없이 도입하지 않는다.
2. 안정화한 collector와 현 기준선을 동일 seed/예산으로 비교한다. 매 반복 초기화하는 신경망의 optimizer 16 steps가 목표를 충분히 fitting하는지도 고정된 replay 데이터에서 측정해 학습 예산과 수집 분산을 구분한다.
3. 개선이 반복되면 새로운 평가 deal과 추가 상대를 사용한다. 이후 GPU/병렬 경로를 검증한다.

별도 이론적 판단은 51개 요약 특징의 imperfect recall, 공유 신경망 근사, 6인 균형 수렴 보장이다. 이번 실험은 이 판단을 해결하지 않는다. 중요도 쏠림과 제한 예산의 강도 미개선은 관측 결과이므로 이론 결론을 기다리지 않고 다음 구현의 기준으로 사용할 수 있다.
