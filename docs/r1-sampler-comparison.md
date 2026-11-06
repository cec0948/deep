# 평균 전략 수집 분산과 고정 replay fitting

2026-11-06. 이전 3-seed 실험에서 관측한 중요도 가중치 집중도를 줄이는 별도 실험이다. 원본 소스/공개 가중치/기본 모델 registry는 보존한다.

## 구현과 평균 전략 목표

`native/fullhouse-r1/support.py`에 `fixed-opponent` collector를 추가했다. 각 trajectory의 focal을 6명 중 균등 선택하고, focal은 현재 regret-matching 정책, 나머지는 학습 내내 고정된 합법 액션 uniform 정책 τ로 행동한다. focal의 결정만 기록하고 sample weight=1을 사용한다. 기존 시간 가중치 t/H는 유지하며, 역확률 clipping은 도입하지 않았다. 같은 trajectory에서 다른 플레이어의 표본을 함께 기록하면 그 플레이어가 보는 상대 정책이 고정되지 않으므로 focal만 기록한다.

완전한 perfect-recall 정보집합 I에 대해, focal i의 방문 확률은 `Cτ(I) · πᵗᵢ(I)`이다. Cτ에는 chance, focal 선택 확률, 고정 상대 reach가 포함되며 iteration에 무관하다. 따라서 방문에 `t · σᵗᵢ(I)`를 기록한 평균의 극한은 `Σ t πᵗᵢ(I) σᵗᵢ(I) / Σ t πᵗᵢ(I)`이다. 유한 표본의 비율 추정치 자체가 불편 추정량이라는 뜻은 아니다. τ의 support는 모든 합법 액션에 양수다. focal의 과거 정책 확률이 0인 경로는 해당 iteration own-reach 질량도 0이므로 수집하지 않아도 된다.

**현재 51개 요약 특징으로 여러 정보집합을 합치면 Cτ가 동일하게 소거된다고 보장할 수 없다.** 이 방식은 완전한 정보집합에서의 own-reach 평균을 보존하지만, 공유 근사 신경망의 상태별 fitting 분포는 이전 importance collector와 달라진다. 특히 uniform 상대가 만드는 상태의 분포에 영향을 받는다. 긴 경로의 역확률 분산을 제거하는 대신 희귀 상태 방문 횟수와 수집 좌석별 coverage를 충분히 확보해야 한다. 가중치 ESS만으로 poker 강도를 판단하지 않는다.

기존 `importance` 모드를 기본값으로 유지하고 CLI `--average-sampler fixed-opponent`로 새 경로를 명시한다. checkpoint revision은 `r1-cpu-v2`, sampler를 설정에 포함시켜 다른 sampler로 재개하면 거절한다. 기존 v1 checkpoint를 v2에서 학습 재개하지 않는다. inference용 npz는 그대로 사용할 수 있다.

## 검증

- 전체 검사 **21개 통과** (`.cache/r1-sampler-tests.log`). 새 oracle은 실제 collector의 모든 작은 게임 sampling path를 열거한다. 상대 정책 변화, 한 플레이어의 반복 결정, own-reach=0을 포함해 노드별 own-reach 평균과 비교했다. 노드의 고정 방문 계수는 남고, 평균 정책은 기준 계산과 일치했다.
- `scripts/run_r1_pilot.py --average-sampler fixed-opponent`로 연속/중단 후 재개/같은 seed 반복의 모델·버퍼·RNG가 정확히 일치함을 확인했다. 기록 `artifacts/r1-training/20261106T020830462386Z/results.json`.
- 실제 checkpoint를 다른 sampler로 재개하는 subprocess가 설정 불일치로 실패하고 기존 파일 hash가 보존됨을 확인했다(`sampler-resume-guard.json`).

## 실험 설계와 재현

`artifacts/r1-sampler/20261106T020828461505Z/`에 새 학습 명령/로그/진단/평가 경로를 기록한다. seed=42/43/44, 100회 반복, traversal=32/회, buffer=20,000/player 및 strategy=20,000, batch=128, advantage/strategy 각각 optimizer 16 steps다. 이전 baseline과 반복·trajectory·회귀 예산을 같게 하고 CPU 시간/수집 표본 수는 별도로 기록한다. sampler가 RNG를 다르게 소비하여 같은 seed여도 이후 regret traversal까지 달라질 수 있다. 같은 seed를 같은 학습 경로의 쌍으로 해석하지 않는다. 표본 부족 시 회귀를 생략하는 기존 warm-up 조건도 유지하므로, optimizer 예산은 반복별 상한이 같다는 뜻이다. 수집 표본 수와 실제 경과 시간까지 동일한 실험은 아니다.

비교 기준은 이전 실험의 동일 예산 importance 100회 모델 3개다. 새 고정 상대 모델 3개 및 공개 모델을 새로운 deal seed=88110600, 100 deals × 6 focal seats로 평가한다. 총 4,200핸드다. 각 deal의 6좌석 손익을 평균하여 fixed−importance의 paired nominal CI를 계산한다. 이전 보고서의 다른 deal 결과와 직접 차감하지 않는다. 모든 후보를 같은 NoRegrets 1M 상대 5명에게 대전시킨다. 후보별 600핸드의 작은 비교로 프로급 실력을 판정하지 않는다.

```bash
cd /workspace/deep
python scripts/build_fullhouse_r1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python -m pytest tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/run_r1_pilot.py --average-sampler fixed-opponent
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/run_r1_sampler_comparison.py
```

## 고정 replay fitting 진단

`scripts/probe_r1_fitting.py CHECKPOINT`는 고정된 checkpoint replay를 seed=701로 80/20 row split한다. 모델 초기화 seed=7/8/9를 사용하고, 같은 초기화에서 optimizer를 유지하며 누적 0/16/128/512 steps를 측정한다. batch=128, lr=0.001, 현재 학습 함수/네트워크/가중치/gradient clipping을 그대로 사용한다. 평가는 전체 해당 split에 대해 가중치 합으로 정규화한 MSE 및 KL(target || model)을 사용한다. KL은 soft target entropy를 빼서 불가피한 entropy와 구분한다.

row holdout은 같은 trajectory의 상관성을 제거하지 않는다. 게임 일반화 검증이나 강도 평가가 아닌, 고정 replay에 대한 fitting 상태 검사다. 오래된 데이터와 새 데이터는 정책/방문 분포가 달라 오차의 절대값을 sampler의 우열로 해석하지 않는다. 각 데이터 안에서 steps 변화만 비교한다. probe는 원래 자기대국 checkpoint나 기본 모델을 덮어쓰지 않는다.

### Fitting 결과

아래는 초기화 seed 3개의 평균 검증 오차다. fitting 반복 중의 batch loss가 아닌, 각 step 시점에서 고정된 전체 검증 split으로 계산했다.

| 고정 데이터 | 지표 | 0 steps | 16 steps | 128 steps | 512 steps |
| --- | --- | ---: | ---: | ---: | ---: |
| 기존 importance | regret MSE | 4776.06528 | 4753.47221 | 4504.18798 | 4175.86768 |
| 기존 importance | strategy KL | 0.78954 | 0.15695 | 0.11767 | 0.16315 |
| 새 fixed-opponent | regret MSE | 4873.46792 | 4854.16793 | 4655.90488 | 4060.41936 |
| 새 fixed-opponent | strategy KL | 0.60728 | 0.35091 | 0.34744 | 0.35657 |

기존 replay의 regret MSE는 16→512 steps에서 약 12.2%, 새 replay에서는 약 16.4% 낮아졌다. 16 steps가 충분한 fitting 예산이라는 근거는 없다. 두 데이터 모두 평균 전략 KL은 512 steps가 128 steps보다 나빠졌다. 새 평균 전략 데이터의 train KL은 16→128→512에서 0.34945→0.34167→0.32166으로 내려갔지만 validation KL은 0.35091→0.34744→0.35657로 변했다. 같은 네트워크라도 두 회귀 목표에 필요한 예산은 구분해야 한다.

iteration마다 다른 정책을 라벨로 갖고 요약 특징도 충돌할 수 있으므로, 개별 표본 target에 대한 평균 KL의 달성 가능한 최솟값이 0이라고 가정하지 않는다. 평균 전략 validation plateau를 표본 수집 방식 하나의 실패로 단정할 수 없다. 이 probe는 강도 개선을 입증하지 않는다.

기존 데이터 기록: `artifacts/r1-fitting/20261106T020908760584Z/`. 새 데이터 기록: `artifacts/r1-fitting/20261106T021248405335Z/`. 각 실행은 모델 초기화 3개 × 네트워크 2종 × 측정 시점 4개=24개 측정 결과를 저장했다.

## 가중치와 자원 결과

| Seed | 기존 표본 / ESS | 새 표본 / ESS | 기존 → 새 상위 1% 점유율 | 새 학습 시간 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 20,000 / 37.74 | 7,409 / 5640.15 | 89.26% → 2.01% | 223.4초 |
| 43 | 20,000 / 18.00 | 7,176 / 5482.08 | 90.90% → 1.92% | 179.2초 |
| 44 | 20,000 / 11.95 | 7,207 / 5493.51 | 88.19% → 1.99% | 171.1초 |

역확률 가중치는 사라졌고 남은 시간 가중치는 t/H≤1이다. 가중치 ESS 증가는 구조적으로 예상된 결과다. 새 방식은 같은 trajectory 예산에서 focal만 기록하므로 총 표본 수가 줄었다. 아직 드물게 방문하는 정보집합의 표본 오차까지 해소됐다는 뜻은 아니다. 새 실행 시간은 실제 관측치이며 일부 검사/probe가 학습과 동시에 실행되어 독립 속도 benchmark가 아니다. 저장된 평균 전략 npz 12개는 모두 유한한 tensor 값을 가지며 해시가 서로 다름을 확인했다.

## 최종 대전 결과

단위는 BB/100 ± 명목 95% deal-block CI 반폭이다. 고정 상대 collector의 학습 상대 τ와 아래 대전 상대 NoRegrets는 서로 다르다.

| 평가 후보 | 손익 |
| --- | ---: |
| public | -37.32 ±126.53 |
| importance-42 | +190.13 ±819.94 |
| fixed-42 | -18.27 ±817.87 |
| importance-43 | -109.83 ±778.61 |
| fixed-43 | -195.95 ±793.84 |
| importance-44 | +208.31 ±849.73 |
| fixed-44 | -182.72 ±794.72 |

| Seed | fixed − importance 손익 차이 |
| --- | ---: |
| 42 | -208.40 ±670.35 |
| 43 | -86.12 ±518.80 |
| 44 | -391.04 ±521.59 |

가중치 집중도 개선은 확인했지만 **대전 강도 개선은 확인하지 못했다.** 새 모델의 세 점추정은 모두 기존 모델보다 불리하며, 세 paired CI는 모두 0을 포함한다. 이는 동등성이나 비열등성의 입증도 아니다. 새 collector를 기본값으로 승격하지 않았다. 공개 모델과 registry는 그대로다.

원시 4,200핸드의 좌석/deal 유일성·칩 보존·focal 손익·summary 평균·paired CI를 다시 계산해 일치를 확인했다. 모든 후보가 동일한 엔진 binary hash와 동일한 NoRegrets 모델 hash를 사용했다. `verification.json`에 기록했다.

## 다음 실행 우선순위와 이론 쟁점

1. 현재 16-step regret 회귀를 128/512 steps로 늘리는 동일 조건 ablation을 우선한다. 평균 전략 예산은 별도로 유지하여 원인을 분리한다. 고정 replay에서 더 낮은 MSE가 실제 자기대국 강도로 이어지는지는 새 실험이 필요하다.
2. fixed-opponent를 실험 옵션으로 유지하고, 정보집합/좌석/street 방문 coverage와 요약 특징의 충돌을 계측한다. 가중치 ESS만으로 선택하지 않는다. 수집량을 맞춘 비교는 같은 trajectory 예산 비교와 구분한다.
3. 강도 개선이 반복되면 새 deal set과 추가 상대에서 검증한 뒤 GPU/병렬을 검토한다.

**별도 이론 쟁점:** 완전한 정보집합에서는 고정 상대 계수가 평균에서 소거되지만, 51개 요약 특징으로 합쳐진 상태에 대해서는 동일한 투영 목표라는 보장이 없다. 공유 신경망 근사·imperfect recall·6인 균형 수렴도 미해결이다. 이 판단과 구분하여 검사·정확 재개·수집 방식 비교·fitting 진단·제한 CPU 학습·대전 평가를 완료했다.
