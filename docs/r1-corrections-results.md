# R1 수정과 CPU 재학습 결과

2026-11-06. 감사 보고서의 반례를 수정 기준으로 삼아 `r1-cpu-v1` 기준 학습기를 만들었다. 원본 공개 모델과 upstream checkout은 보존했다. GPU·병렬 학습·6인 Nash 수렴을 검증했다는 의미는 아니다.

후속 학습량 확대와 중요도 가중치 진단: [R1 CPU 학습 곡선](r1-learning-curve.md). 아래는 최초 12회 기능 검증 기록이다.

## 변경

- `scripts/build_fullhouse_r1.py`가 고정 원본에서 `.cache/fullhouse-r1`을 생성한다. 수정은 `patches/fullhouse-r1.patch`에 남기고 별도 helper `native/fullhouse-r1/support.py`를 추가한다. 원본 파일은 수정하지 않는다.
- regret target의 샘플별 centering/표준화를 제거했다. traversal의 고정 BB 단위를 그대로 회귀한다. Dueling network의 value/advantage head 구조는 유지한다.
- 시간 가중치는 `iteration / 고정 전체 horizon`으로 바꿨다. exponent=1이며 배치별 최댓값에 의존하지 않는다. checkpoint 재개 시 horizon이 달라지면 거절한다.
- baseline은 opponent exploration=0, 고정 100BB, CPU workers=1로 제한한다. 다른 설정은 명시적으로 거절한다. 검증하지 않은 병렬/CUDA 경로로 넘어가지 않는다.
- Python random, 전역 NumPy, NumPy Generator, Torch CPU를 초기화·저장·복원한다. 단일 CPU thread 및 deterministic algorithms를 사용한다. optimizer는 원본처럼 매 반복 새로 만든다.
- `--stop-after`로 전체 horizon을 유지하면서 중단 지점만 선택한다. checkpoint에 revision/설정/RNG를 저장하고, seed/horizon 등 설정 불일치·파일 누락을 거절한다.
- checkpoint는 같은 디렉터리의 임시 파일에 저장·fsync 후 `os.replace`하고 디렉터리도 fsync한다. 실패 시 이전 checkpoint를 유지한다. npz export 자체는 atomic write로 바꾸지 않았으므로 배포 등록은 별도 검증 이후에 한다.

## 평균 전략 수정과 작은 게임 기준 계산

기존 traverser 방문 로그를 평균 전략 버퍼에 넣는 경로는 사용하지 않는다. 별도로 현재 고정 정책에 대해 모든 행동에 양의 확률을 주는 sampling trajectory를 생성한다. q=0.5·정책+0.5·합법 행동 uniform이며, 평균 전략 샘플에 `own_reach / sampling_path_reach`를 곱한다. 이 q는 **평균 전략의 표본 수집용**이며 regret 업데이트의 상대 exploration은 0이다.

고정된 history에서 `E_q[방문 지시자 × own_reach/q_path] = own_reach`다. 독립적으로 배분한 chance는 반복과 무관한 chance weight를 남긴다. 이 방식으로 상대 방문 확률이 반복별로 달라지는 영향을 제거한다. sampling weight를 clip해서 의도적으로 bias를 넣지 않으며 float32 범위를 넘는 경우 실패시킨다.

`tests/test_r1.py`의 exact oracle은 상대가 먼저 행동하고 한 플레이어가 이후 두 번 행동하는 유한 트리다. 두 iteration에서 상대 reach는 0.9→0.1, 자기 정책은 0.2→0.8로 바꿨다. 모든 q 경로를 열거하여 **실제 collector가 생성한 샘플**의 분자·분모를 기준 own-reach 평균과 비교했다. 첫 상태의 선형 평균 0.6을 복원했고, 뒤 상태에서도 이전 자기 행동 확률을 포함한 결과가 일치했다.

이는 평균 표본 추정식의 유한 트리 검사다. 51개 요약 특징에서의 정보 손실, 신경망 근사 오차, 6인 균형 수렴, 전체 CFR 알고리즘의 exploitability 수렴을 증명하지 않는다. inverse sampling probability 방식은 긴 경로에서 분산이 커질 수 있으며, 대규모 학습 전 weight 분포/ESS 계측과 성능 비교가 필요하다.

## 검사

전체 통합 검사 **19개 통과**: 기존 14개 + R1 5개.

R1 검사는 실제 MSE target의 (2,1,-0.375) 보존, 배치 독립 시간 가중치, 저장 실패 시 이전 파일 유지, 모든 CPU RNG 복원, 위 평균 전략 oracle이다. 기록: `.cache/r1-tests.log`, `.cache/r1-all-tests.log`.

```bash
cd /workspace/deep
python scripts/build_fullhouse_r1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python tests/test_r1.py
```

학습기별 Python module 이름 충돌을 피하려면 upstream 원본 검사는 별도 프로세스에서 실행한다. 이번 수정 검사는 생성된 R1 학습기를 대상으로 한다.

## CPU 재학습·정확 재개

실행: `scripts/run_r1_pilot.py`. 기록: `artifacts/r1-training/20261106T014513666340Z/`.

고정 horizon=12, traversals=8/iteration, workers=1, buffer 2,000/player·strategy 2,000, batch 64, 두 네트워크 각각 2 optimizer steps. 정책 전체에 대한 손익 개선 실험이 아니라 기능용 예산이다.

| 실행 | 시간 | 관측 RSS peak |
| --- | ---: | ---: |
| 12회 연속 | 6.23초 | 322.1 MiB |
| 6회 중단 | 4.83초 | 318.3 MiB |
| 새 프로세스 6→12 재개 | 4.42초 | 316.7 MiB |
| 같은 seed로 12회 반복 | 5.85초 | 318.4 MiB |

연속·재개·반복의 checkpoint를 재귀적으로 비교했다. **모델 tensor, 버퍼 배열/카운터, 설정, 저장된 RNG 상태가 정확히 일치했다.** 6→12 가중치 변화와 finite 값도 확인했다. 최종 평균 전략 샘플 1,051개다. seed 변경, horizon 변경, resume 파일 누락도 거절했고 이전 checkpoint가 유지됐다. `results.json`, `resume-guards.json`에 기록했다.

이 정확 재개 검증은 현재 라이브러리/CPU/단일 thread 환경 및 iteration 경계에 한정한다. 다른 hardware, CUDA, worker 수, 반복 도중 임의 중단까지 보장하지 않는다. 파일 byte hash의 동일성 대신 deserialized tensor/state 동일성을 검사했다.

새 모델: `continuous/iter_00012.npz`, `model_type=average_strategy`. 공개 advantage 모델을 덮어쓰지 않았고, 정상 서비스의 기본 모델로 자동 승격하지 않았다.

## 기존 모델과 대전 평가

새 평가 seed 11062026, 같은 NoRegrets 규칙 엔진, 100 deal × 6 focal 좌석 × 3 구성 = **1,800핸드**. 기록: `artifacts/fullhouse-evaluation/20261106T014555566762Z/`. 실제 새 평균 전략 npz를 로드해 평가했다.

| focal vs 5개 field 모델 | BB/100 | 명목 95% CI 반폭 |
| --- | ---: | ---: |
| R1 새 모델 vs NoRegrets 1M | -695.43 | ±697.79 |
| NoRegrets 1M vs R1 새 모델 | +624.25 | ±794.85 |
| NoRegrets 1M vs 동일 모델 | +164.68 | ±608.12 |

칩 보존/합법 액션을 확인하며 모든 핸드가 완료됐다. 실행 52.1초, 관측 RSS 약 1.38 GiB. 모든 구간은 0을 포함한다. 작은 새 모델의 점추정은 불리하며 **성능 개선을 확인하지 못했다.** 이전 공개 모델 pilot과는 모델·seed·학습량이 달라 수정 전후 A/B로 해석할 수 없다. 신경망 loss 감소나 수식 수정만으로 실력 향상을 주장하지 않는다.

runner에 `--model`, `--deal-seed` 및 candidate type/path, engine binary/runner/runtime hash 기록을 추가했다. 공개 모델과 새 모델을 별도 후보로 비교할 수 있다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/run_r1_pilot.py
# 위 명령이 출력한 RUN_DIRECTORY 사용
OPENBLAS_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/evaluate_fullhouse.py \
  --model RUN_DIRECTORY/continuous/iter_00012.npz --deal-seed 11062026 --deals 100
```

## 남은 판단과 다음 단계

- 반례 A/B/C의 기준선 수정, CPU seed/재개/atomic checkpoint, 평균 전략의 유한 트리 oracle을 완료했다. 과거 공개 모델의 학습법이나 가중치 자체를 수정한 것은 아니다.
- **이론적 판단:** 51개 요약 특징의 imperfect-recall 효과, 공유 신경망의 근사 오차, 6인 학습의 균형/강도 보장은 여전히 별도 문제다. 작은 oracle을 이 보장의 증거로 확대하지 않는다.
- **실험적 판단:** 새 평균 sampler의 중요도 분산이 실제 큰 포커 학습에서 감당 가능한지 아직 미확정이다. 분포·ESS를 먼저 계측하고 같은 자원/여러 학습 seed로 예산을 늘린다. baseline 수정과 실력 향상은 구분한다.
- GPU/병렬은 현 기준 학습기에서 의도적으로 거절한다. CPU 검증이 충분해진 뒤 global seed 혼합, worker 상태, sampling capacity/VRAM을 반영해 확장해야 한다.
- R2 강도 검증과 R3 가중치 재배포 범위 판정은 이전 보고서대로 유지한다.
