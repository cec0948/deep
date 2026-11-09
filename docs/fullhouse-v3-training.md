# 엔진 수정 후 새 버퍼로 진행하는 3-seed 학습

2026-11-09. 사용자가 요청한 엔진 수정 → 평균 전략 학습 안정화 → 새 버퍼 재학습을 `r1-cpu-v3`로 구현했다. 실행은 `artifacts/fullhouse-v3/20261109T034714104421Z/`이며, seed 42·43·44를 각각 빈 reservoir와 새 신경망에서 시작했다. **진행 상태는 `status.json`과 상태 명령으로 확인한다. 이 문서의 구현 검증은 강도 개선 판정과 구분한다.**

기존 600회 실행과 원본 upstream은 보존했다. [이전 원인 조사](fullhouse-regression-diagnosis.md)의 잘못된 게임에서 수집한 replay를 새 학습에 섞지 않는다. 기존 버퍼는 평균 전략 fitting 방식을 비교하는 별도 진단에만 사용했다.

## 게임 엔진과 추론 특징

지원 범위는 **6명, 매 핸드 전원 100BB, SB 0.5BB / BB 1BB, 앤티·레이크 없음**이다. 서로 다른 시작 스택은 명시적으로 거절한다.

- 짧은 올인 뒤 이미 행동한 플레이어도 추가 콜 금액이 있으면 다시 행동한다. 콜 의무와 레이즈 권리를 구분한다.
- 한 명만 칩을 남기고 베팅 라운드가 끝나면 나머지 보드를 공개하고 정산한다.
- 3명 이상 동률일 때 남는 칩을 한 명에게 몰아주던 처리를 고쳤다. 참조 엔진과 같은 좌석 순서로 한 칩씩 배분한다.
- 작은 opening all-in도 마지막 실제 bettor로 기록하고, full raise 횟수와 구분한다.
- 학습 상태와 JSON 추론 상태에서 같은 contribution·aggression 특징을 만든다. 콜은 추가 납입액, 레이즈는 해당 스트리트 총액으로 해석한다. 명시적인 deal 기록과 기록이 없는 형식을 모두 검사했다.
- terminal마다 eval7 카드 객체 52개를 다시 만들던 작업을 불변 카드 캐시로 바꿨다. 카드 순서·보상·RNG는 바꾸지 않았다.

NoRegrets의 고정된 Rust 규칙 모듈을 별도 `poker-rules` 실행 파일로 호출해 Python 학습 엔진과 비교한다. **과거 반례 86개와 새 2,000핸드의 38,316개 상태에서 행동 순서, 금액, 51개 입력 특징, 합법 동작, 최종 손익이 일치했다.** 3-way odd chip과 사이드팟 정산·자동 runout은 별도 수치 fixture로도 검사했다. 가변 스택 전체 게임 규칙을 검증했다는 뜻은 아니다.

## 평균 전략의 학습 안정화

기존 중요도 collector와 시간 가중치 `w_i = (iteration_i / horizon) × importance_i`는 유지한다. 평균 전략 신경망을 맞추는 데이터 추출 방식과 optimizer 후처리를 바꿨다.

1. 입력 특징으로만 상태 그룹을 정한다. 최초 프리플롭 위치 6개, 그 밖의 street × pot 크기 그룹이다.
2. 양의 가중치가 있는 그룹을 균등하게 선택한다. 그룹 안에서는 `w_i`에 비례하여 행을 뽑고, 가중치 없는 cross-entropy를 계산한다.
3. 평균 전략은 매 반복 512 optimizer steps, batch 512로 학습한다. 마지막 절반 256 steps의 신경망 파라미터를 산술 평균해 저장한다(SWA 방식). 이는 CFR의 시간 평균 전략과 별개인 optimizer 안정화다.

같은 입력 벡터는 같은 그룹에 속하므로 **그 입력의 정확한 경험적 조건부 평균은 보존**된다. 그러나 유한한 공유 신경망에서 서로 다른 상태를 얼마나 잘 맞출지 결정하는 투영 목적함수는 바뀐다. 중요도 가중치를 잘라내지는 않는다. collector의 높은 분산·낮은 ESS와 정보 손실을 해결한 것으로 해석하지 않는다.

큰 가중치·0 가중치가 있는 유한 데이터에서 조건부 평균을 직접 계산해 일치 여부를 검사했다. 저장 RNG를 복원하면 sampler가 뽑는 행도 정확히 같다.

고정된 과거 replay에서 초기화 seed 7·8·9로 512 steps씩 새로 fitting한 결과:

| 초기화 seed | UTG 169종 입력의 가중 TV 오차 | AA 폴드 | AA 레이즈 합계 |
| --- | ---: | ---: | ---: |
| 7 | 0.05100 | 4.91% | 63.35% |
| 8 | 0.05273 | 5.59% | 63.76% |
| 9 | 0.05413 | 5.15% | 66.15% |

AA 표적은 폴드 1.75%, 레이즈 68.14%이며 기존 600회 모델은 12.66%, 16.76%였다. 사전 설정한 fitting 기준(TV ≤0.08, AA 폴드 오차 ≤5%p, 레이즈 오차 ≤10%p)을 세 seed 모두 통과했다. plain stratified PPS의 첫 실험에서는 한 seed가 기준을 넘었고, 이를 보고 SWA를 추가했다. 최종 비교는 각 예산마다 초기화부터 독립적으로 다시 실행했다. 이 검사는 동일 버퍼의 재현 능력이며 holdout 일반화나 포커 최적 전략의 증명이 아니다.

## 재학습·평가 예산

[설정](../configs/fullhouse-v3.json)을 실행 전에 고정했다.

| 항목 | 값 |
| --- | --- |
| 독립 자기대국 seed | 42, 43, 44 |
| 학습 구간 | 12 → 60 → 120회 |
| 시간 가중치 horizon | 600 고정; 이번 실행은 120회에서 종료 |
| regret root traversals | 회당 256 |
| regret fitting | 회당 512 steps, batch 128 |
| 평균 전략 fitting | 회당 512 steps, batch 512, stratified PPS + SWA |
| reservoir | regret 좌석별 100,000; strategy 200,000 |
| 저장 | 12회 간격; 전체 버퍼·모델·RNG를 atomic replace로 저장 |
| CPU | seed별 단일 worker/thread, 3개 프로세스 병렬 |
| 자원 제한 | 학습 구간별 3,600초, 평가별 1,800초; child RSS 합계 20GiB |

기존과 같은 600 horizon을 유지해 시간 가중치 설정을 명확히 했다. `120회`는 프로급 모델 완성 목표가 아니라 수정 효과를 비교하기 위한 제한된 실험이다. GPU나 유료 서버는 사용하지 않는다.

평가 상대는 각각 **NoRegrets 1M 5명**, **공개 Fullhouse advantage 5명**이다. 후보는 공개 모델, 기존 평균 전략 120·600회, 새 세 seed의 평균 전략 60·120회, 새 세 seed의 120회 advantage다. 12개 후보 × 2개 상대 풀 × 2,000 deals × 6좌석 = **288,000핸드**를 예정했다. deck seed는 이전 진단과 다른 `160261109`부터이며 모든 후보에 같은 카드와 좌석을 적용한다.

주 비교는 각 seed의 최종 평균 전략 − 기존 120회, 그리고 새 120회 − 60회다. 6좌석을 한 deal block으로 묶은 paired 차이와 명목 95% 구간을 보고한다. 세 학습 seed 모두 결과를 공개하고 최상위 seed만 선택하지 않는다. 여러 비교의 구간은 동시 신뢰구간이 아니다. 고정 상대에 대한 성과를 6인 균형·프로급 성능으로 일반화하지 않는다. 기본 모델 자동 승격은 하지 않는다.

## 재현·재개·로컬 추론

```bash
cd /workspace/deep
python scripts/train_fullhouse_v3.py status
```

실행 중이면 유지한다. 프로세스가 사라진 중단 실행만 다음으로 재개한다.

```bash
python scripts/train_fullhouse_v3.py resume
```

학습 소스는 run의 `source/`에 고정하고, 관리 스크립트·추론기·평가기·규칙 엔진·비교 모델도 SHA-256으로 보호한다. 실행 도중 이 파일들을 수정하면 다음 구간을 거절한다. 12회 연속 실행, 6회+재개 6회, 같은 seed 반복 실행의 모델·버퍼·RNG·설정이 모두 정확히 일치했다. V2 checkpoint 재개는 거절한다. 추론용 npz와 버퍼가 없는 iteration pt는 정확 재개용 파일이 아니다.

새 모델을 명시적으로 지정해 JSONL로 호출할 수 있다.

```bash
run_dir=artifacts/fullhouse-v3/20261109T034714104421Z
python scripts/fullhouse_v3_runtime.py \
  --source "$run_dir/source" \
  --model "$run_dir/seed-42/iter_00120.npz"
```

`scripts/run_poker.py --backend fullhouse`의 공개 모델 기본값은 그대로다. V3 runtime을 같은 프로세스에서 생성해도 기존 모델의 특징 처리와 출력이 달라지지 않는 검사를 통과했다.

원본 환경에서 검사를 재현하는 명령:

```bash
python scripts/build_fullhouse_v3.py
python scripts/build_rules_oracle.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python scripts/verify_fullhouse_v3.py \
  --hands 2000 --output artifacts/fullhouse-v3-verification/recheck.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python -m pytest tests -q
```

새 환경에서는 먼저 [작업 재개 안내](continue-work.md)의 의존성과 백업을 복원한다. 전체 새 캠페인 준비는 `python scripts/train_fullhouse_v3.py start`이지만, 기존 실행이 있을 때 중복 시작하지 않는다. 과거 비교 모델·NoRegrets blueprint·검증 gate 파일도 필요하다.

## 남은 해석상의 한계

51개 요약 특징의 imperfect recall, 공유 advantage 근사, 다인 CFR의 수렴 보장은 별도 쟁점이다. 이번 변경은 규칙의 수치 반례와 확인 가능한 적합도 문제를 해결하는 구현이다. 엔진 수정·상태 재가중·학습량 증가·SWA를 함께 적용했으므로, 최종 성능 차이를 각각 몇 BB/100씩 기여했는지 분해할 수는 없다. 성과가 확인되어도 이후 한 요소씩 제거하는 ablation이 필요하다.
