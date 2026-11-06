# 6인 100BB 학습 코드 개선 검토

검토일: 2026-11-06. 대상: 현재 1M 캠페인, 실행 스크립트, NoRegrets 고정 커밋 `757f7692738069522195d2b486eec60a8b010c0c`.

권장 방향은 **평가 보강 → 설정/재개 정비 → 카드 분류 캐시와 저장 비용 최적화 → 제한된 학습 확대 → 탐색 비교 → 필요한 부분만 신경망화**다. 이번 작업은 검토와 개선 계획 작성이며, 학습 알고리즘·현재 모델 교체나 추가 대규모 학습을 수행하지 않았다.

## 근거의 범위

- 직접 확인: 현재 코드, 1M 캠페인 기록, 로컬 소스의 제어 흐름, 아래 VR 보정식의 수치 반례.
- 외부 확인: 논문 저자의 Deep-CFR/DREAM/ReBeL 저장소, OpenSpiel 구현, 2025 IIG-RL-Benchmark, 2026 fullhouse-bot의 README·학습 파일·라이선스.
- NoRegrets의 `BASELINES.md` 결과는 제작자 보고다. 우리의 환경에서 재현한 결과나 독립 학술 검증으로 취급하지 않는다.
- 현재 네트워크 접근 범위에서 논문 원문 전체를 새로 대조하거나 최신 문헌을 전수 조사하지 못했다. 아래는 확인 가능한 공개 구현과 알려진 논문에 근거한 적용 검토이며, 2026년 전체 연구 중 최상의 방법을 확정하는 보고서가 아니다.

## 현재 병목

현재 모델은 딥러닝 모델이 아닌 CPU Linear external-sampling MCCFR 전략 테이블이다. 학습 1M회, 전략 약 963만 개, 모델 약 364 MB, 관측 peak RSS 약 4.1 GiB다. 순회 하나가 여러 정보집합을 방문하므로 `1M / 963만`을 정보집합당 평균 방문 횟수로 해석하면 안 된다. 실제 방문 횟수 계측이 필요하다.

합성 패널의 6인 잔존 flop 조회율은 18.75%, turn/river는 0%다. 실제 플레이 분포의 실패율은 아니지만 취약 상황의 증거다. `src/bot.rs:216`의 fallback은 학습 전략이 없으면 **항상 check/call**한다. 따라서 합법적인 행동을 출력한다는 사실과 충분히 학습했다는 사실을 분리해야 한다.

추상화는 preflop 169종, postflop 12 bucket이다. 세부 소스를 확인하면 현재 비전략 추상화의 river는 `river_bucket`의 정확한 대 1인 무작위 패 상대 equity 분류를 사용한다. 설정의 `rollouts=64`만 보고 모든 river 분류가 MC 64회라고 해석하면 안 된다. 이 정확성은 상대의 실제 range나 다인 전략 가치까지 정확하다는 뜻은 아니다.

## 우선순위와 구현 대상

| 순위 | 개선 | 코드/실험 대상 | 채택 기준 |
| --- | --- | --- | --- |
| P0 | 평가 강화 | `evaluate_selfplay.py`, `measure_coverage.py`, upstream `lbr`, `br`, `aivat` | 독립 학습 seed, 양방향 crossplay, HU·다인 취약성, 실제 대국 중 street/잔존 인원별 fallback을 함께 보고 |
| P0 | 설정·재개 일치 | `train_selfplay.py`, upstream checkpoint와 CLI | 설정 파일과 실제 명령 일치, 재개 후 규칙·추상화·학습 옵션 동일, 중단 단계도 추적 가능 |
| P1 | 카드 분류 사전 계산 | upstream `bucket-table`, `Abstraction::preload`, 재개 분기 | 기존 분류와 일치하고, 사전 계산·로드 비용 포함 총 시간이 감소 |
| P1 | 저장·메모리 계측 | upstream checkpoint loop, 캠페인 기록 | 학습/저장/로드 시간을 분리하고 20 GiB 제한 안에서 비용 대비 개선 |
| P1 | 제한된 학습량 확대 | 우선 1M→3M→10M 후보, 독립 학습 seed 3개 | 신뢰구간·후반 취약성·동일 시간 예산 평가를 통과한 모델만 승격 |
| P2 | 온라인 탐색 연결 | JSON adapter → upstream `Policy`/`RangeTracker` | blueprint 대비 수익률/취약성 개선, 응답시간 p50/p95 기록, 공개 관측만 사용 |
| P3 | 신경망 도입 | 가치망 또는 Deep CFR 별도 실험 | GPU 이득과 6인 성능을 실측; 학습 loss만으로 채택하지 않음 |

3M/10M는 다음 실험 후보이지 현재 메모리에서 실행 완료가 보장된 설정이 아니다. 다중 seed는 한 번에 하나씩 실행하고, 기존 1M 모델은 비교 기준으로 유지한다.

## 구체적으로 발견한 코드 문제

### 1. 설정 파일이 실제 실행의 유일한 기준이 아니다

`scripts/train_selfplay.py:58`은 `--players 6 --stack 10000`을 하드코딩한다. JSON에는 players, stack_chips, blinds, abstraction이 있지만 일부는 CLI에 전달되거나 지원 범위로 검증되지 않는다. **현재 캠페인의 값은 하드코딩과 같으므로 기존 결과가 틀렸다는 뜻은 아니다.** 이후 설정을 바꾸면 기록과 실행이 달라질 위험이 있다.

개선: `--config` 지원, 지원하지 않는 blind/abstraction 값 거부, 실제 명령과 기본값을 포함한 유효 설정 저장. 학습 seed·평가 seed·실험명 분리. 학습 의미를 바꾸는 설정과 단계/시간 제한 같은 실행 계획을 구분하여, 의미가 동일한 캠페인에 후속 milestone을 안전하게 추가할 수 있게 한다.

### 2. 카드 분류표는 재개 경로에서 로드되지 않는다

upstream `src/main.rs:563`의 `Some(resume)` 분기는 `load_checkpoint`만 호출한다. `--bucket-table`을 읽고 검증하는 부분은 새 학습 분기에 있다. `src/cfr.rs:847`의 로더는 추상화 설정과 centroid를 복원하지만 사전 계산 캐시는 복원하지 않는다. 따라서 현재 코드에 옵션만 추가해서는 재개 단계의 캐시 재사용이 되지 않는다.

개선: 저장된 추상화와 분류표의 설정 **및 centroid 일치**를 확인하고, 새 학습·재개 모두 같은 preload 경로를 사용한다. 캐시가 전략을 바꾸지 않는지 샘플 조회 동등성 검사를 먼저 한다. 기존 checkpoint의 centroid를 새 표로 교체하면 안 된다.

제작자 보고: canonical flop/turn table 약 1,525만 항목, 파일 152 MB, 별도 환경의 10M 실험에서 92,630→248,660 순회/초(약 2.7배). 우리의 4 CPU 환경에서 보장되는 수치가 아니며 사전 계산 시간·RAM을 포함해 다시 측정해야 한다.

### 3. 체크포인트 저장이 학습 비용을 키울 수 있다

upstream `src/main.rs:680`은 `--checkpoint` 사용 시 반복 수를 약 20개 구간으로 나누고 매 구간 전체 checkpoint를 저장한다(최소 구간 100K). 현재 마지막 750K 추가 학습도 한 번만 저장하는 것이 아니다. 큰 테이블에서는 직렬화와 디스크 쓰기가 부담이 될 수 있다.

개선: 경과 시간/반복 수 기반 저장 주기 옵션, atomic rename, 복구 가능한 최신 checkpoint와 평가용 milestone 구분. 절감 효과를 재려면 저장 시간·쓰기 바이트·학습 시간 분리 기록이 먼저다. checkpoint에는 전체 TrainConfig가 없으므로 메타데이터와 해시를 계속 함께 관리해야 한다.

### 4. VR-MCCFR와 비정책 샘플링을 동시에 켜면 보정식 점검이 필요하다

`src/cfr.rs:668`은 `v = ratio * child_value`를 만든 뒤 `:680`에서 `expected + (v - b[idx])`를 반환한다. `ratio=σ(a)/q(a)`일 때 필요한 제어변량 형태는 `expected + ratio * (child_value - b[a])`다. 현재 식은 baseline 항에 ratio가 빠져 있다.

고정 baseline 반례: σ=(0.5,0.5), q=(0.25,0.75), b=(1,3), child_value=(1,3). 목표 기대값은 2지만 현재 식의 기대값은 1.5, 보정한 식은 2다. Python 산술로 확인했다. baseline 갱신에 비율이 곱해진 값이 들어가는 부분과 병렬 접근도 별도 검토 대상이다.

**현재 캠페인은 `vr_baseline=false`, `multiway_focus=0`이므로 이 조합 문제를 겪지 않는다.** 해당 기능을 함께 활성화하기 전에 작은 게임에서 기대값·regret 추정 회귀 검증을 해야 한다. 현재 모델의 학습 오류로 확대 해석하지 않는다.

## 논문에서 가져올 것과 적용 한계

1. **Brown & Sandholm, Superhuman AI for multiplayer poker, Science 2019 (Pluribus).**
   [논문](https://doi.org/10.1126/science.aay2400). 6인 목표에 직접적인 근거다. blueprint + 추상화 + 가지치기 + 실행 중 탐색의 조합을 참고한다. NoRegrets는 독립 구현이며 원본 Pluribus의 실력을 이어받았다고 볼 수 없다. 현재 기본 Pluribus식 pruning과 coarse menu는 이미 켜져 있다.

2. **Brown & Sandholm, Solving Imperfect-Information Games via Discounted Regret Minimization, AAAI 2019.**
   [논문](https://arxiv.org/abs/1809.04040), [OpenSpiel 구현](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/discounted_cfr.py). 현재 Linear CFR은 관련 계열이다. DCFR은 양·음 regret 및 평균 전략 할인 방식이 달라 단순 이름 변경이 아니다. 샘플링·가지치기와의 상호작용을 검증해야 한다. OpenSpiel 파일 자체에도 논문 결과 재현 미확인 경고가 있다. 2인 zero-sum에서의 보장을 6인 Nash 수렴 보장으로 옮기지 않는다.

3. **Schmid et al., Variance Reduction in Monte Carlo Counterfactual Regret Minimization for Extensive Form Games using Baselines, AAAI 2019.**
   샘플링 분산 감소의 근거다. 기존 엔진에 구현이 있으므로 새로 만들기 전에 위 보정식과 실제 학습 비용을 확인한다. 제작자 실험에서는 현재 희소 테이블 조건의 VR 옵션이 약 3.3배 느리고 성능도 개선되지 않았다.

4. **Burch et al., AIVAT, AAAI 2018; Lisý & Bowling, Equilibrium Approximation Quality of Current No-Limit Poker Bots, 2017.**
   AIVAT은 평가의 분산 감소, LBR은 취약성 탐색에 활용한다. 엔진에 둘 다 있으므로 재사용 후보다. AIVAT이라는 이름만으로 불편성/분산 감소가 검증되는 것은 아니며 raw 추정치와 평균 일치 및 실제 비용을 확인한다. 평가기는 실제 개인 카드를 참조할 수 있어도 행동 정책에 유출되면 안 된다. 6인 multiway LBR의 수익을 전체 게임의 정확한 exploitability나 NashConv라고 부르지 않는다. probe와 고정된 상대 구성에서 발견한 손실로 보고한다.

5. **Brown et al., Deep Counterfactual Regret Minimization, ICML 2019; Steinberger, Single Deep CFR, 2019.**
   [Deep CFR 논문](https://arxiv.org/abs/1811.00164), [SD-CFR 논문](https://arxiv.org/abs/1901.07621), [저자 구현](https://github.com/EricSteinberger/Deep-CFR). 거대한 테이블을 신경망으로 근사하는 후보다. RTX 4070에서 작은 네트워크·미니배치 학습 실험은 가능하지만 CPU 순회와 replay buffer RAM도 필요하다. 저자 구현은 오래된 PyTorch 0.4.1 의존성을 안내하며, 실험 코드는 바로 쓰는 강한 6인 100BB 모델이 아니다. SD-CFR의 과거 advantage 모델 보관 비용도 함께 계산한다.

6. **Steinberger, Lerer & Brown, DREAM, 2020.**
   [논문](https://arxiv.org/abs/2006.10410), [저자 구현](https://github.com/EricSteinberger/DREAM). advantage baseline을 이용하는 model-free regret 학습 후보다. 공개 재현 실험은 Leduc/FHP 중심이다. 우리에게는 시뮬레이터가 있으므로 model-free라는 이유만으로 현재 external sampling을 교체할 필요는 없다.

7. **Brown et al., ReBeL, NeurIPS 2020.**
   [논문](https://arxiv.org/abs/2007.13544), [공식 구현](https://github.com/facebookresearch/rebel). belief state 가치망 + 탐색의 설계 참고다. 공식 공개 코드는 Liar's Dice다. 현재 NoRegrets의 `valuenet.rs`도 두 range → 두 플레이어 가치 벡터의 **heads-up turn 모델**이다. 6인 시작 후 HU로 줄어든 상황에 쓸 수 있는 후보지만, 3~6명이 남은 상황을 해결하는 가치망으로 간주할 수 없다.

8. **Rudolph et al., Reevaluating Policy Gradient Methods for Imperfect-Information Games, 2025.**
   [논문](https://arxiv.org/abs/2502.08938), [저자 벤치마크](https://github.com/nathanlct/IIG-RL-Benchmark). 더 최근의 비교·평가 참고 자료다. 공개 저장소는 OpenSpiel 게임과 Phantom Tic-Tac-Toe/Dark Hex의 정확 exploitability 계산 등을 제공한다. 이 자료를 6인 100BB에서 policy gradient가 MCCFR보다 강하다는 근거로 사용하지 않는다.

## 바로 켜지 않을 옵션

제작자의 고정 커밋 `BASELINES.md`에는 다음 실패 결과가 남아 있다. 모두 외부 보고이며 우리 환경의 확정 결론은 아니다.

- `--multiway-focus 0.3`: 30M에서 multiway 개선은 불확실했고 HU probe는 악화. 다인 조회율만 올리는 것을 목표로 채택하지 않는다.
- bucket 12→50: 동일 500M에서 다인 성능 악화 사례. 전략 공간이 커지는 만큼 학습이 분산된다.
- `--vr-baseline`, `--snapshot-avg`: 해당 실험 예산에서 성능·시간 악화.
- `--linear-cap`: HU와 multiway 사이 tradeoff. 기존 값을 무조건 바꾸지 않는다.
- 무제한 학습: 500M→2B에서 다인 개선 정체, HU 악화 보고. 반복 수만 늘리면 계속 강해진다는 가정은 부적절하다.
- 온라인 탐색: 내부 상대에서는 개선 보고가 있지만 외부 Slumbot에서는 악화한 결과도 있다. range 추정 오류와 부족한 탐색 예산을 함께 검사해야 한다.

## 새로 찾은 재사용 후보

[advitrocks9/fullhouse-bot](https://github.com/advitrocks9/fullhouse-bot)는 2026년 공개된 6-max Deep CFR 구현으로, README에 학습 CLI, 51개 특징/5개 행동, NumPy 추론, 대회 본선 15/64 결과를 보고한다. 이번에 README·`training/train.py`·LICENSE를 읽었고 실행 및 대회 결과 독립 검증은 하지 않았다. 학습 가중치의 실제 배포/사용 가능성도 추가 확인해야 한다.

자체 작성 부분은 MIT지만 `engine_vendored/`는 별도 upstream 조건이다. 현 엔진을 즉시 교체하는 후보보다 **신경망 학습·추론 구조 참고 및 독립 상대 후보**로 검토할 가치가 있다. 대회 순위는 프로급 보증이 아니다.

## 다음 실험의 구체적 순서

1. 설정·체크포인트 정비, 실제 대국 fallback/인원별 계측, 작은 HU·다인 LBR pilot으로 평가 비용 산정.
2. 동일 centroid의 카드 분류표를 새 학습·재개 모두 재사용. warm/cold 로드와 생성 시간을 포함해 속도 비교.
3. 현재 추상화를 유지해 독립 학습 seed 3개, 1M/3M/10M 후보를 자원 제한 안에서 비교. 평가 seed는 공통으로 맞추되 학습 seed와 구분한다. A/B는 같은 deal/좌석 조건에서 비교하고 CI는 독립 deal 또는 seed 블록 단위로 계산한다.
4. 마지막 평가 세트는 튜닝에 쓰지 않는다. multiway 취약성, HU 퇴행, fallback, 수익률, RAM, 시간 전체로 모델 승격을 결정한다. 3개 학습 seed 결과는 예비 근거이며 최종 확인은 더 많은 seed/held-out 상대에서 한다.
5. 선택 모델에 탐색을 연결하고 예를 들어 100/300/1000ms 예산을 비교한다. 기존 blueprint용 JSON의 임의 hidden-card 재구성을 탐색 입력으로 그대로 쓰면 안 된다. 본인 카드와 공개 기록으로 range를 추적하고 숨은 카드를 그 분포에서 재표본화해야 한다.
6. CPU/RAM 한계 또는 가치 추정 병목이 실측으로 확인되면 4070용 가치망/Deep CFR 실험을 별도 시작한다. GPU로 테이블 순회가 자동 가속되는 것은 아니다.

신경망으로 전략 테이블을 압축하는 것만으로 원래 전략이 강해지지는 않는다. 목표는 논문 기법 수를 늘리는 것이 아니라, 동일 시간·비용에서 실제 6인 플레이의 손실을 줄이는 것이다.
