# R1~R3 검토 판정 및 구현 결과 감사

2026-11-06. Fullhouse 고정 commit `e504793d480b1b975f25258d25939b45c6dbd5a4`와 로컬 구현·실험 파일을 검토했다. 학습 알고리즘·현재 모델은 이번 검토에서 변경하지 않았다.

후속 작업: R1 지적을 별도 CPU 기준 학습기에 반영했다. 원본에 대한 아래 감사 사실은 유지되며, 수정 경로의 회귀 검사·유한 트리 평균 전략 oracle·정확 CPU 재개 및 새 학습 결과는 `r1-corrections-results.md`를 참고한다.

## 판정

| 항목 | 판정 | 다음 행동 |
| --- | --- | --- |
| R1 학습 수식 | **표준 Deep CFR과 동등하다고 승인할 수 없음. 구체적인 목표값/가중치 차이 확인** | 공개 모델의 추론·평가는 계속 활용. 큰 새 학습 전에 목표값 정규화, sampling/averaging, seed를 수정·검증 |
| R2 강도 평가 | **pilot 집계 정확, 일반적 강도 판정은 미완료** | 고정된 held-out 상대·여러 seed/학습 run으로 평가 확대. 프로급 주장/자동 승격 근거 아님 |
| R3 코드·가중치 | **자체 코드 MIT 확인, 외부 엔진 MIT 추가 확인. 가중치 범위 미확정** | 재사용 코드의 고지 보존. 가중치를 우리 배포물에 포함할 권한은 확정하지 않음 |

## R1: 확인된 문제와 설계 차이

### A. 샘플별 regret 평균 제거는 전략을 바꾼다 — 우선 수정

위치: upstream `training/train.py:327`~`336`. 각 샘플의 합법 행동 regret에서 **행동별 산술 평균을 빼고 표준편차로 나눈다**. CFR regret는 현재 전략에 대한 가중 평균이 0이지, 행동별 산술 평균이 0일 필요는 없다. 평행 이동은 regret matching의 양수 부분을 바꾼다. 공통 양의 상수를 곱하는 것과 다르다.

실제 `train_network()`의 MSE target을 가로채 확인했다:

- 현재 전략 σ=(0.1, 0.1, 0.8), 원래 regret=(2, 1, -0.375). σ·regret=0으로 유효한 instant regret 예다.
- 원래 양수 regret 비율: **(2/3, 1/3, 0)**.
- 원본 코드가 만든 target: 약 (1.15549, 0.12839, -1.28388).
- 이 target을 완벽하게 예측해도 regret matching 비율: **(0.9, 0.1, 0)**.

실제 신경망이 항상 이 확률을 낸다는 의미는 아니다. **회귀 목표 자체가 표준 regret와 다름**을 입증한 것이다. 샘플마다 다른 표준편차로 나누는 것 역시 여러 반복의 regret 기여 크기를 바꾼다.

권고: 초기 기준 구현은 이미 traversal에서 적용한 고정 `/100` 칩→BB 변환만 유지하고, 샘플별 centering/표준화는 제거한 별도 학습 분기를 비교한다. 추론용 기존 가중치에 사후 수정을 가해서 복구할 수는 없다. 수정 전후는 새 모델로 관리한다.

### B. 학습 가중치가 배치 구성에 의존한다

위치: `training/buffer.py:101` 이후 `sample()` 및 GPUBuffer.sample(). 사용 가중치는 `(iteration / batch_max_iteration)^1.5`다. 표준적인 선형 iteration 가중치와 다르고 배치별 최대값이 확률적으로 달라진다.

실제 버퍼 실험: iteration 1과 4 샘플을 각각 단독으로 뽑으면 둘 다 weight=1. 같은 배치에서는 (0.125, 1). 작은 예산에서 배치/플레이어 선택에 따라 의도한 시간 가중치가 변할 수 있다. 이 자체가 실력 저하의 증거는 아니지만 명시적인 목적함수 변경이다.

권고: iteration 가중치 exponent를 명시적 설정으로 하고, 배치에서 독립적인 고정 normalization을 쓴다. 기본 비교선은 논문에 맞춘 선형 가중치로 둔다. 손계산 가능한 데이터에서 실제 optimizer 입력까지 검증한다.

### C. 상대 exploration은 다른 상대 전략을 학습 대상으로 만든다

위치: `training/traverse_fast.py:223` 이후, `traverse_numba.py:675` 이후. 상대 샘플링 확률을 q=(1-ε)σ+ε·uniform으로 바꾸고, 반환값/하위 업데이트에 σ/q 보정 없이 사용한다.

예: σ=(0.9,0.1), ε=0.2이면 q=(0.82,0.18). 자식 가치 (0,10)의 기대값은 σ 아래 1.0, q 아래 1.8이다. `audit_fullhouse_research.py`가 실제 `_mix_with_uniform`으로 이 값을 재현한다.

**의도적으로 exploration을 섞은 상대에 대한 regret를 학습하는 설계라면 이것만으로 구현 오류라고 단정하지 않는다.** 그러나 원래 σ에 대한 unbiased external-sampling CFR이라는 설명은 성립하지 않는다. 우선 ε=0 기준을 만들고, exploration을 별도 학습 목표로 둘지 off-policy sampling으로 보정할지 선택해야 한다. 비율 하나만 곱하는 수정은 하위 regret·전략 샘플에도 영향을 주므로 작고 정확히 풀리는 게임에서 검증한다.

### D. 평균 전략과 네트워크 공유는 별도 검증이 필요하다

전략 샘플은 traverser 노드에서 own reach와 함께 저장된다. 상대를 샘플링해서 그 노드에 도달할 확률도 표본 빈도에 들어간다. 반복별 상대 reach가 달라질 때, own reach만 곱하는 이 방식이 원하는 평균 전략을 복원하는지는 기존 테스트가 증명하지 않는다. `test_strategy_weights.py`는 가중치가 worker→buffer→optimizer로 전달되는지 검사한다. 이론적으로 올바른 가중치인지까지 검사하지 않는다.

공유 advantage network 하나, 플레이어별 경험 버퍼, 51개 요약 특징은 확인했다. 좌석·보드·행동 이력을 요약하는 실용적 함수 근사이며 원래 정보집합과 완벽히 일치하지 않는다. 이것들을 곧바로 버그라고 부르지는 않지만, 2인 zero-sum·perfect recall의 CFR 수렴 보장을 이 구현의 6인 결과에 그대로 적용하지 않는다.

### E. seed·실행 경로도 문서보다 제한적이다

- 실제 6→12 pilot은 workers=1이어서 `train.py:593` 부근의 **Python `run_traversals_fast`**를 사용했다. `_init_numba()`의 워밍업 메시지는 Numba 순회를 통한 학습 완료 증거가 아니다. Numba 경로는 별도 parity 테스트로만 확인됐다.
- `--seed`는 `np.random.default_rng(seed)`를 초기화한다. 전역 NumPy와 PyTorch의 초기화는 하지 않는다. 따라서 새 학습도 단일 CLI seed만으로 완전히 재현되지 않는다.
- workers>1 seed는 `iteration*1000+worker_idx`이고 CLI seed가 포함되지 않는다. 학습 seed 변경만으로 독립적인 deal stream이 만들어진다고 가정하면 안 된다.
- RNG checkpoint 없음, upstream checkpoint는 atomic write 아님, GPU cache 18GB 기준 고정. 현재 wrapper는 CPU로만 실행했으므로 실제 GPU 경로는 미검증이다.
- Numba는 순회 결과 배열에 `n_traversals*60` 용량 한도가 있고 초과 시 warning이 있다. 대규모/깊은 tree에서 샘플 누락을 검출하고 실패 또는 동적 확장으로 처리해야 한다.

**R1 작업 순서:** target 정규화 기준선 → deterministic seed/재개 → 고정 시간 가중치 → ε=0 및 올바른 평균 전략의 toy-game oracle → Python/Numba 일치 → 소규모 포커 A/B. 이 후에 GPU 학습량을 늘린다. 공유망 변경이나 완전히 새로운 연구 알고리즘은 이 기준선을 만든 뒤 판단한다.

## R2: 원시 평가 재계산과 채택 기준

`artifacts/fullhouse-evaluation/20261106T013048502614Z/hands.jsonl`의 **1,800행, 모든 핸드 칩 합 0, 각 arm의 100deal×6좌석**을 확인했다. deal별 평균으로 BB/100과 1.96·SE 구간을 재계산했고 보고된 값과 일치했다. BB=100칩이므로 평균 chips/hand와 BB/100의 수치가 같다.

| focal vs 5인 field | BB/100 | 명목 95% 구간 |
| --- | ---: | --- |
| Fullhouse vs NoRegrets 1M | +152.89 | [-108.29, +414.08] |
| NoRegrets 1M vs Fullhouse | -636.83 | [-991.83, -281.84] |
| NoRegrets 1M vs 동일 모델 | +186.15 | [-352.12, +724.43] |

두 번째 구성에서는 기준 모델의 손실이 관측됐다. 첫 번째는 0을 포함한다. 상대 구성 차이 때문에 둘을 평균내어 단일 강도 점수로 만들지 않는다. 결과는 소규모 기능 pilot이며, **Fullhouse를 더 검증할 근거**이지 보편적 강함을 확정할 근거는 아니다.

다음 승격 평가 규칙을 미리 고정하는 것을 권고한다:

1. 별도의 튜닝용 상대와 held-out 상대를 정한다. 공개 Fullhouse, NoRegrets, 새 신경망 checkpoint를 각각 다른 모델 ID로 기록한다.
2. 모든 팔을 동일 deal/hero 좌석 조건에서 비교한다. 결과 차이의 CI도 deal 단위로 계산한다. 학습 seed와 평가 seed를 분리하고 독립 학습 run을 최소 3개부터 비교한다.
3. pilot에서 3개 arm에 재사용된 seed는 독립된 3개 평가 세트가 아니다. 검증용 seed는 새로 고정한다. 필요한 deal 수는 목표 개선폭/CI 폭과 pilot 분산으로 정하며, 임의의 고정 핸드 수만으로 충분하다고 선언하지 않는다.
4. 사전 지정한 핵심 상대/혼합 상대 지표에서 개선 구간을 확인하고, HU로 줄어든 상황과 3인 이상 팟의 퇴행도 검사한다. 여러 지표 중 유리한 결과만 사후 선택하지 않는다.
5. 낮은 loss, 성공적인 추론, fallback 0은 강도 지표가 아니다. 특히 신경망은 모든 입력에 확률을 내므로 테이블 lookup hit와 직접 비교하면 안 된다.

## R3: 확인된 라이선스 범위

- [Fullhouse bot LICENSE](https://github.com/advitrocks9/fullhouse-bot/blob/e504793d480b1b975f25258d25939b45c6dbd5a4/LICENSE): 저자 작성 코드(bot/training/tooling/scripts/tests/docs)는 MIT 명시. 사용·수정·배포 시 저작권 및 허가문 보존.
- 새 확인: vendored 원본의 `UPSTREAM_SYNC.json`이 가리키는 `uzlez/fullhouse-engine@a80b269`에 [MIT LICENSE](https://github.com/uzlez/fullhouse-engine/blob/a80b269/LICENSE)가 실제 존재한다. 원문을 직접 조회했다. 외부 엔진을 무조건 사용 불가로 취급할 필요가 없다. 이 저장소의 복사본에는 LICENSE 파일이 없어 재사용/배포 시 원본 고지를 함께 가져와야 한다. vendored 수정 사항·다른 의존성 고지도 각각 보존한다.
- **공개 model/preflop equity 등 `data/` 파일:** bot LICENSE 마지막 문장은 허가를 저자가 작성한 code로 한정한다. data 범위를 명확히 포함하는 별도 허가를 발견하지 못했다. 공개 접근 가능성과 명시적인 재배포 허가는 다르다. 허가가 없다고 확정한 것도 아니며, 현재 증거로 범위를 확정할 수 없다는 판정이다.
- 사용자 지시대로 외부 문의는 하지 않았다. 코드 기반 연구/연결 작업은 진행할 수 있다. 배포 목적이면 코드 라이선스와 별개로 가중치·학습 보조 데이터의 출처 및 권한을 정리한다. 직접 학습 모델이라도 preflop table 같은 기존 데이터를 자동으로 무관하다고 간주하지 않는다.

엔진 원문 조회 보관: `.cache/research-review/engine-LICENSE`. 이 검토는 파일에서 확인한 라이선스 사실과 실무 판단이며 최종 법적 권리 범위를 확정하는 문서는 아니다.

## 구현 결과 문서 감사

확인됨: 실제 공개 모델 조회, 구현 파일 존재, 14개/10개 검사 로그, 1,800행 원시 결과, 6/12 checkpoint와 모델 종류·버퍼 크기, 저장된 inspection/verification 기록.

정정/제한:

- **S1~S5 전부 완료 → 기능 원형 구현 및 제한된 검증 완료.** 새 머신 재현, 완전한 규칙 differential test, 정확 재개, atomic checkpoint, GPU 여유 VRAM 처리 등 인계 명세의 일부 조건은 미완료다.
- `test_all_streets_and_native_parity`는 Rust에서 만든 같은 상태에서 deal marker를 제거한 변형과 비교하는 검사다. 독립적인 Fullhouse 엔진에서 같은 손을 재생한 대조가 아니므로 완전한 교차 엔진 동등성으로 표현하면 안 된다.
- Fullhouse용 off-grid 베팅·side-pot 전체 경로, 모델 변조, protocol line size/오류 경계의 exhaustive 검증은 완료하지 않았다. 기존 NoRegrets의 일부 검사를 Fullhouse 전체 검사로 합산하면 안 된다.
- S5 wrapper에는 외부 checkpoint/config 호환성 검증과 RNG 복원이 없다. 이번 자체 생성 checkpoint 복사·버퍼 증가를 확인한 정도다. 정상 resume과 중단 전후 완전 동일성은 구분한다.
- `train_fullhouse_pilot.py`의 artifact 검사 subprocess는 timeout은 있지만 RSS 제한이 따로 적용되지 않는다. 현재 작은 파일은 정상 처리됐으나 전체 단계가 모두 메모리 한도로 보호된다는 표현은 부정확하다.
- 평가 로그에는 모델 hash가 있으나 engine binary hash/source revision·정규화 정책 버전을 충분히 저장하지 않는다. 재현 가능한 장기 비교 전에 보강한다.

`docs/sol-implementation-results.md`와 `PLAN.md`에 위 판정을 반영했다. 구현이 무효라는 결론은 아니다. **지금은 작동하는 연구용 연결·학습 원형이며, 검증된 강한 Deep CFR 학습 시스템으로 승격하려면 위 수정이 필요하다.**

## 증거 재현

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/audit_fullhouse_research.py
```

실제 학습 함수 target, 버퍼 가중치, exploration 기대값 및 원시 대전 재집계를 `artifacts/research-review/r1-r2-evidence.json`에 저장한다. 원본 trainer와 기존 모델을 수정하거나 재학습하지 않는다.

비교 기준 문헌: [Deep CFR, ICML 2019](https://arxiv.org/abs/1811.00164), [SD-CFR](https://arxiv.org/abs/1901.07621), [MCCFR, NeurIPS 2009](https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games), [Pluribus, Science 2019](https://doi.org/10.1126/science.aay2400). 이번 판정의 구체적 증거는 고정 코드와 위 반례/재계산이며, 이 구현의 이론적 수렴을 증명한 것은 아니다.
