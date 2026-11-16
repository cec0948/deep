# 2026-11-16 학습 악화 원인 통제 진단

사용자는 기존 조건을 고정하고 한 조건씩 바꾸는 비교를 진행하도록 승인했다. 실행은 `20261116T004818765266Z`이며 2026-11-16 09:48 KST에 실제 fitting과 평가를 시작했다. 새 자기대전 회차는0이다. 기존360회 및 새 입력96회 모델·replay·난수 상태는 원본 경로에서 변경하지 않는다.

현재 확인(2026-11-16 09:54 KST):21개 파생 후보 fitting이 모두 완료됐고 출처·초기값·같은 opening·목적함수 비교의 같은 CDF/추출 순서·입력 비교의 같은 최종 RNG 검사가 통과했다. 실제 평가기3개가90셀 평가를 진행한다. 전체 실험이나 성능 검증이 완료됐다는 뜻은 아니다. 기존360회 및rich96회 복구본의 Drive 크기·소유자 전용 권한도 다시 확인했다. 새 결과 묶음은 평가 완료 후에만 검증·백업한다.

## 결과 확인 전에 고정한 비교

360회 표본을 사용하는 다섯 군은 기존 `compact_regret` 학습기·표본·초기 난수 상태·opening 통계를 공유한다. 기준군은 기존 kernel의 reset-and-fit 계산을 그대로 사용한다. 실제 복원한360회 모델을 다시 export해 원본 배열과 일치하는지도 먼저 확인한다.

| 군 | 좌석당 보존 표본 | 학습 steps | batch | 달라지는 조건 |
| --- | ---: | ---: | ---: | --- |
| legacy_control | 100,000 | 1,024 | 512 | 기존 fitting 기준 |
| legacy_small_replay | 5,000 | 1,024 | 512 | 표본 수만 감소 |
| legacy_fewer_steps | 100,000 | 256 | 512 | steps만 감소 |
| legacy_smaller_batch | 100,000 | 1,024 | 128 | batch만 감소 |
| legacy_raw_objective | 100,000 | 1,024 | 512 | 고정 손실 가중치만 보정 |

표본 축소는 각 seed·좌석에서 사전 고정한 독립 RNG로 뽑은 중복 없는 부분집합이다. 이 선택이 fitting RNG를 소비하지 않는지 검사한다. 계산량 비교군은 같은 전체 표본을 사용한다. 목적함수 비교는 기존 조건부 표본·스케일·CDF·실제 난수 추출 순서를 유지하고 기존 검산된 `RawObjectiveRegretSampler`의 고정 손실 가중치만 적용한다. 예전120회 목적함수 실험을 자동 재시도하는 것이 아니라,360회 부모 표본·동일 초기화 기준의 별도 통제 비교다.

입력 비교의 두 군은 **기존 새 입력96회 replay**를 공유한다. 신경망 크기·초기 파라미터·표본·표적·반복 가중치·1,024steps×batch512·학습률을 같게 두고 첫51개 이후의 입력만 끄거나 켠다. masked 군은 학습 후 입력 마스크를 첫 층 가중치에 무손실 반영해 기존 추론 코드에서도 같은 판단을 내는지 검사한다. 이 둘의 최종 RNG 상태가 다르면 실험을 중단한다.

360회 표본과96회 표본은 서로 다른 자료다. **인과 비교는 각 표본 안에서만 한다.** 두 실험군 사이의 점수 차이를 입력 하나의 효과로 해석하지 않는다. 이번 진단은 새 자기대전·탐색량 차이·온라인 수렴 효과를 측정하지 않는다. 보존된 과거51차원 자료에 없던 행동 이력을 복원했다고 주장하지 않는다.

## 평가와 판정

7개 비교군×3seed의21회 fitting, 원본120회·부모360회·기존 rich96회 기준 모델을 포함한10정책×3seed×3상대의90셀·10,800,000핸드를 평가한다. 새 카드 묶음은20,000개(2037000000~2037019999)로 기존 평가 범위와 겹치지 않음을 확인했다. 좌석·seed·정책은 같은 카드 묶음을 반복하며 독립 표본 수는20,000이다.

주 비교는 기준 대비 표본 수·steps·batch·목적함수의 네 변화와 masked 대비 full 입력 변화, 총5쌍이다. 세 상대 구성까지15개 비교의 Bonferroni 동시95% 구간과 기존 명목95% 구간을 함께 기록한다. 고정3seed별 결과도 공개하며 새로운 seed 전체에 일반화했다고 주장하지 않는다. 원본120회 대비 기존 네 채택 조건은 별도 진단으로 유지한다. 여러 후보를 같은 카드에서 비교한 탐색적 결과이므로 조건을 통과해도 자동 모델 교체나 예산 연장을 하지 않는다.

fitting은 최대2개, fitting 중 평가는1개, 실제 fitting 종료 뒤 평가는 최대3개다.4vCPU·16GiB 환경의 전체 예산은6시간, fitting 작업당45분, 평가 셀당40분이다. 원본 opening 품질 기준과 유한값·입력 출처·같은 초기화·같은 opening·공통 RNG 검사를 통과해야 한다. 실행·검증 실패나 예산 초과 시 중단하고 자동 재시도하지 않는다. fitting 완료와 전체90셀 평가 완료를 구분한다.

시작 전22개 테스트가 통과했다. 실제 부모 자료의 source/checkpoint 고정, 부분집합·마스크의 비변조, 실제 runtime으로 복원한21개 정책 묶음, 손실/gradient 검산, 필수 백업 보존 및 실패 방지를 포함한다. 이것은 실력 향상의 증거가 아니며 실제 평가 결과로 판단한다.

## 실행·백업 확인

- 최신 등록부: `artifacts/controlled-refit/current.json`
- 해당 실행의 `plan.json`, `status.json`, `fitting/*/status.json`·`result.json`, `stage-ready.json`, `evaluation/*/status.json`·`results.json`, `evaluation-summary.json`, `summary.json`, `failure.json`, `backup-ready.json`을 확인한다.
- 감독은 `run_controlled_poker_study.py`, fitting은 `controlled_poker_refit.py`, 평가는 검증된 기존 `evaluate_rich_selfplay.py`다. 실제 PID·시작 토큰·명령행과 기록 시각을 대조한다.
- supervisor가 공개 `reports/controlled-refit-status.json`과 최신 안내용 `reports/training-neural-status.json`을15초마다 로컬 갱신한다. 별도 게시기를 중복 실행하지 않는다. GitHub의 게시 시각이 오래됐으면 현재 실행의 증거로 사용하지 않는다.

기존360회·rich96회 복구본을 모두 보존한다. replay를 중복 백업하지 않으며, 전체 평가 완료 후21개 파생 정책·교사·소스·설정·요약을 묶은 `controlled-refit-policy-bundle-v1` archive 하나만 검증한다. kind는 `derived-policy-probe`, `resumable_selfplay=false`, `new_selfplay_iterations=0`이다. 실제 추론 runtime으로 복원할 수 있지만 자기대전 정확 재개용은 아니다. 원래 fitting을 재현할 때는 기존 두 복구본이 필요하며 복원 의존성을 비공개 인덱스에 기록한다.

`export_controlled_poker_probe.verify_probe`로 실제 복원을 검증하고,100MiB 초과이면 같은 helper의 `prepare_upload`로64MiB 이하 조각과 재결합 SHA를 준비한다. 모든 원격 크기·소유자 전용 권한 확인 뒤에만 인덱스에 완료 기록을 만든다. 새 보존 계획은 `plan_controlled_backup_retention.py`로 만들고 최신 파생2개를 기존 후보와 합산한다. 새 백업 검증 전에는 이전 백업을 대체하지 않는다. 첫60회·필수120회·360회·rich96회 및 복원 의존성, 최신 연구1개, 평가 자산과2GiB 목표를 유지한다. 계정·Drive ID·비공개 journal은 공개하지 않는다.
