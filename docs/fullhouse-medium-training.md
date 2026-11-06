# Fullhouse 중간 규모 CPU 학습

2026-11-06. 사용자 승인에 따라 작은 기능 실험에서 학습량을 확대한 첫 seed의 실제 자기대국 학습을 시작했다. 실행 완료 여부는 고정 문장 대신 아래 상태 파일/명령으로 확인한다.

- 실행 폴더: `artifacts/fullhouse-medium/20261106T023201457778Z/`
- 최신 실행 위치: `artifacts/fullhouse-medium/current.json`
- 현재 상태·진행률·예상 잔여 시간: 실행 폴더의 `status.json`
- 학습과 평가 모두 완료된 뒤의 결과: 실행 폴더의 `results.json`

## 설정

고정 설정은 `configs/fullhouse-medium.json`과 실행 폴더의 `config.json`이다.

| 항목 | 이전 작은 실험 | 이번 실행 |
| --- | ---: | ---: |
| 학습 seed | 42/43/44 | 우선 42 하나 |
| 반복 수 | 100 | 600 |
| 반복당 regret root traversal | 32 | 256 |
| 총 regret root traversal | 3,200 / seed | 153,600 |
| 반복당 regret optimizer steps | 16 | 512 |
| 반복당 평균 전략 optimizer steps | 16 | 128 |
| Batch size | 128 | 128 |
| Regret buffer 용량 / player | 20,000 | 100,000 |
| 평균 전략 buffer 용량 | 20,000 | 200,000 |
| 체크포인트 간격 | 25 | 12 |
| CPU worker / Torch thread | 1 / 1 | 1 / 1 |

600회에서 각 플레이어가 traverser로 표본을 모으는 라운드는 100회다. 외부 sampling의 root traversal은 내부에서 여러 행동 분기를 확장하므로 완료된 포커 핸드 수와 같은 단위가 아니다. 평균 전략 수집도 별도로 반복당 256개 trajectory를 사용한다.

평균 전략 수집기는 기존 기본값 `importance`로 고정한다. 직전 `fixed-opponent` 실험은 가중치 쏠림을 줄였지만 강도 개선을 입증하지 못했다. 이번 실행은 반복/순회/회귀/버퍼 예산을 함께 늘리는 실용적인 확대 실험이며, 어떤 단일 변경의 인과 효과를 분리하는 ablation은 아니다. 6인·100BB 고정 stack, 상대 exploration=0, `r1-cpu-v2` CPU 기준선이다. 프로급 실력이나 6인 균형 수렴은 미검증이다.

## 실행과 재개

```bash
cd /workspace/deep
# 현재 실행을 조회한다.
python scripts/train_fullhouse_medium.py status
# 환경 중단/실패 등으로 실행이 멈춘 경우 동일 실행을 재개한다.
# 살아 있는 supervisor/child가 있으면 중복 실행을 거절한다.
python scripts/train_fullhouse_medium.py resume
# 별도의 새로운 실행을 시작한다. 이미 실행 중이면 거절한다.
python scripts/train_fullhouse_medium.py start
```

`start`는 분리된 supervisor를 실행한다. 터미널 명령이 종료된 후에도 실제 trainer가 계속 실행되는 것을 확인했다. 이 대화의 답변 생성 여부와 분리된 프로세스다. **클라우드 실행 환경 자체가 중단되면 프로세스도 중단된다.** 환경이 유지되는 동안 supervisor가 600회 학습과 지정된 후속 평가를 순서대로 실행한다.

실행마다 `source/`에 training/bot 소스를 복사하고 hash를 기록한다. 원본/공개 모델은 보존한다. 데이터 파일은 고정 원본을 참조하며 사용 equity 파일 hash를 확인한다. 평가기·runtime·엔진·비교 모델 및 registry hash도 확인한다. 관리 중인 소스/비교 대상이 바뀌면 후속 단계를 실패 처리하여 서로 다른 구현의 결과가 섞이지 않게 한다.

전체 horizon=600을 처음부터 고정한다. 첫 12회와 이후 60회 단위의 구간 사이에 최신 checkpoint의 모델/버퍼/모든 CPU RNG를 로드한다. 이는 검증된 동일-horizon 재개 경로를 사용한다. `latest.pt`는 임시 파일+fsync+replace로 저장하며 전체 reservoir와 RNG를 포함한다. `iter_XXXXX.pt`는 upstream 형식의 모델 checkpoint이며 reservoir를 포함하지 않으므로 **재개에는 latest.pt를 사용한다.** `iter_XXXXX.npz`는 추론용 평균 전략 모델이다. supervisor는 닫힌 구간의 checkpoint와 npz의 finite 값, 설정, RNG/버퍼 존재를 확인하고 이후 구간을 시작한다.

상태는 약 10초마다 atomic JSON으로 저장한다. `training/diagnostics.jsonl`은 실제 반복별 진단이며, 구간별 `train-*.log`에 학습 로그를 남긴다. 버퍼가 가득 찬 이후 원본 로그의 `adv_samples`/`strat_samples`는 버퍼 크기 증가분이라 0이 될 수 있다. 실제 root traversal은 반복 수×256, 평균 전략 수집량은 diagnostics의 `collection.count`/`reservoir_total_seen`으로 확인한다.

임의 중단이 반복 중간에 일어나면 가장 최근 저장 시점부터 재실행한다. 재개 시 저장되지 않은 반복의 diagnostics를 제거해 중복을 피한다. 체크포인트 구간 밖에서 optimizer 상태를 이어받는다는 뜻은 아니다. 원래 학습기처럼 매 iteration 신경망/optimizer를 초기화한다.

학습 총 wall-time 상한은 8시간, 학습 프로세스 RSS 상한은 20GiB다. 평가기는 엔진을 포함해 RSS를 확인하고 후보별 900초 상한을 둔다. 제한 초과·비유한 수치·설정 불일치 시 실패와 이유를 기록하고 기본 모델을 교체하지 않는다. 실제 예상 시간은 초기 처리량으로 보정한다. 초기 3회는 약 45초로, 약 2.5~3시간의 학습을 예상했으나 정책에 따라 순회 비용이 달라질 수 있다.

주의: 이번에 고정한 upstream 기반 trainer는 `--stop-after`로 구간을 끝내도 마지막 로그에 전체 예정 횟수인 `Training complete. 600 iterations`를 출력한다. 이 문장은 600회 완료를 뜻하지 않는다. 실제 완료 지점은 supervisor가 검사한 `status.json`의 `completed_training_iteration`과 checkpoint의 `iteration`을 기준으로 한다.

## 사전 지정한 후속 평가

학습 완료 후 다음 후보를 같은 엔진·동일 NoRegrets 1M 상대 5명·새 패 분배로 평가한다.

1. 기존 공개 Fullhouse advantage 모델.
2. 이전 seed 42의 100회 작은 평균 전략 모델.
3. 이번 실행의 120회 평균 전략 모델.
4. 이번 실행의 300회 평균 전략 모델.
5. 이번 실행의 600회 평균 전략 모델.

후보당 독립 deal 1,000개 × focal 6좌석 = 6,000핸드, 총 30,000핸드다. deal seed=99110600으로 학습 RNG/이전 평가와 분리했다. 명목 95% deal-block CI와 공개/작은 모델/이번 120회 모델 대비 paired 차이를 계산한다. 여러 후보의 점추정에서 최고값을 골랐다는 이유로 강도 개선이나 프로급을 선언하지 않는다. 자동 승격은 하지 않는다.

`results.json`은 학습·모든 평가가 정상 완료된 뒤에 생성하며 원시 평가 경로, 표본 수, 손익, paired 차이와 소스/모델 hash를 포함한다. 실패/중단 상태는 `status.json`과 `supervisor.log`에서 확인한다. 다른 seed의 재현은 이 첫 실행의 결과를 확인한 뒤 진행한다.

## 사전 확인

기존 회귀/통합 검사 21개가 통과했다(`.cache/medium-preflight-tests.log`). 이번 작업은 검증된 학습 수식을 유지하고 실행 예산·복구·후속 평가 orchestration을 추가한다. 새 학습 소스는 실행 폴더에 보존된다. 작은 게임 전체 학습 검증은 별도의 보강 항목이며 이번 실제 학습을 막는 조건으로 두지 않는다.

## 시작 확인 기록

첫 구간은 전체 horizon=600을 유지한 채 12회까지 실제 학습했다. regret root traversal 3,072회이며 구간 실행은 약 4분이다. supervisor가 저장된 평균 전략 npz와 모델 tensor의 finite 값, 전체 reservoir/RNG 존재, 설정을 확인하고 **`--resume`으로 다음 12→60 구간을 시작했다.** 따라서 학습 명령만 준비한 상태가 아니다. 현재까지 전체 600회를 완료했다는 뜻은 아니다.

이 시점의 예상 남은 학습 시간은 약 3.5시간이다. 초기 3회 추정보다 길어졌으며, 이후의 상태 파일 추정값을 우선한다. 독립 supervisor/실제 trainer의 생존과 재개 명령을 확인한 기록은 실행 폴더의 `launch-verification.json`이다.
