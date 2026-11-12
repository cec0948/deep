# 2026-11-12 포커 모델 재학습

사용자는 이전 클라우드 환경의 모델 백업을 찾을 수 없어, 모델을 처음부터 다시 학습하고 중간 결과를 알려주며 개선이 있는 동안 계속 학습하도록 요청했다. 학습 장소는 현재 관리형 클라우드이며 CPU 사용 한도는 4 vCPU, RAM 한도는 16 GiB다. PC나 GPU를 학습에 사용하지 않는다.

## 복구 범위와 검증

- 원본 Git 기준은 `codex/poker-ai-handoff-20261106`의 `5afd1fb`다.
- 고정 upstream과 기존 패치로 학습 소스를 다시 만들었다. 이전 연장 실행의 manifest에 있는 소스·보조 코드 24개가 모두 SHA-256으로 일치했다.
- `artifacts/fullhouse-v3/20261109T034714104421Z/source`에는 복구한 **소스만** 있다. 이전 V3/V4 모델·버퍼가 복구되었다는 뜻이 아니다.
- 과거 frozen replay 검증의 압축 원시 보고서와 통과 판정을 보존했다. 잃어버린 과거 가중치와 버퍼를 다시 확인했다고 주장하지 않는다.
- 현재 환경에서 작은 게임의 전체 학습과 독립 검산을 다시 실행했다. 최초 120회 결과는 세 seed 중 72가 exploitability 0.051358로 기준 0.05를 넘었다. 이 실패를 보존하고, 세 seed를 모두 240회로 동일하게 늘리는 재검증을 결과 확인 전에 결정했다. 다른 학습 설정과 수치 기준은 유지한다. `configs/fullhouse-retraining-validation.json`에 고정했다.
- 240회 재검증은 세 seed 모두 통과했다. 독립 검산한 exploitability는 **0.017467·0.021644·0.029152**다. 원시 결과와 실패한 120회 기록은 `reports/retraining-validation/`에 함께 보존한다.
- `scripts/verify_compact_recovery.py`는 소스 일치·과거 보고서 해시·새 작은 게임의 실제 checkpoint 검산을 함께 요구한다. `verification_fixture`로 본 학습의 검사를 건너뛰지 않는다.

## 새 학습과 비교

현재 준비 설정은 `configs/fullhouse-retraining-20261112.json`이다. seed 342·343·344를 각각 빈 버퍼에서 60회 학습한다. 첫 행동 TV≤0.10, 최대 fitting 8,192회, 기존 학습률과 표본 수를 유지한다. 세 모델은 단일 thread로 병렬 학습하고, 평가가 학습과 겹칠 때는 평가 1개만 실행한다.

과거 V3 비교 모델도 없어졌으므로 새 평가에서 제외하며, 누락을 성공한 평가로 처리하지 않는다. NoRegrets 상대는 동일한 고정 소스·설정으로 100만 회 재생성했다. 병렬 실행 때문에 이전 상대와 바이너리가 동일하지 않으며, 과거 점수와 직접 비교하지 않는다. 현재 정책과 전체 시간 혼합을 30·60회, 세 상대 구성에서 같은 새 카드로 평가한다. 공개 모델 기준 3셀과 새 모델 36셀로 **39셀·468,000핸드**다.

첫 30→60회에서 공개 상대 평균 개선량의 명목 95% 구간 하한이 양수이고 3개 중 적어도 2개의 개선량이 양수이면, 이전에 승인한 방향대로 120회까지 제한 연장한다. 상대별 변화는 모두 공개한다. 그 이후의 연장은 `docs/compact-continuation.md`의 네 조건을 모두 적용한다. 평가 카드 seed는 구간마다 새로 정하고 구간 내에서는 공통 카드를 사용한다. 기본 모델의 자동 교체는 하지 않는다.

`scripts/continue_compact_learning.py`가 완료·검산된 실행만 이 규칙으로 연장한다. 이번 자동 계획의 자원 상한은 각 seed **240회**이며 60회씩 늘린다. 성능 조건 미달이나 실패 시 분석을 위해 멈춘다. `scripts/publish_compact_progress.py`는 현재 진도·저장 회차·평가 개수·외부 백업 확인 여부만 `reports/retraining-status.json`에 게시하며, 모델이나 비공개 Drive 링크를 게시하지 않는다.

## 실행과 중간 백업

`scripts/run_compact_retraining.py RUN`은 신규 학습을, `--resume`은 검증된 전체 checkpoint에서의 재개를 수행한다. 기존 supervisor·worker·evaluator의 살아 있는 프로세스가 있으면 중복 실행하지 않는다. 실행 상태는 `artifacts/compact-v4/current.json`과 RUN의 `campaign-status.json`으로 확인한다.

`scripts/snapshot_compact.py RUN`은 매 seed의 atomic checkpoint inode를 복사하고 그 checkpoint가 참조하는 정책 이력을 보존한다. 버퍼·모든 난수 상태·소스·설정·복원용 실행 코드를 함께 묶고 archive 내부 파일을 SHA-256으로 다시 검산한다. seed별 저장 회차는 서로 다를 수 있지만 각 seed 안에서는 일관된다. 복원본에는 실행 중인 PID를 기록하지 않는다. 이 snapshot에는 대전 원시 로그와 별도 평가 상대 자산이 없으므로 평가를 재현하려면 평가 자산 백업도 필요하다.

사용자가 평가 자산과 이후 학습 checkpoint를 연결된 개인 Google Drive에 비공개로 보관하도록 승인했다. 첫 평가 자산 백업은 90,125,585 bytes이며 archive 내부 11개 파일을 검증했다. Drive 업로드 후 같은 크기와 소유자만 있는 권한을 확인했다. 커넥터가 원격 체크섬을 반환하지 않아 원격 파일의 체크섬까지 검증했다고 주장하지 않는다. 비공개 Drive 파일 ID·링크는 Git에 넣지 않고 `artifacts/offsite-backups.json`에 기록한다.

외부 저장 성공과 현재 환경 내 로컬 백업을 구분한다. 로컬 archive만으로 환경 삭제 후 복구를 보장하지 않는다. 클라우드 환경이 중단되면 학습 프로세스도 중단될 수 있으며, 프로세스 실행 여부와 마지막 저장 회차를 새로 확인하고 재개한다.
