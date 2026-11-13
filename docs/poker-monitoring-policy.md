# 포커 학습 점검·백업 기준

예약 작업은 최신 실행의 진도·평가·실패·완료와 백업을 확인한다. 학습은 별도의 클라우드 프로그램이 수행한다. 새 변화가 있을 때만 한국어로 짧게 알리고, 이미 보고한 완료 결과를 반복 알리지 않는다.

## 최신 실행 확인

- 저장소는 `cec0948/deep`, 브랜치는 `codex/poker-retrain-20261112`다. `docs/retraining-20261112.md`의 최신 실행 안내와 `reports/training-neural-status.json`을 먼저 읽는다. 공유 신경망 검증은 `artifacts/neural-information-gate/current.json`, 이후 새 입력 자기대전은 `artifacts/rich-selfplay/current.json`으로 구분한다. 실제 작성된 경로와 run 식별자를 확인한다.
- 소스 환경 `/workspace/deep`에 접근할 수 있으면 해당 실행의 registry가 가리키는 폴더에서 상태·진단·검증·평가·요약을 읽는다. 기록 시각을 현재 시각과 대조하고, PID·시작 토큰·명령행이 모두 일치하며 zombie가 아닌 실제 프로세스인지 확인한다. 오래된 `running` 기록만으로 실행 중이라고 보고하지 않는다.
- 원래120회·120→240회·240→360회 자기대전과 첫 리버 탐색 비교는 완료됐다. 이력은 각각 기존 문서와 `reports/bounded-continuation-status.json`, `reports/poker-river-strength-status.json`에 보존한다. 새 신경망 검증·새 입력 학습은 별도 run으로 추적하며 이전360회 controller를 재개하지 않는다.
- 실제로 살아 있는 학습·평가기·게시기를 중복 실행하거나 사전 기준을 바꾸지 않는다. 새 실행의 고정 예산·검증 조건을 따르고, 성능 조건 미달·실행 실패·비유한값·출처 불일치 뒤 자동 재실험하지 않는다. 학습 종료와 전체 평가 종료를 구분한다.
- 작은 게임의 수치·gradient·exploitability 검증이나 제한된 리버 비교를6인 실력 향상으로 표현하지 않는다. 고정 replay fitting·탐색 모드 추가를 새 자기대전 회차나 기본 모델 교체로 세지 않는다. 통계 단위는 원래 카드 묶음이며 좌석·seed 반복을 독립 표본으로 부풀리지 않는다.
- 소스 환경에 접근할 수 없으면 GitHub 기록과 이미 확인된 Drive 백업까지만 보고한다. 실행 여부·파일 유실을 추측하거나 실행 파일이 없는 새 환경에서 처음부터 학습하지 않는다. 완료·중단 시 이유와 마지막 안전한 복구본을 알린다.

## 백업

- 사용자는 기존에 연결된 개인 Google Drive의 비공개 백업과 불필요한 포커 백업 정리를 승인했다. 계정·파일 ID·URL·삭제 journal은 비공개 인덱스 `artifacts/offsite-backups.json`과 감사 기록에만 둔다. 인덱스의 `folder_id`를 재사용한다.
- 실제 학습 파일이 있는 동일 실행에서만 작업한다. 완료된 새 단계 또는 기존보다 진전된 최신 복구본 하나를 골라 로컬 크기·SHA-256·내부 파일·실제 복원을 검증한다. 매 중간 checkpoint를 누적하지 않고, 최신 모델과 고정 상한 replay를 보존하는 경량 프로필을 사용한다.
- 파일당100MiB를 넘으면64MiB 이하 조각으로 나눈다. 원본과 조각 재결합의 로컬 SHA-256 일치를 확인하고 순서·크기·해시·Drive ID가 담긴 비공개 복원 manifest를 함께 업로드한다. 모든 조각과 manifest의 원격 크기·승인된 소유자만 가진 권한을 확인한 뒤 하나의 완료 항목으로 기록한다. 부분 업로드는 완료가 아니다. 원격 체크섬을 받지 못했으면 원격 해시 검증을 주장하지 않는다.
- 자기대전 복구본은 `training-checkpoint`, 고정 replay 파생 후보는 `derived-policy-probe`, 연구 산출물은 `research-validation`으로 구분한다. 기존 compact 복구본의 `exact_legacy_resume=false`, `resumable_selfplay=false`는 옛 worker의 정확 재개 미지원 표시다. 새 rich family는 새 worker의 복구·다음 업데이트 동일성 검증이 통과할 때만 해당 family의 재개 지원을 기록하고, legacy 재개와 구분한다. `warm_start_state_verified=true`와 별개로 기록하며 모델·표본 보존 실패로 해석하지 않는다.

## 보존·정리

- `scripts/plan_poker_backup_retention.py` 및 새 family를 구분하는 `scripts/plan_rich_backup_retention.py`와 최신 저장 문서에 따라 평가 자산, 첫 전체60회, 파생·연구에 필요한 전체120회, 최신 완료 checkpoint, 그 이후 최신 복구본 하나, 최근 파생2개, 최신 연구1개 및 모든 복원 의존성을 보존한다. 새 rich 회차를 원래360회와 같은 계열로 비교해 원래 복구본을 삭제하지 않는다. 목표 예산은2GiB이며 필수 복구본이 넘으면 알리고 삭제하지 않는다.
- 새 백업의 모든 검증이 끝난 뒤 실제 소스 환경에서 보존 계획을 만든다. 최신 archive의 로컬 크기·SHA-256과 보존·정리 대상의 원격 크기·소유자 전용 권한을 다시 확인한다. 계획 이후 인덱스가 바뀌지 않았을 때만 명확히 대체된 파일을 정리한다.
- 삭제 성공과 readback의 `NOT_FOUND`를 비공개 journal에 기록하고 해당 인덱스 항목을 `deleted_remote`로 표시한다. 이미 삭제된 항목을 현재 복구본으로 보고하지 않는다. 다른 종류·부분 업로드·알 수 없는 파일·복원 의존성은 임의 삭제하지 않는다.
- 2026-11-13 17:04 KST 검증 완료 복구본은 새 rich family12회113,090,689 bytes다. 원래360회 부모 경량본90,022,512 bytes도 보존한다. 당시 관리량은8개 백업·24개 파일·1,179,681,407 bytes(약1.10GiB), 추가 정리 후보0개였다. 매 점검에 현재 인덱스로 재확인한다. 삭제된240회 전체 archive와 이전240회 경량본은 다시 업로드하지 않는다.
