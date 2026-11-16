# 포커 학습 점검·백업 기준

예약 작업은 최신 실행의 진도·평가·실패·완료와 백업을 확인한다. 학습은 별도의 클라우드 프로그램이 수행한다. 새 변화가 있을 때만 한국어로 짧게 알리고, 이미 보고한 완료 결과를 반복 알리지 않는다. 소스에 접근할 수 있으면 비공개 `artifacts/poker-monitoring/last-notified.json`과 run·상태·fitting 수·평가 수·요약 해시·백업 해시를 비교해 이미 알린 완료 결과를 반복하지 않는다.

## 최신 실행 확인

- 최신 실행은2026-11-16 개선 후보 독립 재검증 `20261116T052535674094Z`다.14:25 KST에 시작했으며 `docs/retraining-candidate-confirmation-20261116.md`, `reports/candidate-confirmation-status.json`, `artifacts/candidate-confirmation/current.json`을 먼저 읽는다. 감독은 `run_poker_candidate_confirmation.py`, 평가기는 `evaluate_rich_selfplay.py`다. PID·시작 토큰·명령행과 시각을 대조한다. 후보 가중치는 고정됐고 새 fitting·자기대전은0이다. 새20,000카드 묶음·36셀·432만 핸드의 독립 평가다. 주 판정은 후보−원본120회의 기존 네 조건이며 부모360회·1024steps 대비 진단과 구분한다.36셀 완료 전에 전체 완료·채택으로 표시하지 않으며 결과를 보고 후보·조건을 바꾸거나 자동 재시도·모델 교체·추가 학습하지 않는다. 이전90셀 통제 진단의 완료를 새 평가 완료로 혼동하지 않는다.

- 이전 작업은2026-11-16 통제 fitting 진단이다.13:18 KST에21개 fitting과90개 평가를 모두 완료했고13:56 KST에 파생 정책 묶음의 비공개 백업·대체된 이전 파생본 정리를 마쳤다. `docs/retraining-controlled-refit-20261116.md`, `reports/controlled-refit-status.json`, `artifacts/controlled-refit/current.json`을 먼저 확인한다. 고정 replay에서 파생 후보를 비교하는 것으로 새 자기대전 회차·기본 모델 교체가 아니다. 감독 `run_controlled_poker_study.py`, fitting `controlled_poker_refit.py`, 평가 `evaluate_rich_selfplay.py`의 PID·시작 토큰·명령행을 확인한다.21개 fitting과90개 평가의 완료를 구분한다. 기존360회·rich96회 완료와 혼동하지 않으며 같은 완료·백업 기록은 반복 알리지 않는다.

- 저장소는 `cec0948/deep`, 브랜치는 `codex/poker-retrain-20261112`다. `docs/retraining-20261112.md`의 최신 실행 안내와 `reports/training-neural-status.json`을 먼저 읽는다. 공유 신경망 검증은 `artifacts/neural-information-gate/current.json`, 이후 새 입력 자기대전은 `artifacts/rich-selfplay/current.json`으로 구분한다. 실제 작성된 경로와 run 식별자를 확인한다.
- 소스 환경 `/workspace/deep`에 접근할 수 있으면 해당 실행의 registry가 가리키는 폴더에서 상태·진단·검증·평가·요약을 읽는다. 기록 시각을 현재 시각과 대조하고, PID·시작 토큰·명령행이 모두 일치하며 zombie가 아닌 실제 프로세스인지 확인한다. 오래된 `running` 기록만으로 실행 중이라고 보고하지 않는다.
- 원래120회·120→240회·240→360회 자기대전과 첫 리버 탐색 비교는 완료됐다. 이력은 각각 기존 문서와 `reports/bounded-continuation-status.json`, `reports/poker-river-strength-status.json`에 보존한다. 새 신경망 검증·새 입력 학습은 별도 run으로 추적하며 이전360회 controller를 재개하지 않는다.
- 실제로 살아 있는 학습·평가기·게시기를 중복 실행하거나 사전 기준을 바꾸지 않는다. 새 실행의 고정 예산·검증 조건을 따르고, 성능 조건 미달·실행 실패·비유한값·출처 불일치 뒤 자동 재실험하지 않는다. 학습 종료와 전체 평가 종료를 구분한다.
- 작은 게임의 수치·gradient·exploitability 검증이나 제한된 리버 비교를6인 실력 향상으로 표현하지 않는다. 고정 replay fitting·탐색 모드 추가를 새 자기대전 회차나 기본 모델 교체로 세지 않는다. 통계 단위는 원래 카드 묶음이며 좌석·seed 반복을 독립 표본으로 부풀리지 않는다.
- 소스 환경에 접근할 수 없으면 GitHub 기록과 이미 확인된 Drive 백업까지만 보고한다. 실행 여부·파일 유실을 추측하거나 실행 파일이 없는 새 환경에서 처음부터 학습하지 않는다. 완료·중단 시 이유와 마지막 안전한 복구본을 알린다.

## 백업

- 이번 독립 재검증은 기존21정책 백업의256steps 후보를 byte 동일하게 재사용한다. `backup-reference.json`과 비공개 인덱스의 동일 archive SHA를 확인하며, 가중치가 바뀌지 않았으므로 새 모델 archive를 중복 업로드하지 않는다. 완료된 후속 평가의 요약·통계·실패 기록은 새 공개 reports 경로로 별도 게시한다. 기존 archive에 이번 후속 평가가 들어 있다고 주장하지 않는다. 마지막 안전한 가중치 백업은24,408,351 bytes의 검증 완료 통제 파생 묶음이며, 원본360회·rich96회 등 기존 모든 필수 복구본을 유지한다.

- 최신 통제 진단은 `backup-ready.json`의 완전 검증된 `controlled-refit-policy-bundle-v1` 하나만 `kind=derived-policy-probe`, `resumable_selfplay=false`, `new_selfplay_iterations=0`으로 기록한다. `export_controlled_poker_probe.verify_probe`로 실제 추론 runtime 복원을 확인한다. `prepare_upload`의 분할·재결합 검증과 모든 원격 크기·권한 확인 뒤에만 완료로 기록한다. replay를 중복 업로드하지 않으며, 기존360회 및rich96회 복구본 SHA를 `restore_dependencies`에 넣어 둘 다 보존한다.

- 사용자는 기존에 연결된 개인 Google Drive의 비공개 백업과 불필요한 포커 백업 정리를 승인했다. 계정·파일 ID·URL·삭제 journal은 비공개 인덱스 `artifacts/offsite-backups.json`과 감사 기록에만 둔다. 인덱스의 `folder_id`를 재사용한다.
- 실제 학습 파일이 있는 동일 실행에서만 작업한다. 완료된 새 단계 또는 기존보다 진전된 최신 복구본 하나를 골라 로컬 크기·SHA-256·내부 파일·실제 복원을 검증한다. 매 중간 checkpoint를 누적하지 않고, 최신 모델과 고정 상한 replay를 보존하는 경량 프로필을 사용한다.
- 파일당100MiB를 넘으면64MiB 이하 조각으로 나눈다. 원본과 조각 재결합의 로컬 SHA-256 일치를 확인하고 순서·크기·해시·Drive ID가 담긴 비공개 복원 manifest를 함께 업로드한다. 모든 조각과 manifest의 원격 크기·승인된 소유자만 가진 권한을 확인한 뒤 하나의 완료 항목으로 기록한다. 부분 업로드는 완료가 아니다. 원격 체크섬을 받지 못했으면 원격 해시 검증을 주장하지 않는다.
- 자기대전 복구본은 `training-checkpoint`, 고정 replay 파생 후보는 `derived-policy-probe`, 연구 산출물은 `research-validation`으로 구분한다. 기존 compact 복구본의 `exact_legacy_resume=false`, `resumable_selfplay=false`는 옛 worker의 정확 재개 미지원 표시다. 새 rich family는 새 worker의 복구·다음 업데이트 동일성 검증이 통과할 때만 해당 family의 재개 지원을 기록하고, legacy 재개와 구분한다. `warm_start_state_verified=true`와 별개로 기록하며 모델·표본 보존 실패로 해석하지 않는다.

## 보존·정리

- 통제 진단 파생 묶음이 인덱스에 있으면 `scripts/plan_controlled_backup_retention.py`를 사용한다. 최신 파생2개는 이전 후보와 새 묶음을 합산한다. 알려진 `checkpoint_mode=controlled-refit-policy-bundle-v1`만 새 프로필로 처리하며, 완료 전 후보·부분 업로드·불명확한 파일은 정리하지 않는다. 기존60회·필수120회·부모360회·rich96회·최신 연구·평가 자산·복원 의존성과2GiB 기준은 유지한다.

- `scripts/plan_poker_backup_retention.py` 및 새 family를 구분하는 `scripts/plan_rich_backup_retention.py`와 최신 저장 문서에 따라 평가 자산, 첫 전체60회, 파생·연구에 필요한 전체120회, 최신 완료 checkpoint, 그 이후 최신 복구본 하나, 최근 파생2개, 최신 연구1개 및 모든 복원 의존성을 보존한다. 새 rich 회차를 원래360회와 같은 계열로 비교해 원래 복구본을 삭제하지 않는다. 목표 예산은2GiB이며 필수 복구본이 넘으면 알리고 삭제하지 않는다.
- 새 백업의 모든 검증이 끝난 뒤 실제 소스 환경에서 보존 계획을 만든다. 최신 archive의 로컬 크기·SHA-256과 보존·정리 대상의 원격 크기·소유자 전용 권한을 다시 확인한다. 계획 이후 인덱스가 바뀌지 않았을 때만 명확히 대체된 파일을 정리한다.
- 삭제 성공과 readback의 `NOT_FOUND`를 비공개 journal에 기록하고 해당 인덱스 항목을 `deleted_remote`로 표시한다. 이미 삭제된 항목을 현재 복구본으로 보고하지 않는다. 다른 종류·부분 업로드·알 수 없는 파일·복원 의존성은 임의 삭제하지 않는다.
- 마지막 확인 기록(2026-11-16 13:56 KST): 최신 통제 파생 묶음24,408,351 bytes의 실제 runtime 복원·로컬 해시·원격 크기·소유자 전용 권한 확인과 비공개 인덱스 기록을 마쳤다. 대체된 이전 파생 백업1개14,533,619 bytes만 삭제·NOT_FOUND readback·deleted_remote 처리했다. 원본360회 경량본90,022,512 bytes와rich96회122,723,204 bytes, 필수60/120회·최신 파생2개·최신 연구·평가 자산·모든 의존성을 보존한다. 관리량은8개 백업·24개 파일·1,199,188,654 bytes(약1.12GiB), 추가 정리 후보0개다. 원격 체크섬은 제공되지 않았다. 매 점검에 현재 인덱스로 재확인한다. 삭제된240회 전체 archive와 이전240회 경량본은 다시 업로드하지 않는다.
