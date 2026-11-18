# 포커 학습 점검·백업 기준

예약 작업은 최신 실행의 진도·평가·실패·완료와 백업을 확인한다. 학습은 별도의 클라우드 프로그램이 수행한다. 새 변화가 있을 때만 한국어로 짧게 알리고, 이미 보고한 완료 결과를 반복 알리지 않는다. 소스에 접근할 수 있으면 비공개 `artifacts/poker-monitoring/last-notified.json`과 run·상태·fitting 수·평가 수·요약 해시·백업 해시를 비교해 이미 알린 완료 결과를 반복하지 않는다.

## 최신 실행 확인

- 프리플롭 독립 재검증은 2026-11-18 12:32 KST에 27셀·324만 핸드 평가를 마쳤다. 합격 뒤 `artifacts/preflop-followup/480-v1/online`에서 480→492회 두 방식의 비교를 시작했다. `training-status.json`과 `evaluation-status.json`의 실제 PID·시작 토큰·기록 시각을 대조한다. [후속 실험 기록](poker-preflop-followup-20261118.md)

- 프리플롭 본 실험 `480-v1`은 2026-11-18 11:25 KST에 세 시드의 피팅과 36셀·432만 핸드 평가를 마쳤다. 두 기준 모델에 대한 사전 조건을 모두 통과했고, 검증한 파생 묶음 3,280,171바이트를 비공개 인덱스에 기록했다. `scripts/run_poker_preflop_pilot.py status`와 `result`로 확인한다. [프리플롭 평가와 복원 기록](poker-preflop-pilot-20261118.md)

- 2026-11-17 현재 자동 알림은 사용자 요청으로 비활성화돼 있다. 사용자가 점검이나 새 실행을 요청해도 알림 예약을 임의로 다시 켜지 않는다.

- 최신 완료 확인(2026-11-17 17:34 KST): 아래480회 실행은 학습과36셀·432만 핸드 평가를 모두 마쳤다.480−420회는 공개−13.67·혼합+92.28·NoRegrets+85.79bb/100으로9비교 동시95%구간에서 각각 악화·개선·개선이다. 중간 품질 실패가 아니라 고정 예산 완료이며 자동 재실험·연장·모델 교체는 없다. 새480회 복구본107,049,296 bytes와 두 조각·복원manifest를 완전 검증했고,420회 비교 가중치를 새복구본에 보존한 뒤 대체된 이전420회 archive 하나만 삭제·NOT_FOUND·deleted_remote 처리했다. 원시 집계·통계 사후 검산은 `final-audit.json`, 공개 백업 확인은 `backup-verification.json`을 따른다. 최종9개·27파일·1,306,240,566 bytes(약1.22GiB), 추가 정리 후보0개다. 원격 체크섬은 제공되지 않았다. 같은 완료 결과를 반복 알리지 않는다.

- 가장 최근 승인 작업은420회 결과를 본 사용자의 추가 학습 요청에 따른 `20261117T063900Z`의420→480회 자기대전이다. `docs/retraining-average-extension-20261117.md`, `reports/average-extension-status.json`, `artifacts/average-extension/current.json`을 먼저 확인한다. 감독 `run_poker_average_extension.py`, 학습기 `train_poker_average_extension.py`, 평가기 `evaluate_average_baseline.py`의 실제PID·시작토큰·명령행과 최신 시각을 대조한다. 가중치·replay·첫 행동·모든 학습 및 평균 수집 난수 상태를 연결하며 새회차는3seed각60회다. 주 표본180만행·평균표본30만행 상한과 기존 학습 설정을 유지한다. 유한한 평균TV초과나 중간 성능만으로 학습을 중단하지 않는다. 실행 오류·비유한값·출처 불일치·주 fitting 실패는중단하며 자동 재시도하지 않는다.480회 학습 완료와36셀·432만 핸드 평가 완료를 구분하고, 새20,000카드 묶음 단위로480−420진도와 원본120회 대비 기존 네 채택 조건을 구분한다. 새480회 archive를 검증하기 전 기존420회 복구본을 지우지 않는다. 이번 승인에도 자동 알림은 꺼진 상태를 유지한다.

- 최신 완료 확인(2026-11-17): 아래 `20261117T032100Z`는14:08 KST에420회 학습과36셀·432만 핸드 평가를 모두 끝냈다. 주 모델의360회 대비 공개+1.40·혼합−1.08·NoRegrets−18.68bb/100은 동시95%구간이0을 포함해 개선을 확인하지 못했다. 평균 모델은 주 모델보다 공개 상대에서−22.43bb/100(동시95%−36.05~−8.80) 낮았다. 원본120회 대비 주 모델의 공개 채택 조건2개는 미달했다. 중간 품질 중단이 아니라 고정 예산 완료이며 자동 연장·모델 교체는 없다. 감독zombie·worker종료를 확인했고 최종 원시 집계와 통계는 `final-audit.json`으로 재검산했다. 새420회 복구본99,711,994 bytes는 복원·원격크기·소유자전용권한 확인 후 인덱스에 완료로 기록했다. 원격 체크섬은 없고 학습archive에 후속평가가 포함되지는 않는다. 이 완료·백업 사실은 반복 알리지 않는다.

- 최신 승인 실행은 `artifacts/average-continuation/current.json`의 `20261117T032100Z`다. `docs/retraining-average-continuation-20261117.md`와 `reports/average-continuation-status.json`을 우선 읽는다. 감독 `run_poker_average_continuation.py`, 학습기 `train_poker_average_continuation.py`, 평가기 `evaluate_average_baseline.py`의 실제PID·시작토큰·명령행을 확인한다. 기존 저장366/366/372에서420회까지 새54/54/48회이며 평균 구간은361~420이다. 유한한 보조 평균TV는이전0.15기준대비진단으로남기며그초과만으로주학습을중단하지않는다. 주모델예산과표본은그대로다. 비유한값·출처·실행·주fitting오류에는중단하고자동재시도하지않는다. 사전주후보는최신420회이며 평균은별도진단이다. 학습과36셀·432만핸드평가완료를구분하고모델교체·420회이후연장은하지않는다. 새backup-ready.json은 `snapshot_poker_average_continuation.py`로검증한420회하나만등록하고평균근사기준통과와복원성공을구분한다. 기존개인Drivefolder·64MiB분할·원격크기/소유자전용권한·2GiB보존정책을유지한다. 이전실패보고와registry는덮어쓰지않는다.

- 이전 실행 `20261116T084130Z`는2026-11-16 17:47 KST에첫372회 평균 근사 품질 미달로중단됐다. 세seed TV0.159405/0.169851/0.148366중2개가고정기준0.15를넘었다.420회학습·전체36셀평가는완료되지않았고평가0셀이다. 감독zombie·세worker종료와동결입력유지를확인했다. 마지막로컬저장은366/366/372회이며이를공통372회완료로표시하지않는다. `failure-audit.json`을확인하고자동재실행·조건완화·새후보변경·모델교체를하지않는다. 새로운backup-ready.json은없고기존360회·rich96회원격크기/권한을재확인했다. 같은실패알림은반복하지않는다.

- 이전 승인 작업은2026-11-16의 표본·학습 예산 유지 및 평균 전략 기준 실험이다. `docs/retraining-average-baseline-20261116.md`, `reports/average-baseline-status.json`, `artifacts/average-baseline/current.json`을 우선 확인한다. 감독은 `run_poker_average_baseline.py`, 학습기는 `train_poker_average_baseline.py`, 평가기는 `evaluate_average_baseline.py`다. 기존360회 가중치·replay·첫 행동 통계·난수 상태를 이어받아360→420회 새60회 자기대전을 수행한다. 이전256steps 후보 재평가와 구분한다. 표본100,000행/좌석·탐색256·fitting1,024×512를 유지하며, 추가 평균 전략은361~420구간만의 신경망 근사다.420회 전체 역사 평균이나 논문 전체 재현으로 표시하지 않는다. 실제PID·시작 토큰·명령행·기록 시각과 새폴더 `status.json`, `campaign/seed-*/status.json`·`current.json`·`diagnostics.jsonl`, `preflight.json`, `stage-ready.json`, `evaluation-plan.json`, `evaluation-summary.json`, `summary.json`, `failure.json`, `backup-ready.json`을 대조한다. 학습 완료와 새20,000카드 묶음의36셀·432만 핸드 평가 완료를 구분한다. 사전 후보는평균361~420이며 원본120회 대비 기존 네 조건과 부모360회·최신420회 대비 진단을 구분한다.372회는 품질 확인 단계이고420회 이후 연장·실패 자동 재시도·후보 변경·기준 변경·모델 교체를 하지 않는다. 현재 입력51차원의 정보 손실은 남아 있으므로 작은 게임 평균화 검증을6인 실력 향상으로 표현하지 않는다.

- 이전 완료 실행은2026-11-16 개선 후보 독립 재검증 `20261116T052535674094Z`다.15:44:50 KST에36셀·432만 핸드 평가를 완료했고, 공개 상대 변화−0.30bb/100(명목95%−9.60~+8.99), 개선seed1/3개로 기존 네 조건 중 공개 상대 조건2개에 미달했다. `docs/retraining-candidate-confirmation-20261116.md`, `reports/candidate-confirmation-status.json`, `artifacts/candidate-confirmation/current.json`을 먼저 읽는다. 감독은 `run_poker_candidate_confirmation.py`, 평가기는 `evaluate_rich_selfplay.py`다. 기록된 PID·시작 토큰·명령행과 현재 시각·실제 프로세스를 대조하며 완료된 감독의 zombie 상태를 실행 중으로 해석하지 않는다. 가중치 변경·새 fitting·자기대전은0이다. 통계 단위는20,000개 새 카드 묶음이다. 주 판정은 후보−원본120회의 기존 네 조건이며 부모360회·1024steps 대비 진단과 구분한다. 미달 뒤 자동 재시도·후보 변경·모델 교체·추가 학습하지 않으며, 같은 완료·백업 기록은 반복 알리지 않는다.

- 이전 작업은2026-11-16 통제 fitting 진단이다.13:18 KST에21개 fitting과90개 평가를 모두 완료했고13:56 KST에 파생 정책 묶음의 비공개 백업·대체된 이전 파생본 정리를 마쳤다. `docs/retraining-controlled-refit-20261116.md`, `reports/controlled-refit-status.json`, `artifacts/controlled-refit/current.json`을 먼저 확인한다. 고정 replay에서 파생 후보를 비교하는 것으로 새 자기대전 회차·기본 모델 교체가 아니다. 감독 `run_controlled_poker_study.py`, fitting `controlled_poker_refit.py`, 평가 `evaluate_rich_selfplay.py`의 PID·시작 토큰·명령행을 확인한다.21개 fitting과90개 평가의 완료를 구분한다. 기존360회·rich96회 완료와 혼동하지 않으며 같은 완료·백업 기록은 반복 알리지 않는다.

- 저장소는 `cec0948/deep`, 브랜치는 `codex/poker-retrain-20261112`다. `docs/retraining-20261112.md`의 최신 실행 안내와 `reports/training-neural-status.json`을 먼저 읽는다. 공유 신경망 검증은 `artifacts/neural-information-gate/current.json`, 이후 새 입력 자기대전은 `artifacts/rich-selfplay/current.json`으로 구분한다. 실제 작성된 경로와 run 식별자를 확인한다.
- 소스 환경 `/workspace/deep`에 접근할 수 있으면 해당 실행의 registry가 가리키는 폴더에서 상태·진단·검증·평가·요약을 읽는다. 기록 시각을 현재 시각과 대조하고, PID·시작 토큰·명령행이 모두 일치하며 zombie가 아닌 실제 프로세스인지 확인한다. 오래된 `running` 기록만으로 실행 중이라고 보고하지 않는다.
- 원래120회·120→240회·240→360회 자기대전과 첫 리버 탐색 비교는 완료됐다. 이력은 각각 기존 문서와 `reports/bounded-continuation-status.json`, `reports/poker-river-strength-status.json`에 보존한다. 새 신경망 검증·새 입력 학습은 별도 run으로 추적하며 이전360회 controller를 재개하지 않는다.
- 실제로 살아 있는 학습·평가기·게시기를 중복 실행하거나 사전 기준을 바꾸지 않는다. 새 실행의 고정 예산·검증 조건을 따르고, 성능 조건 미달·실행 실패·비유한값·출처 불일치 뒤 자동 재실험하지 않는다. 학습 종료와 전체 평가 종료를 구분한다.
- 작은 게임의 수치·gradient·exploitability 검증이나 제한된 리버 비교를6인 실력 향상으로 표현하지 않는다. 고정 replay fitting·탐색 모드 추가를 새 자기대전 회차나 기본 모델 교체로 세지 않는다. 통계 단위는 원래 카드 묶음이며 좌석·seed 반복을 독립 표본으로 부풀리지 않는다.
- 소스 환경에 접근할 수 없으면 GitHub 기록과 이미 확인된 Drive 백업까지만 보고한다. 실행 여부·파일 유실을 추측하거나 실행 파일이 없는 새 환경에서 처음부터 학습하지 않는다. 완료·중단 시 이유와 마지막 안전한 복구본을 알린다.

## 백업

- 새 평균 기준 실험은 `snapshot_poker_average_baseline.py`가 만든 최신420회 `backup-ready.json` 하나를 `kind=training-checkpoint`, `checkpoint_mode=compact-window-average-recovery-v1`로 검증·백업한다. 실제 archive 추출·파일 SHA/크기·current/average 모델 배열·평균 추론·reservoir 다음 난수 상태 검증이 모두 통과해야 한다. `warm_start_state_verified=true`, `average_policy_available=true`, `averaging_window=[361,420]`, `resumable_selfplay=false`, `exact_legacy_resume=false`, `historical_mixture_available=false`를 구분한다. 원격 크기·소유자 전용 권한 확인을 끝내기 전 완료로 기록하지 않는다.100MiB초과이면64MiB조각·재결합 SHA·복원 manifest 규칙을 그대로 적용한다. 학습 종료 시점 archive에는 이후 최종36셀 평가가 포함되지 않으므로 `archive_includes_final_evaluation=false`를 유지하고 최종 평가를 별도 게시한다. 동일 archive SHA와 이미 인덱스에 있는 source SHA를 재업로드하지 않는다.

- 이번 독립 재검증은 기존21정책 백업의256steps 후보를 byte 동일하게 재사용한다. `backup-reference.json`과 비공개 인덱스의 동일 archive SHA를 확인하며, 가중치가 바뀌지 않았으므로 새 모델 archive를 중복 업로드하지 않는다. 완료된 후속 평가의 요약·통계·실패 기록은 새 공개 reports 경로로 별도 게시한다. 기존 archive에 이번 후속 평가가 들어 있다고 주장하지 않는다. 마지막 안전한 가중치 백업은24,408,351 bytes의 검증 완료 통제 파생 묶음이며, 원본360회·rich96회 등 기존 모든 필수 복구본을 유지한다.

- 최신 통제 진단은 `backup-ready.json`의 완전 검증된 `controlled-refit-policy-bundle-v1` 하나만 `kind=derived-policy-probe`, `resumable_selfplay=false`, `new_selfplay_iterations=0`으로 기록한다. `export_controlled_poker_probe.verify_probe`로 실제 추론 runtime 복원을 확인한다. `prepare_upload`의 분할·재결합 검증과 모든 원격 크기·권한 확인 뒤에만 완료로 기록한다. replay를 중복 업로드하지 않으며, 기존360회 및rich96회 복구본 SHA를 `restore_dependencies`에 넣어 둘 다 보존한다.

- 사용자는 기존에 연결된 개인 Google Drive의 비공개 백업과 불필요한 포커 백업 정리를 승인했다. 계정·파일 ID·URL·삭제 journal은 비공개 인덱스 `artifacts/offsite-backups.json`과 감사 기록에만 둔다. 인덱스의 `folder_id`를 재사용한다.
- 실제 학습 파일이 있는 동일 실행에서만 작업한다. 완료된 새 단계 또는 기존보다 진전된 최신 복구본 하나를 골라 로컬 크기·SHA-256·내부 파일·실제 복원을 검증한다. 매 중간 checkpoint를 누적하지 않고, 최신 모델과 고정 상한 replay를 보존하는 경량 프로필을 사용한다.
- 파일당100MiB를 넘으면64MiB 이하 조각으로 나눈다. 원본과 조각 재결합의 로컬 SHA-256 일치를 확인하고 순서·크기·해시·Drive ID가 담긴 비공개 복원 manifest를 함께 업로드한다. 모든 조각과 manifest의 원격 크기·승인된 소유자만 가진 권한을 확인한 뒤 하나의 완료 항목으로 기록한다. 부분 업로드는 완료가 아니다. 원격 체크섬을 받지 못했으면 원격 해시 검증을 주장하지 않는다.
- 자기대전 복구본은 `training-checkpoint`, 고정 replay 파생 후보는 `derived-policy-probe`, 연구 산출물은 `research-validation`으로 구분한다. 기존 compact 복구본의 `exact_legacy_resume=false`, `resumable_selfplay=false`는 옛 worker의 정확 재개 미지원 표시다. 새 rich family는 새 worker의 복구·다음 업데이트 동일성 검증이 통과할 때만 해당 family의 재개 지원을 기록하고, legacy 재개와 구분한다. `warm_start_state_verified=true`와 별개로 기록하며 모델·표본 보존 실패로 해석하지 않는다.

## 보존·정리

- 420→480회 연장에서는 `scripts/plan_average_extension_backup_retention.py`로 최신480회 복구본과 그 안의420회 비교 가중치를 검증한 뒤 명시적으로 대체된420회 archive만 정리 후보로 다룬다. 다른 활성 백업의 복원 의존성이 있으면420회도 보존한다. 이 계획은 삭제를 직접 수행하지 않으며 기존 원격 재검증·인덱스 불변·privatejournal·NOT_FOUND·deleted_remote 절차를 그대로 따른다. 원본360회·rich96회·첫60/필수120회·최근파생2개·최신연구·평가자산·모든 의존성과2GiB목표는 유지한다.

- 새 평균 기준 복구본을 보관할 때는 `scripts/plan_average_backup_retention.py`를 사용한다. 기존 통제 진단 보존 정책을 포함하며 최신 평균 복구본 하나와 원본360회·rich96회·첫60회·필수120회·최근 파생2개·최신 연구·평가 자산 및 모든 복원 의존성을 유지한다.2GiB목표를 넘는 필수 복구본은 알리고 임의 삭제하지 않는다. 매 회차 네트워크·표본을 누적 백업하지 않는다.

- 통제 진단 파생 묶음이 인덱스에 있으면 `scripts/plan_controlled_backup_retention.py`를 사용한다. 최신 파생2개는 이전 후보와 새 묶음을 합산한다. 알려진 `checkpoint_mode=controlled-refit-policy-bundle-v1`만 새 프로필로 처리하며, 완료 전 후보·부분 업로드·불명확한 파일은 정리하지 않는다. 기존60회·필수120회·부모360회·rich96회·최신 연구·평가 자산·복원 의존성과2GiB 기준은 유지한다.

- `scripts/plan_poker_backup_retention.py` 및 새 family를 구분하는 `scripts/plan_rich_backup_retention.py`와 최신 저장 문서에 따라 평가 자산, 첫 전체60회, 파생·연구에 필요한 전체120회, 최신 완료 checkpoint, 그 이후 최신 복구본 하나, 최근 파생2개, 최신 연구1개 및 모든 복원 의존성을 보존한다. 새 rich 회차를 원래360회와 같은 계열로 비교해 원래 복구본을 삭제하지 않는다. 목표 예산은2GiB이며 필수 복구본이 넘으면 알리고 삭제하지 않는다.
- 새 백업의 모든 검증이 끝난 뒤 실제 소스 환경에서 보존 계획을 만든다. 최신 archive의 로컬 크기·SHA-256과 보존·정리 대상의 원격 크기·소유자 전용 권한을 다시 확인한다. 계획 이후 인덱스가 바뀌지 않았을 때만 명확히 대체된 파일을 정리한다.
- 삭제 성공과 readback의 `NOT_FOUND`를 비공개 journal에 기록하고 해당 인덱스 항목을 `deleted_remote`로 표시한다. 이미 삭제된 항목을 현재 복구본으로 보고하지 않는다. 다른 종류·부분 업로드·알 수 없는 파일·복원 의존성은 임의 삭제하지 않는다.
- 마지막 보존 확인(2026-11-17): 최신480회 복구본107,049,296 bytes의 복원·재결합 로컬SHA·두 조각과복원manifest의 원격크기·소유자전용권한 검증을 마쳤다. 대체된420회 archive99,711,994 bytes 하나만 삭제하고NOT_FOUND·deleted_remote를 기록했다.420회 비교 가중치는480회 archive에 포함된다. 원본360회90,022,512 bytes·rich96회122,723,204 bytes·첫60/필수120회·최근파생2개·최신연구·평가자산·모든 복원의존성을 보존한다. 관리량은9개 백업·27파일·1,306,240,566 bytes(약1.22GiB), 추가 정리 후보0개다. 원격 체크섬은 제공되지 않았다. 삭제된240회 archive와420회 이전복구본은 중복 재업로드하지 않는다.
