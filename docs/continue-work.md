# 여기서 작업 이어가기

목표는 자체 프로그램에서 실행하는 6인·100BB 포커 AI다. **[V3의 새 버퍼 3-seed×120회 학습과 288,000핸드 평가를 완료했다](fullhouse-v3-results.md).** 규칙·특징·평균 전략 fitting을 수정해 기존 평균 전략보다 개선됐지만 60→120회의 일관된 이득은 미확인이다. 같은 학습의 advantage 정책이 더 강했으며 seed 42가 다음 독립 평가 후보다. 기본 모델은 유지한다. V2 600회 실행은 [원인 조사](fullhouse-regression-diagnosis.md)와 비교용으로 보존한다. 프로급 강도는 미검증이다.

## 현재 상태와 재개

```bash
cd /workspace/deep
python scripts/train_fullhouse_v3.py status
```

`artifacts/fullhouse-v3/current.json` → 실행 폴더의 `status.json`이 실제 상태다. 문서에 적힌 진행 숫자는 스냅샷이다. `running`이며 `alive: true`이면 그대로 둔다. `complete`이면 `results.json`을 분석한다. 환경이 종료되어 프로세스가 사라진 중단 실행만 재개한다.

```bash
python scripts/train_fullhouse_v3.py resume
```

중복 실행을 거절하며, 실행 잠금과 PID 시작 시각을 함께 확인한다. 구간 완료 시점은 12 → 60 → 120회다. 상태의 `committed`는 supervisor가 확인한 **구간 완료** 반복 수, `latest`는 최근 진단 로그다. 이 둘 사이에도 매 12회 `latest.pt`가 저장되며 재개는 그 전체 checkpoint부터 진행한다.

각 seed는 단일 CPU worker/thread이며 세 seed를 병렬 처리한다. 회당 regret root 256개, regret 512×128, strategy 512×512 fitting이다. 시간 가중치 horizon은 600, 이번 실행 종료점은 120이다. 학습 후 두 상대 풀에 12개 후보를 각 2,000 deals×6좌석으로 자동 평가한다. 총 예정 288,000핸드이며 후보별 결과를 모두 보존한다.

학습 소스는 실행 폴더의 `source/`에 고정되어 있다. supervisor·평가기·runtime·비교 모델·엔진의 SHA-256도 저장한다. 실행 중 이 파일들을 바꾸면 다음 구간의 검사를 거절한다. 과거 V2 checkpoint로 V3 학습을 재개하지 않는다.

완료한 평가를 원시 로그부터 검산하고 Git용 결과를 만들려면:

```bash
python scripts/summarize_fullhouse_v3.py
```

원시 평가·적합도·설정·해시와 요약은 `reports/fullhouse-v3/RUN_ID/`에 저장한다. 이전 중간 규모 실행은 `python scripts/train_fullhouse_medium.py status`로 확인할 수 있지만, **기존 규칙 오류가 있는 V2 학습을 연장하지 않는다.**

## Git과 checkpoint 백업

Git에는 고정 upstream 준비·패치·규칙 oracle·JSON runtime·학습 및 평가 실행기·검사·문서·압축 원시 대전 로그를 보존한다. `.cache/`, `artifacts/`, venv, 모델·reservoir는 Git에 포함하지 않는다. upstream 라이선스는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다.

최신 V3 백업 도구:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python scripts/export_fullhouse_v3.py
```

생성 파일은 `artifacts/exports/fullhouse-v3-TIMESTAMP.tar.gz`, 이름·용량·SHA-256·seed별 저장 반복 수는 `reports/fullhouse-v3-export.json`과 해당 archive 옆 `.json`에 기록된다. **archive는 현재 환경의 로컬 파일이다. 환경이 사라져도 보존하려면 별도로 내려받아야 한다. GitHub 푸시는 큰 checkpoint 백업을 대신하지 않는다.**

백업에는 다음이 들어간다.

- 세 seed의 전체 모델·reservoir·CPU RNG·설정이 담긴 `latest.pt`.
- 저장 시점 이하 모델, 고정 학습 소스·설정·진단·검증 gate.
- 이번 실행에서 별도로 보존한 세 seed의 60회 전체 버퍼 checkpoint와 해시.
- 자동 평가에 필요한 기존 120·600회 모델, 공개 모델, equity 데이터, NoRegrets 1M blueprint, 원래 bridge binary와 라이선스 고지.
- PID를 제거한 복원용 상태와 파일별 SHA-256 manifest.
- 완료 실행이면 대전 로그·평가 결과도 포함한다. 진행 중 snapshot이면 평가는 재실행하도록 초기화한다.

실행 중에도 한 번 연 atomic checkpoint inode를 복사하므로 seed별 저장 시점은 일관된다. 서로 다른 seed가 같은 반복까지 도달하지 않았어도 각각 정확히 재개할 수 있다. 별도 NoRegrets의 큰 학습 checkpoint는 포함하지 않는다. 기존 V2 600회 백업은 [이전 백업 정보](../reports/checkpoint-export.json)에 있다.

## 새 환경에 복원

현재 bundle은 **Debian 13, glibc 2.41, Linux x86_64, Python 3.12, `/workspace/deep` 경로**를 기준으로 한다. CPU·라이브러리가 바뀌면 bit 단위 재현을 보장하지 않는다. 먼저 저장소와 고정 Fullhouse 의존성을 준비한다.

```bash
git clone --branch codex/poker-ai-handoff-20261106 https://github.com/cec0948/deep.git /workspace/deep
cd /workspace/deep
bash scripts/setup_fullhouse.sh --training
/workspace/.venvs/fullhouse/bin/python -m pip install pytest==8.4.2
```

별도로 보관한 archive를 옮기고 metadata의 SHA-256과 비교한 후 프로젝트 루트에 푼다.

```bash
resume_archive=/path/to/fullhouse-v3-TIMESTAMP.tar.gz
sha256sum "$resume_archive"
# archive 옆 .json 또는 reports/fullhouse-v3-export.json의 sha256과 비교
tar -xzf "$resume_archive" -C /workspace/deep
python scripts/train_fullhouse_v3.py status
```

완료 bundle은 분석·비교용으로 사용할 수 있다. 미완료 bundle만 `python scripts/train_fullhouse_v3.py resume`으로 재개한다. 정확 재개에는 **`latest.pt`**가 필요하다. 버퍼 없는 `iter_XXXXX.pt`나 추론용 `.npz`로는 RNG와 reservoir를 복구할 수 없다. hash 검사 실패는 버전·복원 파일을 확인해 해결하며 검사를 무조건 해제하지 않는다.

다른 플랫폼에서 Rust binary를 다시 빌드할 때는 [JSON API 안내](json-api.md)의 고정 NoRegrets commit과 Rust 1.90.0을 사용한다. oracle·전체 회귀 검사를 새 환경에서 실행하려면 NoRegrets 원본과 별도 smoke 모델 등 기존 검사 자산도 준비해야 한다. V3 훈련 재개용 bundle이 프로젝트의 모든 과거 실험 자산을 포함하는 것은 아니다.

## 다음 담당자

1. 상태를 먼저 확인하고 기존 실행이 살아 있으면 유지한다.
2. V3 구현 문서와 완료 결과를 읽는다. 승률뿐 아니라 세 학습 seed의 편차, 평균 전략 적합도, 60→120회 paired 차이를 확인한다.
3. 최종 결과의 AA regret 반례를 먼저 읽는다. seed 42는 경험적 폴드 표적 -73.85를 +0.49로 예측했고, Torch/npz 출력은 일치했다. 새 고정 버퍼에서 regret fitting 예산·상태 균형을 비교하고, 평균 전략 collector의 분산과 구분한다. 좋은 seed만 고르거나 같은 설정을 바로 유료 장기 학습으로 확대하지 않는다.
4. 51개 요약 특징의 imperfect recall, 공유 신경망 근사와 6인 균형 수렴은 별도 이론 쟁점으로 남긴다.

새 대화에서 사용할 요청:

> `/workspace/deep/docs/continue-work.md`와 `docs/fullhouse-v3-training.md`를 읽고 현재 실행 상태를 확인하세요. 실행 중이면 유지하고, 완료했다면 세 seed의 학습·평가 원시 결과를 검산해 다음 개선 우선순위를 정하세요. 기존 실험과 체크포인트를 보존하고 독립적으로 가능한 구현은 계속하세요.
