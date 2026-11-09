# 여기서 작업 이어가기

목표는 자체 프로그램에서 실행하는 6인·100BB 포커 AI다. 600회 확대 학습과 후속 30,000핸드 평가를 완료했으며, 프로급 강도는 아직 검증하지 않았다. 최신 분석은 [학습·평가 결과](fullhouse-medium-results.md)를 먼저 읽는다. 구현·연구 판정·작은 실험 요약은 Git에 저장한다. **현재 학습 가중치와 프로세스는 Git으로 전송되지 않으므로, 다른 환경에서 같은 학습을 재개하려면 별도 checkpoint 백업이 필요하다.**

## 현재 환경에서 이어가기

```bash
cd /workspace/deep
python scripts/train_fullhouse_medium.py status
```

먼저 `status`가 `complete`이면 학습·평가가 끝난 상태이므로 결과 분석부터 한다. 실패/중단 상태에서만 재개한다. 진행 중 상태에서 `supervisor_alive`와 `child_alive`가 true이면 기존 학습이 실행 중이다. 그대로 둔다. 실행 환경이 유지되는 동안 supervisor가 600회 학습과 후속 평가를 계속한다. 환경이 꺼진 뒤 다시 실행되어 프로세스가 없는 경우에는:

```bash
python scripts/train_fullhouse_medium.py resume
```

기존 실행이 살아 있으면 중복 재개를 거절한다. 현재 실행의 `artifacts/fullhouse-medium/current.json` → 실행 폴더의 `status.json`이 실제 상태다. 고정 문서의 진행 숫자는 스냅샷일 뿐이다. 정상 완료 시 해당 폴더에 `results.json`이 생성된다. 원본 trainer 마지막 줄의 `600 iterations`는 `--stop-after` 구간에서도 예정 horizon을 출력하므로 완료 판정에 사용하지 않는다.

600회×256 regret root traversals, regret/strategy 512/128 optimizer steps, seed 42, CPU 단일 worker/thread다. 학습 후 공개 모델·기존 작은 모델·이번 120/300/600회 모델을 동일 NoRegrets 1M 상대 5명에게 후보당 독립 1,000 deals×6좌석으로 평가한다. 세부 사항은 [중간 규모 학습](fullhouse-medium-training.md)을 따른다.

실행 중인 학습 소스는 해당 run의 `source/`에 복사되어 있다. supervisor가 관리하는 평가기/runtime/자기 자신의 소스, 비교 모델, 엔진을 수정하면 다음 구간의 hash 검사가 실패한다. 새로운 구현은 별도로 준비하고 현재 실행과 버전을 섞지 않는다.

## Git에 저장한 것

- JSON 추론 연결, Rust adapter/bridge, 고정 원본 소스 준비·빌드 코드.
- R1 반례 수정, 평균 전략 oracle, RNG·atomic checkpoint·정확 재개 검사.
- 작은 학습·가중치/fitting 비교·중간 규모 학습/평가 실행기와 설정.
- 연구 판정과 구현 결과 문서, `reports/`의 작은 실험 요약.
- `scripts/export_medium_checkpoint.py`: 실행을 멈추지 않고 최신 저장 시점의 재개 백업을 만드는 도구.

`.cache/`, `artifacts/`, venv, 빌드 결과, 모델 가중치, 원시 대전 로그는 Git에 포함하지 않는다. 각 upstream 고정 commit과 다운로드 경로는 설정/문서에 기록했다. 재사용 코드 라이선스 고지는 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)에 있다.

## 재개 백업

최신 600회 최종 백업의 이름·용량·SHA-256·저장 반복 수는 [reports/checkpoint-export.json](../reports/checkpoint-export.json)에 기록했다. **백업 파일 자체는 현재 환경의 `artifacts/exports/`에 있으며 GitHub에는 올라가지 않는다. 환경을 없애기 전에 파일을 따로 내려받아 보관해야 한다.**

원하는 시점의 더 최신 백업을 만들려면:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python scripts/export_medium_checkpoint.py
```

진행 중인 `latest.pt`는 같은 디렉터리의 atomic replace로 갱신되므로 exporter가 한 번 연 파일은 일관된 저장 시점을 유지한다. 새 백업에는 다음을 포함한다.

- 해당 반복의 전체 모델·regret/strategy reservoir·모든 CPU RNG가 들어 있는 `latest.pt`.
- 저장 시점 이하의 추론 npz와 고정 학습 소스/설정/진단.
- 현재 실행 위치와 재개용 상태(원래 실행 중인 PID를 제거한 복사본).
- 자동 평가에 필요한 NoRegrets 1M blueprint, 작은 비교 모델, 원래 엔진 binary와 라이선스.
- 검사용 smoke 모델과 JSON adapter binary.
- 파일별 SHA-256 manifest.

현재 실행이나 파일은 수정하지 않고 복사본만 준비한다. 별도 NoRegrets 학습기의 큰 `checkpoint-1000000.bin`은 포함하지 않으므로, 이 백업은 Fullhouse 학습 재개와 예정 평가를 위한 것이다. 다른 하드웨어에서 bit 단위로 동일한 결과를 보장하지 않는다.

## 새 환경으로 옮겨 재개하기

이 백업 형식은 **Debian 13(glibc 2.41), Linux x86_64, Python 3.12, `/workspace/deep` 경로**를 기준으로 만들었다. 정확한 패키지 버전은 [환경 기록](../reports/environment.json)에 있다. 비어 있는 새 환경에서 저장소 브랜치를 받고 Python 환경을 준비한다.

```bash
git clone --branch codex/poker-ai-handoff-20261106 https://github.com/cec0948/deep.git /workspace/deep
cd /workspace/deep
bash scripts/setup_fullhouse.sh --training
/workspace/.venvs/fullhouse/bin/python -m pip install pytest==8.4.2
```

Git과 별도로 보관한 `.tar.gz` 파일을 새 환경으로 옮긴다. 이번 푸시와 함께 만든 백업은 `reports/checkpoint-export.json`의 checksum으로 확인할 수 있다. 나중에 만든 백업이면 그 백업의 `.tar.gz.json`에 있는 checksum을 사용한다.

```bash
resume_archive=/path/to/fullhouse-medium-YYYYMMDD.tar.gz
sha256sum "$resume_archive"
# reports/checkpoint-export.json 또는 해당 백업 .tar.gz.json의 sha256과 비교한다.
tar -xzf "$resume_archive" -C /workspace/deep
python scripts/build_fullhouse_r1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python -m pytest tests -q
python scripts/train_fullhouse_medium.py status
python scripts/train_fullhouse_medium.py resume
```

정확한 재개에는 최신 `latest.pt`를 사용한다. 모델만 담은 `iter_XXXXX.pt`나 추론용 `.npz`로는 reservoir/RNG를 복구할 수 없다. hash 불일치가 발생하면 기록된 버전과 파일을 확인하고, 검사를 무조건 해제하지 않는다. Rust를 수정하거나 다른 플랫폼에서 binary를 다시 빌드할 때는 [JSON API 빌드 안내](json-api.md)와 고정 Rust 1.90.0/NoRegrets commit을 사용한다.

## 다음 담당자가 먼저 할 일

1. 상태 명령으로 실제 실행이 계속되는지 확인한다. 이미 실행 중인 학습을 다시 시작하지 않는다.
2. 600회 완료 보고서와 Git에 보존된 원시 평가·결과 파일을 확인한다. 작은 실험에서 개선이 안 보였다는 이유로 확대 학습의 결과를 미리 판정하지 않는다.
3. 중간→최종 모델의 동일 deal 손익 차이와 오차 범위를 확인하고, 개선 가능성이 있으면 독립 학습 seed로 재현한다.
4. 실행 정확성과 강도 판정을 구분한다. 51개 요약 특징의 imperfect recall·공유 신경망 근사·6인 균형 수렴은 별도 이론 쟁점이다.

새 대화에는 다음처럼 요청하면 된다.

> `/workspace/deep/docs/continue-work.md`를 읽고 현재 학습 상태를 확인하세요. 실행 중이면 유지하고, 완료됐으면 학습 및 평가 결과를 분석하세요. 이론적 쟁점만 따로 보고하고 독립적으로 가능한 구현은 계속하세요.
