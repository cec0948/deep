# Sol 구현 인계 계획

2026-11-06 사용자 요청: 어려운 연구 판단만 별도로 남기고 실제 구현은 Sol에게 맡긴다. 이 문서는 계획과 작업 명세이며, Sol 실행이나 코드 구현이 완료되었다는 뜻은 아니다.

## 목표와 확정 선택

- 사용자 프로그램에서 독립 실행하는 6인 NLHE, 전원 매 핸드 100BB, SB/BB=50/100, 시작 10000칩, 앤티·레이크 없음.
- Fullhouse Deep CFR를 기존 JSONL 인터페이스에 추가한다. 기존 NoRegrets 1M 모델은 기준 상대다. 첫 단계의 성공 기준은 연결·정확성·측정 가능성이고, 주력 모델 승격은 대전 결과를 보고 결정한다.
- 원본 게임·신경망을 다시 만들지 않는다. Fullhouse 인코더/추론/학습기, NoRegrets 규칙 엔진을 재사용한다.
- 첫 연결은 **공개 advantage 모델 + 원본 regret matching**. short-stack overlay와 온라인 탐색은 제외한다. 이를 평균 전략 모델이라고 표시하지 않는다. 후속 평균 전략 export는 별도 artifact로 구분한다.
- 모델은 프로세스당 한 번 로드. stdout은 JSONL, 진단은 stderr. 기존 요청 필드를 유지하고 backend 선택은 실행 옵션에 둔다.
- 변경 사항은 우리 저장소에 남기고 `.cache` 원본을 임의 수정하지 않는다. upstream 수정이 필요하면 재현 가능한 패치와 고정 commit을 관리한다.
- 일반적인 구조 선택·의존성 설치·테스트·기존 환경의 제한된 기능 학습은 Sol이 진행한다. 외부 문의, 새 유료 서버, 외부 배포는 포함하지 않는다.

## 별도 연구 검토로 남길 것 — 세 가지

모델별 능력을 단정하는 분류가 아니라, 구현과 수학적/경험적 판단을 분리하기 위한 역할 배분이다. Sol은 증거와 재현 사례를 수집하고, 아래 판단이 필요한 변경만 따로 정리한다.

| 쟁점 | 확인하기 어려운 이유 | Sol이 준비할 증거 | 검토 전 진행 가능한 범위 |
| --- | --- | --- | --- |
| R1. 실제 학습기가 어떤 regret/평균 전략을 근사하는가 | 공개 artifact는 단일 advantage 모델이고 현재 trainer는 평균 전략도 출력한다. 6인 공유 네트워크, 상대 exploration, reach/iteration weighting, reservoir sampling의 조합이 논문과 같은지는 이름·단위 테스트만으로 판단 불가 | `traverse_fast.py`, worker 및 Numba 경로의 확률·샘플 가중치 흐름; 작은 게임의 손계산/정확 CFR와 비교; 학습용 테스트 결과 | 공개 모델 추론 연결, 현재 수식 그대로 작은 기능 학습. 수식 변경·대규모 학습 확대는 근거를 먼저 검토 |
| R2. 어떤 결과면 6인 성능이 개선되었다고 판단하는가 | 2인 zero-sum 수렴 보장이 6인에 그대로 적용되지 않음. 단일 계열 crossplay와 LBR만으로 프로급/정확 exploitability를 판단 불가 | 공통 엔진, 양방향/좌석별 결과, 독립 학습 seed, 고정 상대 풀과 held-out 상대, deal별 원시 수익, fallback·지연·RAM | 평가 runner와 pilot 구현. 우세가 불확실하면 '미확정'으로 보고하고 자동 승격하지 않음 |
| R3. 공개 가중치·외부 엔진을 배포할 수 있는가 | 자체 코드 MIT와 `engine_vendored/` 별도 조건이 구분되어 있고 `data/` 권한 범위는 명확히 확인되지 않음 | 고정 commit의 LICENSE/출처/파일별 고지; 배포 예정 파일 목록 | 현재 로컬 검증과 코드 연결. 가중치·외부 엔진 재배포는 범위 확인 후 결정. 사용자에게 재문의하거나 저자에게 연락하는 작업은 요청되지 않음 |

R1에서 아직 확인되지 않은 것을 버그로 단정하지 않는다. NoRegrets의 VR+off-policy 보정식 반례는 **다른 엔진의 문제**이며 Fullhouse에 그대로 적용하지 않는다. 연구 검토가 필요해도 관련 없는 연결·로그·평가 구현은 계속한다.

## 이미 확인한 사실 — 재조사 최소화

- NoRegrets commit: `757f7692738069522195d2b486eec60a8b010c0c`.
- Fullhouse commit: `e504793d480b1b975f25258d25939b45c6dbd5a4`.
- Fullhouse 원본: `.cache/fullhouse-bot`. 공개 모델: `data/deep_cfr_model.npz`, 약 577 KiB, dueling advantage, 입력 51/행동 5.
- 모델 SHA-256: `1102326b68da95564de147106612df71cb891b42f0726ba0212d3b9a5bcae295`.
- preflop table SHA-256: `6b3b74778854fcebcd3179753c0252392e5f4118e6b623375c98fafa372863b3`.
- 실제 공개 모델 조회: `examples/query_fullhouse.py`. 위 commit·파일 해시를 검사한다.
- upstream `tests/test_bot.py`, `tests/test_bot_scenarios.py`, `tests/test_encoding.py`: 29개 통과. 학습/Numba 전체 검증은 아님.
- NoRegrets 1M 모델은 `artifacts/selfplay/current.json`으로 찾는다. `scripts/run_poker.py`가 현재 실행 경로다. 기존 JSON 테스트 7개 통과 기록이 있다.
- Fullhouse 실행 venv: `/workspace/.venvs/fullhouse`; Python 3.12, NumPy 2.2.6, eval7 0.1.10, pytest 8.4.2. PyTorch/Numba 학습 환경은 추가 준비 필요.
- 마지막 확인 환경은 4 CPU/32 GiB, GPU 미확인. 실행 전에 실제 자원을 확인하고 기존 CPU 캠페인처럼 RSS 20 GiB 제한을 둔다. GPU가 없으면 CUDA 테스트를 수행했다고 기록하지 않는다.
- Git에는 이전 작업의 미커밋/미추적 파일이 많다. reset/clean으로 지우지 않는다. 기존 artifacts와 원본 캐시를 보존한다.

## Sol 작업 순서와 완료 기준

### S1. 재현 가능한 후보 준비

작업: pinned source fetch/검증, inference 의존성 파일, 모델 메타데이터 manifest를 만든다. 공개 npz를 저장소 배포 파일에 복사하지 않고 원본 확보 경로와 해시로 관리한다. source와 model 타입, 51개 특징/5개 행동의 버전, 규칙 조건, 강도 검증 상태를 기록한다.

완료: 새로운 checkout에서 고정 commit과 해시를 검사하고 `query_fullhouse.py`를 실행 가능. 파일 누락·해시 불일치가 명시적인 실패다. 기존 29개 검사는 관련 변경이 있으면 재사용하며 불필요하게 반복하지 않는다.

### S2. 모델과 분리된 공개 상태 재생

작업 대상: `native/noregrets-json/adapter.rs`, `scripts/build_json_adapter.py`. 기존 replay 검증을 재사용하되 Fullhouse 조회를 위해 NoRegrets 364MB blueprint를 로드하지 않아도 되도록 상태 재생 경로를 분리한다. 내부 helper 또는 별도 실행 모드는 Sol이 선택한다.

출력 계약: 현재 actor/button/street, 본인 hole·공개 board, pot, to_call, raise_bounds, 각 좌석 stack/street commitment/fold/all-in 상태, 공개 action history와 street 경계. terminal에서는 정책 결정을 거절한다. 상대 개인 카드와 미래 보드를 출력하거나 모델 입력에 넣지 않는다.

기존 `Table::apply_concrete`는 abstraction을 인수로 받는다. 단순히 함수 이름을 바꾸지 말고 순수 규칙 replay와 NoRegrets shadow history의 의존성을 정리한다. placeholder cards는 공개 상태 재생에만 쓰며 평가 시뮬레이터의 실제 showdown 결과에 쓰지 않는다.

완료: all streets/버튼 회전/오프그리드 레이즈/short all-in·레이즈 재개/사이드팟 관련 상태 비교. 다른 placeholder deck으로도 정책 입력이 동일한지 확인. 기존 JSON 오류 처리와 NoRegrets 호출 회귀 없음. 규칙 차이를 발견하면 최소 행동 기록을 남기고 해당 case만 미지원/실패로 표시하며 조용히 clamp하지 않는다.

### S3. Fullhouse backend와 공통 실행기

작업: 예를 들어 `python scripts/run_poker.py --backend fullhouse`를 지원한다. 인자 없는 기존 실행은 NoRegrets를 유지한다. public state → Fullhouse native state → `encode_state_dict`/`DeepCFRLookup.get_strategy` → 액션 변환을 연결한다.

구현 계약:

- 원본 `bot.decide()`의 예외를 숨기는 경로 대신 직접 모델을 호출한다. non-finite/잘못된 shape/확률 합/불법 액션은 명시적으로 거절한다.
- 합법 마스크 적용과 정규화 뒤 최종 분포에서 seed로 샘플링한다. 같은 구체 행동으로 합쳐지는 확률은 합산 가능하며 선택 행동과 보고 분포가 일치해야 한다.
- raise는 **street 누적 raise-to**. half-pot/pot은 원본의 콜 이후 pot 공식과 rounding을 따른다. all-in call과 raise를 구분한다.
- 확률 해석이 달라지는 메뉴/특징을 임의 확장하지 않는다. 51개 특징만 쓰므로 과거 전체 기록을 기억하는 모델이라고 설명하지 않는다.
- 응답에 backend, model identity/type, policy_source, strength_verified=false를 추가한다. Fullhouse의 모르는 학습 반복 수는 null 또는 별도 미상 필드로 표시한다. NoRegrets의 1M 값을 복사하지 않는다.
- short-stack overlay는 이번 기본 경로에서 끈다. 후속 추가 시 policy source와 분포를 별도로 맞춘다.

완료: 실제 공개 모델로 전체 street 입력·여러 요청·같은 seed 재현·잘못된 JSON 후 복구·모델 누락/변조 검사를 통과. 네이티브 Fullhouse 입력과 변환된 입력의 51개 특징·마스크·확률이 일치한다. 실제 요청에서 숨은 fallback 없이 신경망 확률을 반환한다.

### S4. 같은 엔진에서 대전 평가

작업: 평가만 사용하는 최소 게임 실행 인터페이스를 기존 NoRegrets 규칙 엔진에 추가/재사용하여 두 정책을 대전시킨다. 실제 전체 deck과 showdown은 신뢰된 평가기 안에 유지하고 정책별로 자기 카드·공개 정보만 제공한다. Fullhouse vendored engine을 전체 서비스에 편입하지 않는다.

초기 pilot: 1 Fullhouse vs 5 NoRegrets, 역방향, 같은 정책끼리 sanity 비교. 각 방향 최소 100개 독립 deal × 6 hero 좌석을 후보 시작값으로 하고, 시간 한도에 맞게 조정하되 실제 완료 수를 기록한다. pilot은 수익률 결론용이 아니다.

저장: deal ID/seat/seed/모델 hash/칩 수익, BB/100, actor별 실패 및 fallback, street·잔존 인원, 응답시간 p50/p95, 자원과 로그. 반복 좌석은 독립 표본으로 세지 않고 deal 평균 또는 deal-block bootstrap으로 CI를 계산한다. 두 방향은 상대 구성부터 다르므로 수익률 합이 0이어야 한다고 테스트하지 않는다. 같은 seed만 같다고 paired deal이 보장되지 않으므로 deck 생성과 행동 RNG를 분리한다.

완료: 칩 보존·개인 정보 차단·좌석 순환·완료 핸드 수·유효 액션 검증. 600초 등 제한을 두고 중단된 실험은 incomplete로 기록한다. 비교 결과를 파일로 남긴다. 추가 대규모 평가는 R2 기준과 비용을 보고 결정한다.

### S5. 작은 직접 학습과 재개

작업: 별도 학습 환경/설정/wrapper를 만들고 원본 trainer를 호출한다. 첫 실험은 workers=1, 작은 batch/buffer, `--no-stack-randomization`, `--save-resume-buffers`로 시작한다. 예비값: capacity 2,000/player 및 strategy 2,000, batch 64, train steps 2, traversals 8/iteration, 총 12 iterations. 최소 두 차례의 6인 traverser 순환과 실제 파라미터 갱신이 발생했는지 검사한다. 자원/데이터가 부족하면 값과 이유를 기록해 조정한다.

정확성 주의:

- 공개 npz 하나는 정상 resume용 checkpoint가 아니다. 새 학습으로 명시하고 모델 artifact를 분리한다.
- 코드에는 **플레이어별 경험 버퍼 6개와 공유 advantage net 1개**가 있다. 플레이어별 네트워크 6개라고 가정하지 않는다.
- 원본 `--resume`은 파일이 없으면 새 학습으로 넘어간다. wrapper는 실행 전에 반드시 실패시킨다. config/iteration/buffer/model 호환성과 저장 atomicity를 확인한다.
- 현재 checkpoint에 RNG 상태가 보이지 않고 resume 시 RNG가 seed에서 다시 시작한다. 총 iteration에 따라 exploration schedule도 달라질 수 있다. 계획된 총 학습 horizon을 고정하고 중단 지점만 분리한다. 재현 가능한 재개를 구현하려면 Python/NumPy Generator/전역 NumPy/Torch/CUDA/worker RNG 등 실제 사용 상태를 식별해 저장·복원한다.
- optimizer는 현재 코드가 매 반복마다 새로 만든다. checkpoint에 optimizer가 없다는 이유만으로 버그라고 단정하지 않는다. checkpoint 경계에서 연속 실행과 무엇이 같아야 하는지를 정의한다.
- 원본 GPU cache 선택 기준 `gpu_buf_limit = 18e9`는 12GB급 4070의 실제 여유 VRAM을 반영하지 않는다. 실제 free VRAM과 학습 여유분을 고려하거나 CPU sampling으로 명시적으로 전환한다. 실 GPU가 없으면 이 경로는 미검증으로 보고한다.

완료: 유한 loss/가중치, 실제 업데이트, export/추론, 새 프로세스에서 재개, 누적 iteration·버퍼 유지 검사. RNG까지 고치지 않았다면 '학습 상태를 보존하는 재개'까지만 주장하고 uninterrupted와 정확 동일하다고 쓰지 않는다. 기능용 checkpoint를 현재 주력 모델로 자동 등록하지 않는다. GPU 대규모 학습은 R1/R2 검토 이후다.

## 인계 산출물과 보고 형식

각 단계 완료 시 파일 경로, 실행 명령, 검사 결과, 남은 불확실성만 간단히 기록한다. 최종 산출물은 pinned fetch/환경 설정, 공통 상태 replay, Fullhouse backend, 공통 평가 runner와 pilot 결과, bounded training/resume wrapper와 작은 checkpoint다.

계속 진행할 수 있는 구현은 Sol이 직접 처리한다. 별도 판단 요청이 필요할 때는 R1/R2/R3 ID, 실제 코드 위치/재현 명령, 관측 결과, 선택지·권고만 제시한다. 논문을 다시 광범위하게 검색하는 작업부터 시작하지 않는다.

## Sol 시작 메시지

> `/workspace/deep/docs/sol-handoff.md`를 읽고 S1부터 순서대로 구현하세요. 6인·100BB Fullhouse Deep CFR를 기존 JSONL에 연결하고, 동일 규칙 엔진 대전 pilot과 제한된 새 학습/재개까지 검증하는 작업입니다. 이미 확인한 사실을 재조사하기보다 재사용하고, S1~S5의 완료 기준으로 진행하세요. 연구 판단 R1~R3는 증거와 선택지를 별도 보고하되 독립적인 구현은 계속하세요. 기존 모델·미커밋 파일을 보존하고 공개 모델 실력이나 재개 정확성을 검증 이상으로 주장하지 마세요. 새 유료 서버나 외부 연락은 수행하지 마세요.
