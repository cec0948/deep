# 자체 프로그램용 로컬 JSON 연결

2026-11-06 추가: `python scripts/run_poker.py --backend fullhouse`로 동일 요청 형식의 신경망 backend도 사용할 수 있다. `backend=fullhouse`, `model_type=advantage`, `model_iterations=null`, `policy_source=neural_policy`를 반환한다. 새 구현·검증 결과와 제약은 `docs/sol-implementation-results.md`를 참고한다. 아래는 기존 NoRegrets backend 설명이다.

2026-11-06: 기존 NoRegrets 소스를 재사용하는 얇은 JSONL 프로세스를 구현했다.
포커 규칙·CFR·전략 조회 알고리즘을 새로 구현하지 않았다. 고정한 upstream의 public 메서드를 직접 호출하며 원본 파일은 변경하지 않는다.

## 지원 범위

- 6인, 전원 매 핸드 100BB, SB/BB 50/100칩, 레이크·앤티 없음.
- 로컬 blueprint 조회. 전략 파일은 프로세스 시작 시 한 번 로드한다.
- 입력: 현재 액션을 결정할 플레이어의 카드, 공개 보드, 핸드 시작부터의 모든 베팅 액션.
- 출력: 선택한 액션, 모델 액션 확률, 기본 동작 대체 여부.
- 프리플랍/플랍/턴/리버 지원. 임의의 합법적인 상대 베팅은 upstream의 shadow-history 매핑을 사용한다.
- 온라인 탐색, 가치망, 전략적 클러스터링의 predecessor 파일, 서로 다른 시작 스택은 아직 지원하지 않는다.
- 상용·비공개 버전의 파일 포맷 호환성은 확인되지 않았다.

**연결 성공은 강한 모델 확보를 의미하지 않는다.** 현재 파일은 누적 220회 학습의 기능 확인용 모델이다. 모든 응답은 `strength_verified: false`를 포함한다.

## 빌드와 실행

현재 환경에는 고정 소스·Rust 1.90.0·기능 확인용 모델이 남아 있는 것을 확인했다.

```bash
cd /workspace/deep
python scripts/build_json_adapter.py
python -m unittest discover -s tests -v
python examples/query_poker.py artifacts/noregrets-smoke/run-NK70RShx/resumed.bin
```

새 머신에서 소스나 모델이 없으면 먼저 `bash scripts/try_noregrets.sh`를 실행한다. 새로 생성한 `artifacts/noregrets-smoke/run-*/resumed.bin` 경로를 예제에 전달한다. 테스트용 모델 경로는 `POKER_TEST_MODEL` 환경 변수로 지정할 수 있다. 기존 모델의 테스트 경로가 없을 때 검사를 건너뛰거나 0개 검사로 성공 처리하지 않는다.

원본 Cargo.toml·Cargo.lock 의존성과 라이선스를 재사용하고, 소스 모듈 참조와 JSON 진입점만 `.cache/noregrets-json`에 생성한다. 배포 시 upstream 및 의존성 라이선스 고지를 함께 보존해야 한다.

## 프로토콜

실행 파일: `.cache/noregrets/target/release/poker-json MODEL.bin`
표준 입력 한 줄마다 JSON 요청 하나, 표준 출력 한 줄마다 JSON 응답 하나를 반환한다. 진단은 stderr다. 동일 프로세스에서 여러 요청을 보낼 수 있으며 각 요청은 전체 액션 기록을 포함한다.

```json
{"hero":3,"button":0,"hole":["As","Kd"],"board":[],"actions":[],"seed":42}
```

- 좌석 번호: 0~5. 버튼이 0이면 프리플랍 첫 액션은 3, 플랍 이후 첫 생존 좌석은 1부터 시작한다.
- `hole`: 본인 카드 2장. `board`: 현재 공개된 카드만 0/3/4/5장. 형식은 `As`, `Td`, `2c` 등.
- `actions`: 블라인드 자동 납부 이후의 모든 액션. 카드 배분 이벤트는 넣지 않는다.
- 액션 예: `{"seat":3,"kind":"raise","to":357}`, `{"seat":4,"kind":"call"}`, `{"seat":5,"kind":"fold"}`, `{"seat":2,"kind":"check"}`.
- `to`: 그 스트리트에 넣은 **누적 칩 금액**이다. 올인은 합법적인 최대 raise-to로 표현한다. 콜로 올인하는 경우 `call`을 쓴다.
- `seed`: 액션 추출·Monte Carlo 카드 분류 RNG에 사용하는 요청별 시드. 같은 요청과 같은 모델은 재현 가능한 응답을 낸다.

핸드 기록은 실제 규칙 엔진에서 재생해 차례, 체크/콜 구분, 레이즈 범위를 검증한다. 최종 차례가 hero가 아니거나 보드 길이가 진행 단계와 맞지 않으면 `ok:false`로 거절한다. 알 수 없는 필드와 중복 카드도 거절한다. 다른 플레이어의 개인 카드를 전달할 필드는 없다.

누락된 상대 카드와 미래 보드는 내부 재생용 placeholder다. 추론은 현재 플레이어의 카드·공개 보드·공개 액션만 읽는 upstream `Policy::blueprint_dist`를 호출한다. placeholder를 사용한 온라인 탐색은 실행하지 않는다.

## 응답과 진단

- `ok`: 요청 성공 여부. 실패 응답은 `error` 문자열을 포함한다.
- `action`: 실제로 내 프로그램에서 적용할 액션.
- `distribution`: upstream 추상 액션 각각의 구체 액션과 확률. 여러 추상 액션이 같은 금액으로 변환될 경우 해당 확률을 합산해서 표시할 수 있다.
- `policy_source`: `blueprint`, `unseen_or_zero_mass_fallback`, `menu_mismatch_fallback` 중 하나. 미학습 상황이나 액션 메뉴 불일치를 정상적인 학습 전략과 구분한다.
- `pot`, `to_call`, `raise_bounds`: 현재 상태의 칩 기준 값. `raise_bounds`는 `[최소 raise-to, 최대 raise-to]` 또는 `null`이다.
- `model_iterations`: 파일에 기록된 누적 반복 수. 모델의 강도를 보증하는 값이 아니다.
- `strength_verified`: 현재 항상 false.

blueprint 포맷에는 학습 시작 스택이 기록되지 않는다. 프로그램은 **100BB 모델이라는 출처를 별도로 확인**해야 한다. 연결부의 `starting_stack_bb:100`은 재생한 게임 조건을 뜻하며, 파일의 학습 조건을 검증했다는 뜻이 아니다.

## 실제 검증

2026-11-06 통합 테스트 7개 통과: 실제 모델 조회, 같은 요청 재현, 모든 스트리트, off-tree 베팅, blueprint hit와 fallback의 구분, 잘못된 관측 거절, JSON 오류 후 계속 실행, 모델 누락 시 시작 실패를 포함한다. 강한 전략, 프로 상대 수익률, 상용 모델 호환성은 미검증이다.
