# Fullhouse Deep CFR를 이 프로젝트에 사용하는 방법

2026-11-06 확인. 후보: https://github.com/advitrocks9/fullhouse-bot
고정 커밋: `e504793d480b1b975f25258d25939b45c6dbd5a4`.

## 권장 구조

우리 프로젝트가 공개 관측 입력, 규칙 검증, 모델 선택, 평가 및 학습 기록을 담당하고, Fullhouse의 특징 인코더와 신경망 추론·학습 코드를 재사용한다. 기존 NoRegrets는 비교 모델로 남긴다. Fullhouse를 검증 우선 후보로 올리되 성능 비교 전 주력 모델로 자동 승격하지 않는다.

예정된 흐름:

`사용자 프로그램 → 기존 JSON 요청 → 공개 상태 재생/검증 → backend별 변환 → 모델 확률 → 합법 액션 변환 → 기존 JSON 응답`

NoRegrets는 기존 Rust 실행기를 사용하고, Fullhouse는 별도 Python 프로세스에서 모델을 한 번 로드한다. `--backend fullhouse` 같은 선택 옵션을 추가하는 설계이며 **아직 이 공통 인터페이스를 구현한 상태는 아니다**.

## 이번에 직접 확인한 결과

- `.cache/fullhouse-bot`에 원본을 확보하고 commit을 고정했다. 원본 소스 수정 없음.
- `data/deep_cfr_model.npz`: 약 577 KiB, 입력 51개, 행동 5개. 실제 배열은 **dueling advantage network**다. README의 Deep CFR 명칭만으로 평균 전략 신경망 또는 SD-CFR 모델 집합이라고 가정하지 않는다.
- `data/preflop_equity.npz`: 약 2.2 KiB, 특징 계산에 필요한 별도 데이터.
- NumPy만으로 신경망 순전파가 가능하다. postflop 특징 계산에는 `eval7`도 필요하다. GPU는 추론의 필수 조건이 아니다.
- upstream `test_bot.py`, `test_bot_scenarios.py`, `test_encoding.py`: **29개 통과**. 테스트에는 stub 전략도 포함되며 실제 대전 승률 검증이 아니다. PyTorch·Numba 학습 검사는 실행하지 않았다.
- 예제 `examples/query_fullhouse.py`는 실제 공개 가중치를 직접 조회한다. 입력은 올바른 6인 100BB 첫 액션 상태다. `decide()`의 예외 흡수 경로 대신 `DeepCFRLookup.get_strategy()`를 호출하여 모델 오류가 대체 행동에 가려지지 않게 한다.

현재 격리된 실행 환경에서:

```bash
cd /workspace/deep
OPENBLAS_NUM_THREADS=1 /workspace/.venvs/fullhouse/bin/python examples/query_fullhouse.py
```

이 예제는 Fullhouse 원래 상태 형식의 고정 입력으로 추상 행동 확률을 출력한다. 기존 `scripts/run_poker.py`를 대체하지 않으며, 공통 API 연결 완료를 뜻하지 않는다.

실행 환경: Python 3.12 venv `/workspace/.venvs/fullhouse`, numpy 2.2.6, eval7 0.1.10, pytest 8.4.2. 원본 다시 확보 시 commit과 위 모델 두 파일 해시를 검사한다. 해시는 예제 코드에 기록했다.

## 재사용할 부분

| 구성 요소 | 활용 |
| --- | --- |
| `bot/features.py` | 훈련/추론 공통 51개 특징 인코더를 그대로 사용 |
| `bot/deep_cfr_lookup.py` | npz 로드, regret matching, 추론용 평균 전략 모델 로드 기능 |
| `training/` | 시뮬레이션·순회·네트워크·경험 버퍼·학습 저장 경로 재사용 후보 |
| `tests/` | 인코딩/합법 행동/학습 가중치 검증을 재사용 |
| `bot/bot.py` | 행동 금액 변환과 short-stack overlay 참고. 통째로 호출하면 예외를 숨기므로 상태를 구분해서 연결 |
| `engine_vendored/` | 별도 라이선스 조건의 외부 엔진. 배포 묶음으로 바로 편입하지 않음 |

`LICENSE`는 자체 작성한 bot/training/tooling 등에 MIT를 명시하고 vendored engine은 제외한다. 모델이 있는 `data/`의 권한 범위는 별도로 명시적으로 확인되지 않았으므로 공개 저장소에 있다는 사실만으로 재배포 가능을 확정하지 않는다. 이번에는 로컬 검증만 수행했다.

## 연결 코드에서 처리할 것

1. **공개 상태 변환:** 기존 입력 `hero/button/hole/board/actions`를 재생하여 `seat_to_act/dealer/your_cards/community_cards/players/pot/amount_owed/current_bet/min_raise_to/action_log`를 만든다. 기존 Rust 재생기에 각 좌석 stack·street bet·fold/all-in 상태와 street 경계 공개 출력을 추가하는 방법이 있다. 현재 응답 필드만으로는 충분하지 않다.
2. **칩·액션 의미:** 기존 50/100 blind 및 10000 시작 스택과 일치시킨다. 우리 `raise.to`와 Fullhouse `raise.amount`는 해당 스트리트 누적 raise-to로 연결한다. 하프팟/팟은 콜 이후 팟 기준 공식을 원본과 동일하게 사용한다. all-in이 콜인지 레이즈인지도 규칙 엔진으로 확인한다.
3. **학습과 입력 일치:** 51개 특징 순서와 5개 행동을 임의로 바꾸지 않는다. 버튼/좌석 회전, 스트리트별 기록, short all-in의 레이즈 재개 여부, 사이드팟을 비교한다. 전체 행동 기록은 51개 요약 특징으로 압축되므로 모든 과거 정보를 기억하는 신경망은 아니다.
4. **실패를 노출:** 원본 `decide()`는 모든 예외를 잡고 기본 행동을 반환한다. 우리 연결부는 `neural_policy`, `short_stack_overlay`, `fallback`, `error`를 구분하고 확률과 실제 선택 행동의 출처를 일치시킨다. 확률을 출력한 뒤 다른 휴리스틱으로 액션을 바꾸는 경우 같은 모델 결정으로 표시하지 않는다.
5. **공통 평가:** 양쪽 정책을 같은 규칙 엔진에서 실행한다. 모델별 자체 대회 결과의 BB/100을 직접 비교하지 않는다. 고정 6인 100BB, 레이크 없음, 좌석 순환, 동일 deal, 독립 seed/상대 풀을 사용한다. 일반화 주장은 모델이 출력한 확률이 아니라 대전 결과로 판단한다.

## 공개 모델 사용과 직접 학습의 차이

공개 npz는 추론용이다. 이 checkout에서 재개용 `.pt/.pth` 파일을 찾지 못했다. 현재 trainer는 공유 advantage 모델 하나, 평균 전략 모델, 플레이어별 경험 버퍼 등을 쓰므로 npz 하나로 정상적인 전체 학습 상태를 복원할 수 없다. 다른 학습기의 기존 1M `.bin`도 Deep CFR 체크포인트로 변환되는 것은 아니다.

첫 실행은 **공개 모델 평가**, 다음은 **기존 trainer를 이용한 작은 새 학습**이다. 가중치를 일부 초기값으로 활용하는 것은 별도 이식 실험이며 정상 resume과 구분한다. 증류도 가능하지만 기존 모델의 약점을 옮길 수 있으므로 우선순위는 낮다.

직접 학습 시 확인한 중요 옵션:

- `--no-stack-randomization`: 현재 목표인 매 핸드 전원 100BB에 맞춘다. 원본 기본값은 stack randomization 활성화다.
- `--save-resume-buffers`: 이후 재개에 필요한 버퍼를 저장한다. `--resume`만 지정하고 파일이 없으면 원본은 새 학습을 시작하므로 우리 wrapper는 이를 오류로 처리해야 한다.
- 원본 기본 workers=16, 플레이어별 buffer capacity=5,000,000, 별도 strategy buffer=5,000,000. 현재 4 CPU 환경에 그대로 적용하지 않는다.
- buffer 배열 자체만 계산해도 기본 7개 버퍼 × 5M × (51+5+1+1) float32 ≈ 8.12 GB다. worker 복사, GPU 캐시, 샘플/직렬화 임시 메모리는 별도이므로 RTX 4070의 VRAM만으로 학습 자원 요구량을 판단하지 않는다.
- CPU 기능 검사 후 GPU 프로파일을 별도 만든다. 입력·행동·모델 export와 로드 일치, 재개 상태, 수치 유한성, 작은 대전 결과까지 확인하고 확장한다.

## 완료된 것과 다음 작업

완료: 공개 모델 확보·형상/해시 확인·실제 조회 예제·관련 upstream 검사 29개 통과·연결 설계.

다음: 공통 공개 상태 출력 → Fullhouse adapter → 같은 엔진에서 대전 비교 → 작은 학습 및 재개 검증 → GPU 학습 규모 결정. 현재 모델의 프로급 실력, 현재 프로젝트와 완전한 규칙 일치, 재학습 성능 향상은 아직 검증하지 않았다.
