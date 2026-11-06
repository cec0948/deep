# robopoker 실사 및 실행 기록

검증일: 2026-11-05. 목표: 독립 실행하는 52장 덱의 6인·100BB NLHE AI.

## 현재 판단

**게임 엔진·연구 구성요소의 재사용 후보로 유지한다. 바로 사용할 수 있는 고성능 AI로 채택하지는 않는다.**

소스 체크아웃과 문서에서 학습된 전략 파일·추상화 자료 또는 이들의 공식 다운로드 경로를 찾지 못했다. 확인한 v3.0.0/v1.1.0/v1.0.0 GitHub 릴리스 첨부물은 소스 ZIP/TAR만 포함한다. 다른 배포 경로까지 자료가 없다고 단정하지 않는다.

작성자는 [공개 issue #52](https://github.com/krukah/robopoker/issues/52)의 2026-10-25 답변에서 **다인 blueprint 해를 검증한 적이 없다고 명시**했다. 작성자의 일반화된 코드가 동작할 것이라는 기대를 6인 실력 검증으로 해석하지 않는다. 다른 사용자의 학습 자료 제공 요청에도 그 대화에는 다운로드 경로가 제시되지 않았다.

## 고정한 소스와 환경

- 소스: https://github.com/krukah/robopoker
- 커밋: `b6bf9e2788d43a129c0938f24b278ac2f910a5ed`
- 라이선스: MIT. 재배포 시 저작권·라이선스 고지 보존. 의존성과 모델·자료의 사용 조건은 별도 확인 필요.
- 체크아웃: `.cache/robopoker` (Git 추적 제외). 원본 소스와 Cargo.lock은 수정하지 않았다.
- 도구체인: Rust/Cargo 1.90.0. 상위 저장소의 Cargo.lock을 `--locked`로 사용.
- 현재 환경: cgroup 기준 4 CPU, RAM 32 GiB. NVIDIA 도구는 발견하지 못했다. GPU 사용은 검증하지 않았다.
- Rust 설치 위치: `/workspace/.tools/cargo`, `/workspace/.tools/rustup`.

## 코드에서 확인한 지원

- `crates/pokerkit/src/consts.rs`: sixmax 기능에서 `N=6`, `STACK=200`, `B_BLIND=2`, `S_BLIND=1`. 시작 스택은 100BB다.
- `bin/trainer/Cargo.toml`: `sixmax`가 `pokerkit/sixmax`에 연결되어 있다.
- `crates/kicker/src/table/game.rs`: 실제 다인 상태 머신 `GameN<P>`와 `GameN<6>` 테스트가 있다.
- `crates/kicker/src/settle/showdown.rs`: 다인·사이드팟 분배 테스트가 있다.
- `crates/forge/src/fast.rs`: 표 기반 학습도 초기화 시 추상화 생성과 PostgreSQL을 사용한다.
- `crates/neural/README.md`: 신경망의 Kuhn/RPS 수렴 테스트 두 개가 실패 상태라 ignored라고 명시한다. 이번에는 이 테스트나 대규모 신경망 훈련을 실행하지 않았다.
- `crates/neural/README.md`: 저장 파일은 전략망만 포함하며 advantage net과 학습 메모리는 저장하지 않는다. `load` 성공을 완전한 학습 재개 성공으로 해석하면 안 된다.

## 실행 결과

재현 스크립트: `python scripts/verify_robopoker.py`.

| 검사 | 확인한 결과 |
| --- | --- |
| sixmax 기능의 `six_player` 전용 검사 | 14 통과, 0 실패 |
| sixmax 기능의 `multiway` 검사 | 7 통과, 0 실패; 랜덤 플레이아웃·다인 분배 포함 |
| 기본 모드 kicker 전체 | 215 통과, 0 실패 |
| 기본 모드 mccfr 전체 | 10 통과, 0 실패 |
| Kuhn의 실제 CFR 학습·균형 검사 | 1 통과, 0 실패; 262,144회 반복 후 알려진 균형 조건 검사 |

이 결과는 규칙 엔진과 해당 연구 구현의 일부를 검증한다. NLHE에서의 전략 강도, 6인 전체 학습·추론 통합, 프로 상대 수익률은 검증하지 않았다. 표의 테스트 실행에는 서로 겹치는 검사도 있다.

첫 탐색에서는 kicker/mccfr 전체 테스트에 sixmax를 동시에 적용했다. kicker 테스트 여러 개가 2인 초기 상태를 전제해 실패했고, 종료되지 않는 검사 두 개도 있어 해당 실행을 SIGTERM으로 중단했다. 이것을 전체 테스트 통과나 모든 6인 로직의 실패로 보고하지 않는다. 원본 테스트를 삭제·비활성화하지 않고, 기본 모드 전체 검사와 sixmax 전용 검사를 구분해 재실행했다.

## 채택 전에 해결할 사항

1. **학습 자료:** 기존 강한 체크포인트·카드 추상화 파일을 확보하고 사용 조건 및 6인 설정을 확인해야 한다.
2. **학습 비용:** README의 예시는 학습 16 vCPU/120 GB, DB 8 vCPU/64 GB다. 반면 작성자의 issue #52 답변은 RAM 16GB 장비에서 클러스터링 약 24시간, 학습 수일을 언급한다. 설정과 플랫폼 차이가 있어 어느 쪽도 6인 재현 비용으로 확정하지 않는다. 현재 머신에서 전체 추상화 생성을 시도하지 않았다.
3. **규칙 차이:** `three_player_short_allin_min_raise_pins_lenient_rule`과 `three_player_short_allin_reopens_action_pins_lenient_rule`이 표준적인 full-raise/reopen 처리와 다른 동작을 의도적으로 고정한다. 전원 동일 스택으로 리셋하는 초기 범위와 다른 스택을 허용하는 확장 범위를 구분해 검토해야 한다.
4. **성능 근거:** README의 최고 Slumbot 결과는 평균 -13.1 bb/100 (86K hands)이다. 2인 외부 봇 평가이며 6인 프로급 성능을 입증하지 않는다.
5. **제품 연결:** source-level sixmax 지원은 확인했지만, 전략 자료를 읽는 로컬 의사결정 엔드포인트는 아직 검증하지 않았다.

## 다른 후보의 1차 비교

| 후보 | 확인한 내용 | 판단 |
| --- | --- | --- |
| https://github.com/fedden/poker_ai | README는 변경 없이 20장 덱만 지원한다고 명시. GPLv3 | 표준 52장·6인 목표에 곧바로 맞지 않아 첫 실행 후보에서 제외 |
| https://github.com/EricSteinberger/PokerRL | MIT, 엔진은 다인 지원. 일부 알고리즘/평가는 2인 전용. README는 PyTorch 0.4.1 사용 | 대안으로 보존; 현대화 비용과 완성된 전략 유무를 먼저 확인 |

상용 제품은 현재 네트워크 정책으로 공식 페이지 접근이 차단돼 최신 SDK·엔진 라이선스를 조사하지 못했다. 업체에 연락하거나 제품을 구매하지 않았다.

## 재현

이미 설치된 현재 머신에서는 다음을 실행한다.

```bash
cd /workspace/deep
bash scripts/fetch_robopoker.sh
python scripts/verify_robopoker.py
```

새 머신에서는 먼저 Rust 1.90.0을 공식 rustup으로 설치한다. HTTPS 및 rustup/Cargo 기본 검증을 유지한다.

```bash
curl --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
  https://sh.rustup.rs --output /tmp/poker-rustup-init.sh
CARGO_HOME=/workspace/.tools/cargo RUSTUP_HOME=/workspace/.tools/rustup \
  sh /tmp/poker-rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.90.0
```

검증 스크립트는 체크아웃 커밋과 변경 여부를 확인하고, 각 검사에 제한 시간을 적용하며 실제 종료 코드·테스트 요약·로그 해시를 기록한다. 0개 테스트 실행을 성공으로 판정하지 않는다. 전체 로그와 JSON 결과는 `artifacts/verification/` 아래에 보존한다. 필요한 네트워크 작업은 기존 프록시와 실행 도구의 지원되는 권한 경로를 사용한다.
