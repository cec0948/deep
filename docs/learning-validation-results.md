# 학습기 검증 결과: 작은 게임 통과, 6인 확대 보류

2026-11-09. 실행 `20261109T055825812142Z`. **고정 버퍼 비교와 작은 게임의 학습 전체 검사를 완료했다.** 조건부 표적 평균을 사용하는 개선 후보는 Kuhn poker의 세 seed에서 모두 사전 기준을 통과했다. 그러나 실제 6인 replay의 적합도 기준은 두 후보 모두 실패했다. 새 6인 자기 대전·대전 평가·유료 학습을 시작하지 않았고 기본 모델도 교체하지 않았다.

사전 [검증 계획](learning-validation-plan.md), [고정 설정](../configs/fullhouse-learning-validation.json), [기계 판정](../reports/learning-validation/20261109T055825812142Z/gates.json), [전체 증거](../reports/learning-validation/20261109T055825812142Z/)를 보존했다. 설정 SHA-256은 `d45e1ecd554d915848a0c9c4905a73e57ce7ffe0ea2c0e754be3eee485823f26`다. 실패 이후 통과 기준을 낮추지 않았다.

## 1. 고정 6인 버퍼: 학습량만 늘려서는 통과하지 못함

V3 120회의 세 replay를 그대로 두고 새로운 초기화로 fitting했다. 기존 설정은 512×128, 두 후보는 최종 8,192×512다. 버퍼/초기화 seed는 (42,7), (43,8), (44,9)로 미리 고정했다. 후보 중 하나는 기존 함수를 그대로 쓰고 예산만 늘렸으며, 다른 하나는 상태 그룹별 비중을 균형화하고 동일 preflop 입력의 표적을 시간 가중 평균했다.

아래 TV는 **UTG의 169개 입력에서 경험적 평균 regret을 regret matching한 정책과 신경망 정책 사이의 시간 가중 총변동거리**다. 평균 전략 신경망의 TV와 다른 지표다. 0은 일치, 1은 최대 차이이며 통과 기준은 ≤0.10이다.

| replay seed | 기준 512회 fitting | 예산 확대 | 상태 균형·조건부 평균 |
| --- | ---: | ---: | ---: |
| 42 | 0.6766 | 0.5458 | 0.4959 |
| 43 | 0.5419 | 0.4517 | 0.4802 |
| 44 | 0.5714 | 0.5707 | 0.5825 |

두 후보는 전체 MSE와 각 스트리트 MSE의 악화 방지 기준을 모두 통과했지만, UTG 전략 오차는 모두 실패했다. 전체 MSE는 기준 대비 예산 확대에서 약 6.9~7.9%, 상태 균형에서 6.2~8.9% 감소했다. 이것을 실력 향상으로 해석하지 않는다.

AA 폴드율은 예산 확대 세 모델 모두 0%, 상태 균형은 0%·0%·25%였다. **새 초기화의 512회 기준 모델도 세 개 모두 AA 폴드 0%였다.** 따라서 원래 seed 42 모델의 29%가 사라진 것만으로 개선 효과를 주장할 수 없다. AA 한 가지를 보고 후보를 고르지 않는 기준이 실제로 필요했다.

## 2. 저장 표적 자체의 변동성과 신경망 용량을 구분

후속 진단은 채택용 후보가 아니며 통과 기준을 바꾸지 않았다. 상세 계산은 [target-diagnostics.json](../reports/learning-validation/20261109T055825812142Z/target-diagnostics.json)에 있다.

- UTG의 169개 패 입력마다 저장된 표본 수 중앙값은 20·18·17개다. 전체 regret 데이터의 시간 가중 질량에서 이 UTG 상태들이 차지하는 비율은 약 0.64~0.66%다.
- 각 입력의 행을 무작위로 절반씩 나눠 별도로 평균하고 regret matching하면, 두 정책 사이 평균 TV가 약 **0.59~0.60**이다. 100번의 분할을 수행했다. 이는 유한 표적의 민감도 진단이며 독립 표본의 신뢰구간이 아니다.
- 같은 입력에서도 원시 regret 표적이 크게 달라진다. 각 입력을 정확한 경험적 조건부 평균으로 예측해도 남는 행별 MSE는, 0을 예측할 때의 MSE의 약 92~93%다. 경로 샘플링과 반복별 정책 변화 등의 변동이 섞여 있으며 원인별 기여를 분리하지 않았다.
- 같은 256폭 신경망에 **UTG 169개의 조건부 평균 표적만** 학습시키면 TV가 0.0404까지 내려갔다. 모든 표적에 동일한 양의 상수 하나를 적용해 학습 크기를 맞춘 진단에서는 약 `2.5e-7`까지 내려갔다. 샘플별 centering이나 AA 행동 규칙은 사용하지 않았다.

마지막 진단은 이 작은 표를 표현할 신경망 용량이 있다는 증거다. 그 모델은 다른 포커 상태를 학습하지 않았고 배포 후보가 아니다. 또한 불안정한 유한 표적을 잘 외우는 것이 강한 포커 전략이라는 뜻도 아니다. **표적의 변동성과, 여러 상태를 함께 학습할 때의 적합도 문제가 모두 남았다**는 범위까지 확인했다.

## 3. Kuhn poker: 실제 구성요소를 연결한 학습 전체 검사

각 플레이어의 64개 결정적 전략을 전수 조사해 best response를 계산했다. 알려진 균형의 값 `-1/18 ante`, exploitability 0을 재현했다. 균등 정책의 실제 `traverse_external` 출력도 가능한 카드와 무작위 경로를 모두 열거한 손계산과 일치했다.

학습에는 실제 V3 순회 함수, reservoir, 같은 256폭 신경망, regret fitting, importance 평균 수집기와 평균 신경망 fitting을 사용했다. 게임 규칙과 입력 표현은 완전기억의 Kuhn용으로 교체했다. 회당 순회 256개, regret fitting 256×128, 120회 학습이며 평균 신경망은 12·60·120회에 학습·측정했다. **6인 설정의 그대로인 성능 검사가 아니다.**

exploitability는 두 플레이어의 최적 대응 개선량 합의 절반이며 단위는 ante/hand다. 낮을수록 좋다. 초기 균등 전략은 0.4583, 최종 통과 기준은 ≤0.05다.

| 방식 | seed 71 | seed 72 | seed 73 | 세 seed 통과 |
| --- | ---: | ---: | ---: | --- |
| 기존 regret fitting | 0.06713 | 0.05926 | 0.02866 | 실패 |
| 상태 균형·조건부 평균 | **0.01500** | **0.02409** | **0.02193** | **통과** |

개선 후보는 정확한 own-reach 평균 대비 collector의 최대 TV가 0.0036~0.0127, 버퍼 평균 대비 신경망 최대 TV가 0.0037~0.0061로 각각 기준 0.05·0.03을 통과했다. 최종 checkpoint에서 네트워크·버퍼·누적 정확 평균을 다시 읽어 정책과 best response를 재계산했다.

독립적인 표 기반 Linear CFR은 120회에서 0.02337, 5,000회에서 0.00373이었다. 순회 방식과 업데이트 예산이 다르므로 이 숫자만으로 신경망 방식의 우열을 선언하지 않는다. 개선 후보의 seed 72는 60→120회에 0.02285→0.02409로 소폭 상승했다. 단조 개선이나 일반적인 수렴률을 입증한 실험도 아니다.

기존 방식에서 평균 수집·평균 신경망 오차는 이미 기준 안이었다. 그 상태에서 원래 정책들의 정확한 평균도 120회 exploitability가 0.06659·0.05697·0.03098이었다. 이 작은 게임에서는 평균 신경망 하나만 고쳐 해결할 상황이 아니었다. 개선 후보는 상태 재가중과 조건부 평균을 함께 바꿨으므로 둘의 개별 기여는 분리하지 않았다.

## 4. 구현과 재현성

- 일반 회귀 검사 **31개 통과**. 최종 gate의 같은 방식 결합 조건 등을 보강한 뒤 관련 검사 5개도 다시 통과했다.
- `check_learning_gates.py --require-six-player`는 같은 후보가 고정 replay 세 개와 검증된 Kuhn 세 seed를 모두 통과해야 종료 코드 0을 낸다. 이번 실행에서는 **종료 코드 2**로 6인 진입을 거절했다. 미완료 결과, 다른 후보끼리의 통과 결과 합치기, 변경된 설정·소스·모델·checkpoint도 허용하지 않는다.
- 고정 버퍼, 학습 소스, 출력 모델, 실행 스크립트, 검증 checkpoint의 해시와 원시 결과를 보존했다. 과거 V3 엔진·학습기·버퍼·기본 모델은 수정하지 않았다.
- 첫 Kuhn 기존 방식 실행은 120회 checkpoint 저장까지 완료했지만 마지막 판정 JSON에서 NumPy bool 직렬화 오류가 났다. 수치 결과를 바꾸지 않고 기본 Python 수치형으로 내보내도록 수정했다. 원래 실패 로그와 보고서를 유지하고, checkpoint에서 정책을 재구성해 별도의 `verified-results.json`을 만들었다. 개선 후보 실행은 정상 종료했다.

검증 상태 확인:

```bash
cd /workspace/deep
python scripts/check_learning_gates.py artifacts/learning-validation/20261109T055825812142Z --require-six-player
# 현재 예상 종료 코드: 2 (성능 gate 미통과)
```

새로운 결과 폴더에 개별 실험을 다시 실행하려면:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python scripts/validate_frozen_regret.py \
  --replay-seed 42 --output artifacts/validation-repeat/frozen-42

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python scripts/validate_kuhn_learning.py \
  --seed 71 --method stratified-conditional --output artifacts/validation-repeat/kuhn-71

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/.venvs/fullhouse/bin/python scripts/verify_kuhn_result.py \
  artifacts/validation-repeat/kuhn-71
```

현재 코드와 저장된 V3 replay가 필요하며, 출력 폴더의 기존 실험은 덮어쓰지 않는다.

이번 실험의 모델·Kuhn checkpoint·로그·실행 도구 snapshot 백업은 `artifacts/exports/learning-validation-20261109T055825812142Z.tar.gz`이며 약 18.1 MB다. [백업 metadata](../reports/learning-validation/20261109T055825812142Z/backup.json)에 SHA-256과 검증한 파일 수를 기록했다. 재학습 재현에는 원래 V3 replay가 든 기존 `fullhouse-v3-20261109T045233369268Z.tar.gz`도 필요하다. 두 archive는 로컬 파일이며 Git에는 결과·코드·해시만 보존한다.

## 5. 다음 구현의 초점

현재 6인 실패를 신경망 크기 부족 하나로 설명하거나, 장기 학습만 늘려 해결됐다고 가정하지 않는다. 다음 비교는 **regret 표적 수집의 분산·입력별 표본 수를 개선하고, 고정 양의 스케일 및 상태별 적합도를 관리하는 학습 방식**에 집중한다. 작은 게임의 통과 사례는 그 변경의 회귀 기준으로 유지한다.

희귀 입력을 자주 학습하는 것과 해당 입력의 정확한 표적을 더 많이 수집하는 것은 별개다. 후자의 개선은 올바른 표본 보정과 작은 게임 oracle을 함께 검증해야 한다. 6인 균형 수렴, 공유 신경망과 51개 요약 특징의 정보 손실은 여전히 별도 이론적 제한이다.
