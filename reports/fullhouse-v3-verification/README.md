# V3 구현 검증 증거

2026-11-09. 고정 6인·전원 100BB 범위의 엔진 수정과 평균 전략 fitting 검증이다. 실력 평가는 별도 새 버퍼 캠페인의 결과를 따른다.

- `rules-2000.json`: 참조 Rust 엔진과 새 2,000핸드 및 과거 86개 반례, 총 38,316상태를 비교했다. terminal 손익·51개 특징·deal marker가 없는 이력도 검사했다. 이 검사 이후 SWA optimizer만 추가했고 규칙/특징 소스는 같다.
- `final-verification.json`: 최종 SWA 버전에서 조건부 평균 oracle·RNG 재생·6+6 재개=연속 12회=같은 seed 재실행·V2 재개 거절을 확인했다. 해당 실행의 규칙 표본은 10핸드+86개 반례다.
- `strategy-fitting.json`: 과거 replay에서 seed 7·8·9 각각 128/512 steps를 초기화부터 독립 fitting했다. 최종 512-step 설정이 세 seed 모두 정해둔 기준을 통과했다. 이 버퍼/모델은 새 자기대국에 사용하지 않는다.
- `plain-pps-diagnostic.log`: SWA를 추가하기 전의 실패도 보존한다. 첫 PPS 실험에서 한 seed가 기준을 초과했으며, 이를 계기로 SWA를 추가했다.
- `pytest.log`: 기존 검사와 당시 V3 검사를 합한 25개 통과. 이후 추가한 독립 사이드팟/runout 검사는 `pytest-layered.log`의 1개 통과다. 추가 후 전체 suite를 다시 실행한 것으로 세지 않는다.
- `provenance.json`: 실제 학습에 고정한 소스 SHA-256과 검증 스크립트 해시. 원시 평가 로그에 같은 deal/좌석을 중복 삽입했을 때 검산기가 거절하는 부정 검사도 기록했다.
- `restore-live.json`: 진행 중 백업을 임시 디렉터리에 풀어 세 seed의 모델·RNG·설정이 해당 원본 iteration과 일치하고, 버퍼가 있으며, native engine의 실행 권한과 JSON reset이 정상임을 확인했다. 임시 위치에서 실제 재학습은 하지 않았다.
- `restore-final.json`: 최종 백업의 세 120회 전체 checkpoint, 세 60회 snapshot, 완료 상태·24개 평가 기록, 복원한 native engine을 검사했다. exporter는 압축 안의 일반 파일 215개를 모두 SHA-256으로 확인했다.

AA의 자기 버퍼 평균을 잘 재현한다는 사실은 해당 표적이 최적 포커 전략이라는 뜻이 아니다. 상태 그룹 재가중은 유한한 공유 신경망의 투영 목적함수를 바꾼다. collector의 가중치 쏠림이나 6인 CFR 수렴 보장은 이 검사로 해결하지 않았다.
