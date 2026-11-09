# 실험 요약 스냅샷

결과 JSON과 검증에 필요한 압축 대전 로그를 `artifacts/`에서 보존한다. `fullhouse-medium/`에는 최초 완료 평가, `fullhouse-regression/20261109/`에는 추가 564,000핸드 원인 조사 결과와 원시 `.jsonl.gz`, 규칙 반례, fitting 진단, 파일별 SHA-256 manifest가 있다. 큰 가중치·학습 replay·체크포인트는 포함하지 않는다. 이전 작은 실험은 요약만 보존한 경우도 있다.

최신 해석과 다음 구현 순서는 [성능 후퇴 원인 조사](../docs/fullhouse-regression-diagnosis.md)를 따른다. `summary.json`의 paired 비교는 같은 deal seed와 상대 field에서 공통 deal 묶음을 사용한다. 개별 모델에 대한 ± 값의 단순 비교로 paired 오차를 대신하지 않는다.

`checkpoint-export.json`은 별도 재개 백업 파일의 위치/반복/용량/체크섬을 설명한다. 파일 자체는 GitHub에 올라가지 않는다. [작업 이어가기](../docs/continue-work.md)를 따른다.

이 폴더의 결과는 고정된 과거 기록이며 현재 살아 있는 학습의 상태가 아니다. 현재 환경에서는 `python scripts/train_fullhouse_medium.py status`를 사용한다.
