# 실험 요약 스냅샷

Git에 포함할 수 있는 작은 결과 JSON을 `artifacts/`에서 복사했다. 이름과 하위 경로는 원래 실행 ID를 보존한다. 원시 핸드·가중치·체크포인트는 이 폴더에 포함하지 않는다. 따라서 과거 대전 통계의 원시 재계산에는 기존 artifacts 또는 같은 명령의 재실행이 필요하다.

`checkpoint-export.json`은 별도 재개 백업 파일의 위치/반복/용량/체크섬을 설명한다. 파일 자체는 GitHub에 올라가지 않는다. [작업 이어가기](../docs/continue-work.md)를 따른다.

이 폴더의 결과는 고정된 과거 기록이며 현재 살아 있는 학습의 상태가 아니다. 현재 환경에서는 `python scripts/train_fullhouse_medium.py status`를 사용한다.
