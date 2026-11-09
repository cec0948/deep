# 학습기 검증 원시 결과

[결과 해석](../../../docs/learning-validation-results.md)을 먼저 읽으세요.

- `config.json`: 실험 전 고정한 기준.
- `gates.json`: 같은 방법이 두 검사를 통과했는지 계산한 판정. 현재 6인 진입 불가.
- `frozen-*/results.json.gz`: 원래 V3 모델 및 3가지 새 fitting 설정, 169개 입력별 표적과 출력. 압축 해제한 내용은 실행 폴더의 원본 JSON과 동일하다.
- `kuhn-*/verified-results.json`: checkpoint에서 재계산한 정책과 정확 best response.
- `kuhn-*/results.json`, `kuhn-*.log`: 최초 보고서와 실행 로그. budget 방식은 최종 JSON bool 직렬화 오류가 있었으며 별도 검증 보고서로 완료 상태를 복원함.
- `target-diagnostics.json`: 유한 표적의 민감도, UTG만 학습한 용량 진단. 배포 후보가 아님.
- `execution-sources/`: 실제 실행한 도구 버전과 수정된 검증 도구.
- `pytest.log`, `validation-tests-final.log`: 전체 31개 및 최종 관련 5개 검사 결과.

모델과 replay는 Git에 포함되지 않습니다. checkpoint는 `artifacts/learning-validation/20261109T055825812142Z`와 원래 V3 실행 폴더에 있습니다.
