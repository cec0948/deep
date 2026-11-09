# V3 새 버퍼 3-seed 학습 결과

실행 `20261109T034714104421Z`. 원시 **288,000핸드**의 좌석·deal·칩 보존·손익을 다시 계산해 확인했다.

단위: BB/100 ± 명목 95% deal-block 신뢰구간 반폭. 각 셀은 2,000 deals × 6좌석이다.

| 후보 | NoRegrets 1M 5명 상대 | 공개 Fullhouse 5명 상대 |
| --- | ---: | ---: |
| public | +88.82 ±42.75 | +21.10 ±24.81 |
| old-120 | -208.69 ±134.00 | -278.48 ±52.34 |
| old-600 | -284.27 ±167.02 | -328.13 ±53.39 |
| seed-42-average-60 | +41.63 ±135.80 | -204.24 ±49.97 |
| seed-42-average-120 | +91.18 ±138.24 | -179.62 ±53.71 |
| seed-42-advantage-120 | +264.63 ±91.92 | +22.60 ±20.67 |
| seed-43-average-60 | +120.98 ±136.36 | -202.85 ±54.75 |
| seed-43-average-120 | +15.60 ±147.90 | -227.64 ±55.99 |
| seed-43-advantage-120 | +225.69 ±78.39 | +1.37 ±15.92 |
| seed-44-average-60 | +96.93 ±121.67 | -135.84 ±47.36 |
| seed-44-average-120 | +73.80 ±141.81 | -180.14 ±50.80 |
| seed-44-advantage-120 | +104.38 ±70.54 | -3.77 ±15.64 |

## 같은 카드의 paired 차이

| 비교 | BB/100 차이 |
| --- | ---: |
| noregrets/seed-42-final minus public | +2.36 ±140.91 |
| noregrets/seed-42-final minus old-120 | +299.87 ±121.20 |
| noregrets/seed-42-final minus old-600 | +375.44 ±149.87 |
| noregrets/seed-42-final minus seed-42-average-60 | +49.55 ±88.87 |
| noregrets/seed-42-final minus seed-42-advantage-120 | -173.45 ±143.23 |
| noregrets/seed-42-advantage minus public (secondary) | +175.80 ±92.99 |
| noregrets/seed-43-final minus public | -73.22 ±151.35 |
| noregrets/seed-43-final minus old-120 | +224.30 ±110.41 |
| noregrets/seed-43-final minus old-600 | +299.87 ±135.39 |
| noregrets/seed-43-final minus seed-43-average-60 | -105.38 ±87.30 |
| noregrets/seed-43-final minus seed-43-advantage-120 | -210.09 ±143.73 |
| noregrets/seed-43-advantage minus public (secondary) | +136.87 ±82.89 |
| noregrets/seed-44-final minus public | -15.03 ±145.12 |
| noregrets/seed-44-final minus old-120 | +282.49 ±112.98 |
| noregrets/seed-44-final minus old-600 | +358.06 ±137.69 |
| noregrets/seed-44-final minus seed-44-average-60 | -23.13 ±105.07 |
| noregrets/seed-44-final minus seed-44-advantage-120 | -30.58 ±138.27 |
| noregrets/seed-44-advantage minus public (secondary) | +15.55 ±77.12 |
| public/seed-42-final minus public | -200.72 ±57.70 |
| public/seed-42-final minus old-120 | +98.86 ±53.97 |
| public/seed-42-final minus old-600 | +148.51 ±55.51 |
| public/seed-42-final minus seed-42-average-60 | +24.62 ±42.68 |
| public/seed-42-final minus seed-42-advantage-120 | -202.22 ±56.31 |
| public/seed-42-advantage minus public (secondary) | +1.50 ±30.39 |
| public/seed-43-final minus public | -248.74 ±60.11 |
| public/seed-43-final minus old-120 | +50.84 ±53.08 |
| public/seed-43-final minus old-600 | +100.49 ±55.94 |
| public/seed-43-final minus seed-43-average-60 | -24.79 ±41.81 |
| public/seed-43-final minus seed-43-advantage-120 | -229.01 ±57.88 |
| public/seed-43-advantage minus public (secondary) | -19.73 ±28.25 |
| public/seed-44-final minus public | -201.24 ±55.81 |
| public/seed-44-final minus old-120 | +98.34 ±50.09 |
| public/seed-44-final minus old-600 | +147.99 ±52.77 |
| public/seed-44-final minus seed-44-average-60 | -44.30 ±41.92 |
| public/seed-44-final minus seed-44-advantage-120 | -176.37 ±52.60 |
| public/seed-44-advantage minus public (secondary) | -24.87 ±28.35 |

세 seed 평균의 구간은 같은 deal 안에서 세 모델을 먼저 평균해 계산했다. 새로운 학습 seed의 불확실성까지 포괄하는 구간이 아니다. 원시 결과와 설정·해시·적합도는 `summary.json` 및 하위 폴더에 보존한다.

기본 모델은 자동 승격하지 않았다. 여러 비교는 탐색적이며, 고정 상대의 성적을 프로급이나 균형 수렴으로 해석하지 않는다.
