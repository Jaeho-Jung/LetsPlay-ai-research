# 동시성 처리 방식 벤치마크 기록

> 환경: Google Colab T4 GPU, faster-whisper/CTranslate2 FP16, 아동 음성 Whisper-small 모델. 아래 수치는 기록된 동일 오디오 경로의 요청 처리 실험이며, 서버 네트워크 I/O와 모델 로딩을 포함한 종단 간 서비스 성능이 아닙니다. 오디오 길이와 독립적인 워밍업 횟수는 기록되지 않았습니다.

| 전략 | 50건 처리량(QPS) | P95(초) |
|---|---:|---:|
| Baseline: 순차 실행 | 5.28 | 0.259 |
| num_workers: 동시 executor 호출 | 6.62 | 7.394 |
| Async Queue | 2.18 | 22.016 |
| Micro-batch 구현 | 1.41 | 32.990 |

수치는 기존 실험 기록입니다. 재현에는 동일 모델·입력·생성 옵션 및 측정 조건의 확인이 필요합니다.

## 코드와 측정값

- `benchmark_baseline()`은 `run_in_executor`로 보낸 요청이 끝나면 다음 요청을 시작합니다. 따라서 Baseline은 순차 실행입니다. `benchmark_num_workers()`와 큐 방식은 `asyncio.gather`로 여러 요청을 시작합니다.
- num_workers 방식은 처리량이 더 높았고 P95도 길었습니다. 요청 대기 시간과 실행 시간을 따로 측정하지 않아 원인을 GPU contention이나 내부 파이프라이닝으로 특정할 수 없습니다.
- 이 구현에서는 Async Queue와 Micro-batch의 처리량이 낮고 지연이 길었습니다. 고부하 장애 실험은 하지 않았으므로 큐의 서버 장애·OOM 방지 효과는 확인되지 않았습니다. Micro-batch 구현도 모델을 순차 호출합니다. 다른 배치 추론 방식에 같은 결과를 적용할 수는 없습니다.
- Baseline의 낮은 P95와 num_workers의 높은 QPS는 서로 다른 요청 시작 패턴에서 관찰된 값입니다. 실제 동시 요청 서비스의 우열이나 SLA를 이 표만으로 결정할 수 없습니다.
- Rate Limiting은 운영 시 고려한 전략이며 이 실험에서 효과를 따로 측정하지 않았습니다. Redis 큐, 다중 GPU 확장 및 선형 확장도 구현·검증 결과가 아니라 후속 검토안입니다.

## 재검증 항목

오디오 길이, 모델·생성 옵션, 워밍업, 반복 횟수, 동시 요청의 시작 조건을 고정하고 모델 로딩·전처리·네트워크 시간을 구분해 측정해야 합니다. 성공·오류 건수와 GPU 메모리도 함께 기록해야 실제 서빙 전략을 비교할 수 있습니다.

실험 구현: [benchmark_concurrency.py](benchmark_concurrency.py)
