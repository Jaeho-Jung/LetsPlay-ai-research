# 동시성 최적화 벤치마크 분석

> **환경**: Google Colab T4 GPU (15GB VRAM) · `faster-whisper` (CTranslate2 float16) · `elmenwol/whisper-small_aihub_child`

---

## 핵심 결과 요약

| 전략 | N=1 QPS | N=50 QPS | N=50 P95 | 총평 |
|------|---------|----------|----------|------|
| **Baseline** (순차) | 2.56 | **5.28** | **0.259s** | ✅ 가장 낮은 지연시간 |
| **num_workers** (CTranslate2) | 1.23 | **6.62** | 7.394s | ✅ 가장 높은 처리량 |
| **Async Queue** | 0.40 | 2.18 | 22.016s | ⚠️ 안정성↑ 속도↓ |
| **Micro-batch** | 0.80 | 1.41 | 32.990s | ❌ 단일 GPU에서 비효율 |

---

## 1. Baseline이 가장 빠른 이유

```
N=50: Total=9.48s, QPS=5.28, P95=0.259s
```

Baseline은 `asyncio.gather` + `run_in_executor`로 **모든 요청을 동시에 스레드풀로 보냅니다.** CTranslate2는 내부적으로 GPU 커널 스케줄링을 최적화하므로, 외부 큐 없이도 효율적으로 처리합니다.

주목할 점:
- 단건(N=1) 0.39s → N=50 평균 0.19s로 **per-request 시간이 절반으로 감소**
- GPU 커널 캐시, 배치 스케줄링 효과로 동시 요청이 많을수록 개별 속도 향상

---

## 2. num_workers — 최고 처리량, 최악 꼬리 지연

```
N=50: Total=7.55s, QPS=6.62, P95=7.394s
```

| 지표 | 값 | vs Baseline |
|------|-----|------------|
| Total time | 7.55s | **20% 단축** |
| Throughput | 6.62 QPS | **25% 향상** |
| P95 latency | 7.394s | **28배 악화** |

CTranslate2의 `num_workers=2`가 인코더/디코더 파이프라이닝으로 처리량을 높이지만, GPU contention으로 인해 **일부 요청이 7초 이상 대기**합니다. 평균은 빠르지만 꼬리 지연이 심각합니다.

> [!WARNING]
> P95=7.4s는 실시간 서비스에서 허용 불가. 처리량만 보면 최선이지만 SLA 위반 위험.

---

## 3. Async Queue — 안정적이지만 느림

```
N=50: Total=22.88s, QPS=2.18, P95=22.016s
```

Worker 1개가 순차 처리하므로 `50 × ~0.45s ≈ 22.5s`로 예측과 정확히 일치합니다.

**장점**: 서버 crash 없음, GPU OOM 불가, backpressure 확보
**단점**: 단일 GPU에서 처리량이 Baseline의 **41% 수준**

> Queue의 가치는 속도가 아니라 **운영 안정성**. N=500, N=1000 상황에서 서버가 죽지 않는 것이 목적.

---

## 4. Micro-batching — 단일 GPU에서 역효과

```
N=50: Total=35.37s, QPS=1.41, P95=32.990s
```

**4가지 전략 중 가장 느림.** 원인:
1. `batch_window_ms=100ms` 대기 시간이 매 배치마다 추가
2. `faster-whisper` API가 **단일 파일 transcribe만 지원** → 배치 내에서도 순차 처리
3. 실질적 batch 효과 없이 윈도우 오버헤드만 누적

> [!CAUTION]
> Micro-batching은 batch 추론 API가 있는 엔진(PyTorch `model.generate()` 등)에서만 유효. `faster-whisper` 단일 GPU에서는 사용 금지.

---

## 5. 동시성 증가에 따른 처리량 추이

```
N     Baseline    num_workers    Queue    Micro-batch
1       2.56         1.23        0.40       0.80
5       2.30         2.86        0.90       0.73
10      4.32         6.83        2.03       2.29
20      5.82         5.46        2.29       2.19
50      5.28         6.62        2.18       1.41
```

- **Baseline**: N=20에서 5.82 QPS로 포화, 이후 감소 (GPU 한계)
- **num_workers**: N=10에서 6.83 QPS 피크, 이후 감소
- **Queue/Micro-batch**: N에 관계없이 2 QPS 미만으로 수렴 (직렬화 병목)

---

## 6. 최종 결론 및 배포 전략

### 단일 GPU 환경 (Cloud Run, T4 1장)

```
✅ 권장: Baseline (순차 처리) + 요청 수 제한 (rate limiting)
```

- `faster-whisper`의 CTranslate2가 이미 내부 최적화를 수행
- 외부 Queue/Batch는 오버헤드만 추가
- **Rate limiting** (예: max 20 concurrent)으로 GPU 포화 방지

### 고가용성 서비스 (N > 100)

```
✅ 권장: Async Queue + Worker 수 = GPU 수
```

- 안정성 확보가 속도보다 중요
- `QueueFullError` → HTTP 429로 매핑하여 graceful degradation
- 수평 확장 시 Redis Queue로 전환

### 다중 GPU 환경

```
✅ 권장: num_workers=GPU당 1 + Redis Queue로 부하 분산
```

- GPU별 Worker 프로세스 분리
- 처리량 선형 확장 가능

---

## 실험 한계 및 후속 과제

1. **Batch API 부재**: `faster-whisper`가 batch transcribe를 지원하지 않아 Micro-batching의 실효성을 검증하지 못함. PyTorch 직접 추론(`model.generate(batch)`)으로 재실험 필요.
2. **OOM 시뮬레이션 미수행**: Queue의 핵심 가치인 crash 방지를 검증하려면 N=500+ 테스트 필요.
3. **네트워크 I/O 미반영**: 실제 서비스에서는 오디오 업로드/다운로드 시간이 추가되므로, Queue 대기 중 I/O 오버랩 효과 측정 필요.
