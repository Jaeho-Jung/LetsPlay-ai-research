"""
동시 요청 시뮬레이션 벤치마크.

4가지 전략의 동시성 성능을 비교합니다:
1. Baseline — 기존 WhisperService (순차 처리)
2. num_workers — CTranslate2 내장 동시성 활용
3. Async Queue — asyncio.Queue 기반 backpressure
4. Micro-batching — 요청 수집 후 batch 처리

사용법:
    # Mock 모드 (GPU 없이 테스트)
    python benchmark_concurrency.py --mock

    # 실제 GPU 모드
    python benchmark_concurrency.py --model-path ./whisper-small_child-ct2 --audio ./test.wav

    # 동시 요청 수 커스텀
    python benchmark_concurrency.py --mock --concurrency 1,5,10,20,50
"""

import argparse
import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock model for CPU/no-GPU testing
# ---------------------------------------------------------------------------

class MockTranscriptionResult:
    """Mock 전사 결과."""
    def __init__(self, text="", duration=3.0, inference_time=0.0, queue_wait_time=0.0):
        self.text = text
        self.duration = duration
        self.inference_time = inference_time
        self.queue_wait_time = queue_wait_time


class MockSegment:
    def __init__(self, text="테스트 전사 결과입니다."):
        self.text = text
        self.start = 0.0
        self.end = 3.0


class MockInfo:
    def __init__(self):
        self.language = "ko"
        self.language_probability = 0.99
        self.duration = 3.0


class MockWhisperModel:
    """GPU 없이도 벤치마크 로직을 테스트할 수 있는 Mock 모델."""

    def __init__(self, inference_delay: float = 0.5):
        self.inference_delay = inference_delay

    def transcribe(self, audio_path, **kwargs):
        time.sleep(self.inference_delay)
        return [MockSegment()], MockInfo()


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    strategy: str
    concurrency: int
    total_time: float
    latencies: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0

    @property
    def p95(self) -> float:
        if not self.latencies:
            return 0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99(self) -> float:
        if not self.latencies:
            return 0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def throughput(self) -> float:
        return len(self.latencies) / self.total_time if self.total_time > 0 else 0


# ---------------------------------------------------------------------------
# Benchmark strategies
# ---------------------------------------------------------------------------

async def benchmark_baseline(
    model, audio_path: str, n_requests: int
) -> BenchmarkResult:
    """Strategy 1: 순차 처리 (Baseline)."""
    latencies = []
    errors = 0

    start = time.monotonic()
    for _ in range(n_requests):
        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: model.transcribe(audio_path, language="ko")
            )
            latencies.append(time.monotonic() - t0)
        except Exception:
            errors += 1

    total = time.monotonic() - start
    return BenchmarkResult("Baseline (sequential)", n_requests, total, latencies, errors)


async def benchmark_num_workers(
    model, audio_path: str, n_requests: int
) -> BenchmarkResult:
    """Strategy 2: num_workers 튜닝 (동시 executor 호출)."""
    latencies = []
    errors = 0

    async def single_request():
        t0 = time.monotonic()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: model.transcribe(audio_path, language="ko")
            )
            return time.monotonic() - t0
        except Exception:
            return None

    start = time.monotonic()
    results = await asyncio.gather(*[single_request() for _ in range(n_requests)])
    total = time.monotonic() - start

    for r in results:
        if r is not None:
            latencies.append(r)
        else:
            errors += 1

    return BenchmarkResult("num_workers tuning", n_requests, total, latencies, errors)


async def benchmark_queued(
    model, audio_path: str, n_requests: int, queue_maxsize: int = 50
) -> BenchmarkResult:
    """Strategy 3: Async Queue."""
    from whisper_service_queue import QueuedWhisperService

    # QueuedWhisperService를 직접 생성하지 않고
    # 이미 로드된 모델로 내부를 구성
    service = QueuedWhisperService.__new__(QueuedWhisperService)
    service.device = "cpu"
    service.compute_type = "float32"
    service.model_path = ""
    service.num_workers = 1
    service.language = "ko"
    service.task = "transcribe"
    service.beam_size = 5
    service.queue_maxsize = queue_maxsize
    service.queue_timeout = 60.0
    service.total_processed = 0
    service.total_errors = 0
    service.total_rejected = 0
    service._model = model
    service._queue = asyncio.Queue(maxsize=queue_maxsize)
    service._running = True
    service._worker_task = asyncio.create_task(service._worker())

    latencies = []
    errors = 0

    async def single_request():
        t0 = time.monotonic()
        try:
            await service.transcribe(audio_path)
            return time.monotonic() - t0
        except Exception:
            return None

    start = time.monotonic()
    results = await asyncio.gather(*[single_request() for _ in range(n_requests)])
    total = time.monotonic() - start

    service._running = False
    service._worker_task.cancel()
    try:
        await service._worker_task
    except asyncio.CancelledError:
        pass

    for r in results:
        if r is not None:
            latencies.append(r)
        else:
            errors += 1

    return BenchmarkResult(
        "Async Queue", n_requests, total, latencies, errors
    )


async def benchmark_microbatch(
    model, audio_path: str, n_requests: int,
    batch_window_ms: float = 100.0, max_batch_size: int = 4,
) -> BenchmarkResult:
    """Strategy 4: Micro-batching."""
    from whisper_service_queue import MicroBatchWhisperService

    service = MicroBatchWhisperService.__new__(MicroBatchWhisperService)
    service.device = "cpu"
    service.compute_type = "float32"
    service.model_path = ""
    service.language = "ko"
    service.task = "transcribe"
    service.beam_size = 5
    service.queue_maxsize = max(n_requests, 50)
    service.queue_timeout = 120.0
    service.batch_window_ms = batch_window_ms
    service.max_batch_size = max_batch_size
    service.total_processed = 0
    service.total_batches = 0
    service.total_errors = 0
    service.total_rejected = 0
    service._model = model
    service._queue = asyncio.Queue(maxsize=service.queue_maxsize)
    service._running = True
    service._worker_task = asyncio.create_task(service._batch_worker())

    latencies = []
    errors = 0

    async def single_request():
        t0 = time.monotonic()
        try:
            await service.transcribe(audio_path)
            return time.monotonic() - t0
        except Exception:
            return None

    start = time.monotonic()
    results = await asyncio.gather(*[single_request() for _ in range(n_requests)])
    total = time.monotonic() - start

    service._running = False
    service._worker_task.cancel()
    try:
        await service._worker_task
    except asyncio.CancelledError:
        pass

    for r in results:
        if r is not None:
            latencies.append(r)
        else:
            errors += 1

    return BenchmarkResult(
        f"Micro-batch (w={batch_window_ms}ms, b={max_batch_size})",
        n_requests, total, latencies, errors
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(results: list[BenchmarkResult]):
    """벤치마크 결과를 표로 출력."""
    header = (
        f"{'Strategy':<42} {'N':>4} {'Total':>7} {'Avg':>7} "
        f"{'P50':>7} {'P95':>7} {'P99':>7} {'QPS':>7} {'Err':>4}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print(
        f"{'':42} {'':>4} {'(sec)':>7} {'(sec)':>7} "
        f"{'(sec)':>7} {'(sec)':>7} {'(sec)':>7} {'(r/s)':>7} {'':>4}"
    )
    print("-" * len(header))

    for r in results:
        print(
            f"{r.strategy:<42} {r.concurrency:>4} "
            f"{r.total_time:>7.2f} {r.avg_latency:>7.3f} "
            f"{r.p50:>7.3f} {r.p95:>7.3f} {r.p99:>7.3f} "
            f"{r.throughput:>7.2f} {r.errors:>4}"
        )

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(
    model,
    audio_path: str,
    concurrency_levels: list[int],
):
    """모든 전략에 대해 벤치마크 실행."""
    all_results: list[BenchmarkResult] = []

    for n in concurrency_levels:
        print(f"\n--- Concurrency: {n} ---")

        # Strategy 1: Baseline
        print(f"  [1/4] Baseline (sequential)...")
        r = await benchmark_baseline(model, audio_path, n)
        all_results.append(r)

        # Strategy 2: num_workers
        print(f"  [2/4] num_workers tuning...")
        r = await benchmark_num_workers(model, audio_path, n)
        all_results.append(r)

        # Strategy 3: Async Queue
        print(f"  [3/4] Async Queue...")
        r = await benchmark_queued(model, audio_path, n)
        all_results.append(r)

        # Strategy 4: Micro-batching
        print(f"  [4/4] Micro-batching...")
        r = await benchmark_microbatch(model, audio_path, n)
        all_results.append(r)

    print_results(all_results)

    # 전략별 요약
    print("\n📊 전략별 요약 (가장 높은 concurrency 기준):")
    max_n = max(concurrency_levels)
    final_results = [r for r in all_results if r.concurrency == max_n]
    for r in final_results:
        print(
            f"  {r.strategy:<42} "
            f"Total={r.total_time:.2f}s, "
            f"Throughput={r.throughput:.2f} req/s, "
            f"P95={r.p95:.3f}s, "
            f"Errors={r.errors}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Whisper 동시성 벤치마크"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="CTranslate2 모델 경로 (미지정 시 mock 모드)",
    )
    parser.add_argument(
        "--audio", type=str, default=None,
        help="테스트 오디오 파일 경로",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock 모드 (GPU 없이 테스트)",
    )
    parser.add_argument(
        "--mock-delay", type=float, default=0.5,
        help="Mock 추론 지연 시간 (초, 기본 0.5)",
    )
    parser.add_argument(
        "--concurrency", type=str, default="1,5,10,20",
        help="동시 요청 수 (콤마 구분, 기본 1,5,10,20)",
    )
    args = parser.parse_args()

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    if args.mock or args.model_path is None:
        print("🔧 Mock 모드로 실행 (GPU 불필요)")
        model = MockWhisperModel(inference_delay=args.mock_delay)
        audio_path = args.audio or "mock_audio.wav"
    else:
        from faster_whisper import WhisperModel
        print(f"🚀 실제 모델로 실행: {args.model_path}")
        model = WhisperModel(
            args.model_path,
            device="cuda",
            compute_type="float16",
            num_workers=2,
        )
        audio_path = args.audio

    print(f"📋 동시 요청 수: {concurrency_levels}")
    asyncio.run(run_benchmark(model, audio_path, concurrency_levels))


if __name__ == "__main__":
    main()
