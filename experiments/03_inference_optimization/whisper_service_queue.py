"""
Async Queue + Micro-batching 기반 WhisperService.

동시 요청이 많은 환경에서 안정성과 처리량을 최적화하기 위한
3가지 전략을 구현합니다:

1. QueuedWhisperService — asyncio.Queue로 backpressure 확보
2. MicroBatchWhisperService — 요청을 모아 batch 추론으로 throughput 향상
3. WhisperServiceFactory — 전략별 서비스 인스턴스 생성

사용 전 CTranslate2 모델 변환 필요:
    ct2-whisper-converter --model whisper-small_child \
        --output_dir whisper-small_child-ct2 --quantization float16
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QueueFullError(Exception):
    """큐가 가득 찼을 때 발생 (HTTP 429 매핑용)."""
    pass


class QueueTimeoutError(Exception):
    """큐 대기 시간 초과 시 발생 (HTTP 408 매핑용)."""
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionTask:
    """추론 큐에 넣을 작업 단위."""
    audio_path: str
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.monotonic)


@dataclass
class TranscriptionResult:
    """추론 결과."""
    text: str
    duration: float           # 오디오 길이 (초)
    inference_time: float     # 추론 소요시간 (초)
    queue_wait_time: float    # 큐 대기시간 (초)


# ---------------------------------------------------------------------------
# Strategy 1: Baseline (기존 WhisperService — 참조용)
# ---------------------------------------------------------------------------
# → whisper_service.py의 WhisperService를 그대로 사용


# ---------------------------------------------------------------------------
# Strategy 2: num_workers 튜닝
# ---------------------------------------------------------------------------

def create_tuned_model(
    model_path: str,
    device: str = "cuda",
    compute_type: str = "float16",
    num_workers: int = 2,
    cpu_threads: int = 4,
) -> WhisperModel:
    """
    CTranslate2 num_workers를 활용한 모델 생성.

    num_workers > 1이면 CTranslate2 내부적으로 요청 큐잉을 수행합니다.
    추가 인프라 없이 동시성을 확보하는 가장 저비용 방법입니다.

    Args:
        model_path: CTranslate2 변환된 모델 경로.
        device: 'cuda' 또는 'cpu'.
        compute_type: 연산 정밀도.
        num_workers: CTranslate2 동시 추론 워커 수.
        cpu_threads: CPU 전처리 스레드 수.

    Returns:
        WhisperModel: 튜닝된 모델 인스턴스.
    """
    model = WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        num_workers=num_workers,
        cpu_threads=cpu_threads,
    )
    logger.info(
        f"Tuned model created: num_workers={num_workers}, "
        f"cpu_threads={cpu_threads}"
    )
    return model


# ---------------------------------------------------------------------------
# Strategy 3: Async Queue
# ---------------------------------------------------------------------------

class QueuedWhisperService:
    """
    asyncio.Queue 기반 Whisper 서비스.

    Backpressure를 제공하여 동시 요청 폭주 시 서버 안정성을 확보합니다.
    큐가 가득 차면 QueueFullError를 발생시켜 HTTP 429 응답에 매핑할 수 있습니다.

    Example:
        service = QueuedWhisperService(model_path="whisper-ct2")
        await service.start()

        result = await service.transcribe("audio.wav")
        print(result.text)

        await service.stop()
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        compute_type: str = "float16",
        num_workers: int = 1,
        queue_maxsize: int = 50,
        queue_timeout: float = 30.0,
        language: str = "ko",
        task: str = "transcribe",
        beam_size: int = 5,
    ):
        """
        Args:
            model_path: CTranslate2 변환된 모델 경로.
            device: 디바이스 ('auto', 'cuda', 'cpu').
            compute_type: 연산 정밀도.
            num_workers: CTranslate2 동시 워커 수.
            queue_maxsize: 큐 최대 크기 (초과 시 QueueFullError).
            queue_timeout: 큐 대기 최대 시간 (초, 초과 시 QueueTimeoutError).
            language: 음성 인식 언어 코드.
            task: 'transcribe' 또는 'translate'.
            beam_size: 빔 서치 크기.
        """
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.compute_type = compute_type if self.device == "cuda" else "float32"
        self.model_path = model_path
        self.num_workers = num_workers
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.queue_maxsize = queue_maxsize
        self.queue_timeout = queue_timeout

        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._model: Optional[WhisperModel] = None
        self._running = False

        # 메트릭
        self.total_processed = 0
        self.total_errors = 0
        self.total_rejected = 0

    async def start(self):
        """서비스 시작: 모델 로드 + 워커 태스크 생성."""
        if self._running:
            return

        logger.info("Starting QueuedWhisperService...")
        self._model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=self.num_workers,
        )
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info(
            f"QueuedWhisperService started: "
            f"maxsize={self.queue_maxsize}, timeout={self.queue_timeout}s"
        )

    async def stop(self):
        """서비스 종료: 큐 소진 후 워커 태스크 취소."""
        if not self._running:
            return

        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"QueuedWhisperService stopped. "
            f"Processed={self.total_processed}, "
            f"Errors={self.total_errors}, "
            f"Rejected={self.total_rejected}"
        )

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        오디오 전사 요청을 큐에 추가하고 결과를 대기합니다.

        Args:
            audio_path: 오디오 파일 경로.

        Returns:
            TranscriptionResult: 전사 결과.

        Raises:
            QueueFullError: 큐가 가득 찬 경우.
            QueueTimeoutError: 대기 시간 초과.
        """
        if not self._running:
            raise RuntimeError("Service is not running. Call start() first.")

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        task = TranscriptionTask(audio_path=audio_path, future=future)

        # 큐에 넣기 (non-blocking)
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            self.total_rejected += 1
            raise QueueFullError(
                f"Queue is full ({self.queue_maxsize}). "
                f"Try again later."
            )

        # 결과 대기 (timeout 적용)
        try:
            result = await asyncio.wait_for(future, timeout=self.queue_timeout)
            return result
        except asyncio.TimeoutError:
            self.total_rejected += 1
            raise QueueTimeoutError(
                f"Transcription timed out after {self.queue_timeout}s"
            )

    async def _worker(self):
        """백그라운드 워커: 큐에서 작업을 꺼내 순차 처리."""
        logger.info("Worker started")
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            start_time = time.monotonic()
            queue_wait = start_time - task.enqueue_time

            try:
                # 추론은 블로킹이므로 executor에서 실행
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_inference,
                    task.audio_path,
                )
                inference_time = time.monotonic() - start_time
                result.queue_wait_time = queue_wait
                result.inference_time = inference_time

                if not task.future.cancelled():
                    task.future.set_result(result)
                self.total_processed += 1

            except Exception as e:
                self.total_errors += 1
                if not task.future.cancelled():
                    task.future.set_exception(e)
                logger.error(f"Worker inference error: {e}")

            finally:
                self._queue.task_done()

    def _run_inference(self, audio_path: str) -> TranscriptionResult:
        """동기 추론 실행."""
        segments, info = self._model.transcribe(
            audio_path,
            language=self.language,
            task=self.task,
            beam_size=self.beam_size,
        )
        text = "".join(segment.text for segment in segments)
        return TranscriptionResult(
            text=text,
            duration=info.duration,
            inference_time=0.0,    # 호출자가 설정
            queue_wait_time=0.0,   # 호출자가 설정
        )

    @property
    def queue_size(self) -> int:
        """현재 큐에 대기 중인 작업 수."""
        return self._queue.qsize() if self._queue else 0

    def get_metrics(self) -> dict:
        """서비스 메트릭 반환."""
        return {
            "queue_size": self.queue_size,
            "queue_maxsize": self.queue_maxsize,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "total_rejected": self.total_rejected,
            "running": self._running,
        }


# ---------------------------------------------------------------------------
# Strategy 4: Micro-batching
# ---------------------------------------------------------------------------

class MicroBatchWhisperService:
    """
    Micro-batching 기반 Whisper 서비스.

    일정 시간 윈도우 동안 요청을 모아 batch로 처리하여
    GPU 활용률과 throughput을 높입니다.

    Example:
        service = MicroBatchWhisperService(model_path="whisper-ct2")
        await service.start()

        result = await service.transcribe("audio.wav")
        print(result.text)

        await service.stop()
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        compute_type: str = "float16",
        queue_maxsize: int = 50,
        queue_timeout: float = 30.0,
        batch_window_ms: float = 100.0,
        max_batch_size: int = 4,
        language: str = "ko",
        task: str = "transcribe",
        beam_size: int = 5,
    ):
        """
        Args:
            model_path: CTranslate2 변환된 모델 경로.
            device: 디바이스.
            compute_type: 연산 정밀도.
            queue_maxsize: 큐 최대 크기.
            queue_timeout: 큐 대기 최대 시간 (초).
            batch_window_ms: 배치 수집 윈도우 (밀리초).
            max_batch_size: 최대 배치 크기.
            language: 음성 인식 언어 코드.
            task: 'transcribe' 또는 'translate'.
            beam_size: 빔 서치 크기.
        """
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self.compute_type = compute_type if self.device == "cuda" else "float32"
        self.model_path = model_path
        self.language = language
        self.task = task
        self.beam_size = beam_size
        self.queue_maxsize = queue_maxsize
        self.queue_timeout = queue_timeout
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size

        self._queue: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._model: Optional[WhisperModel] = None
        self._running = False

        # 메트릭
        self.total_processed = 0
        self.total_batches = 0
        self.total_errors = 0
        self.total_rejected = 0

    async def start(self):
        """서비스 시작."""
        if self._running:
            return

        logger.info("Starting MicroBatchWhisperService...")
        self._model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._queue = asyncio.Queue(maxsize=self.queue_maxsize)
        self._running = True
        self._worker_task = asyncio.create_task(self._batch_worker())
        logger.info(
            f"MicroBatchWhisperService started: "
            f"window={self.batch_window_ms}ms, "
            f"max_batch={self.max_batch_size}"
        )

    async def stop(self):
        """서비스 종료."""
        if not self._running:
            return

        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"MicroBatchWhisperService stopped. "
            f"Processed={self.total_processed}, "
            f"Batches={self.total_batches}, "
            f"Errors={self.total_errors}, "
            f"Rejected={self.total_rejected}"
        )

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        오디오 전사 요청을 큐에 추가하고 결과를 대기합니다.

        Args:
            audio_path: 오디오 파일 경로.

        Returns:
            TranscriptionResult: 전사 결과.

        Raises:
            QueueFullError: 큐가 가득 찬 경우.
            QueueTimeoutError: 대기 시간 초과.
        """
        if not self._running:
            raise RuntimeError("Service is not running. Call start() first.")

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        task = TranscriptionTask(audio_path=audio_path, future=future)

        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            self.total_rejected += 1
            raise QueueFullError(
                f"Queue is full ({self.queue_maxsize}). Try again later."
            )

        try:
            result = await asyncio.wait_for(future, timeout=self.queue_timeout)
            return result
        except asyncio.TimeoutError:
            self.total_rejected += 1
            raise QueueTimeoutError(
                f"Transcription timed out after {self.queue_timeout}s"
            )

    async def _batch_worker(self):
        """
        배치 워커: 윈도우 동안 요청을 모아 일괄 처리.

        수집 로직:
        1. 첫 번째 요청이 올 때까지 대기
        2. 첫 요청 도착 후 batch_window_ms 동안 추가 요청 수집
        3. max_batch_size에 도달하면 즉시 처리
        """
        logger.info("Batch worker started")
        while self._running:
            batch: list[TranscriptionTask] = []

            # 첫 번째 요청 대기
            try:
                first_task = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                batch.append(first_task)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # 윈도우 동안 추가 요청 수집
            window_deadline = time.monotonic() + (self.batch_window_ms / 1000.0)
            while (
                len(batch) < self.max_batch_size
                and time.monotonic() < window_deadline
            ):
                remaining = window_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(task)
                except asyncio.TimeoutError:
                    break

            # 배치 처리
            logger.debug(f"Processing batch of {len(batch)} tasks")
            self.total_batches += 1

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._run_batch_inference,
                batch,
            )

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    self.total_errors += 1
                    if not task.future.cancelled():
                        task.future.set_exception(result)
                else:
                    self.total_processed += 1
                    if not task.future.cancelled():
                        task.future.set_result(result)
                self._queue.task_done()

    def _run_batch_inference(
        self, batch: list[TranscriptionTask]
    ) -> list[TranscriptionResult | Exception]:
        """
        배치 내 각 작업을 순차 추론.

        Note: Whisper의 encoder는 batch 가능하지만, faster-whisper API는
        현재 단일 파일 transcribe만 지원. 따라서 배치 내에서는 순차 처리하되,
        배치 수집 자체가 queue contention을 줄여 전체 안정성을 높입니다.
        향후 faster-whisper가 batch API를 지원하면 여기를 교체할 수 있습니다.
        """
        results: list[TranscriptionResult | Exception] = []

        for task in batch:
            start_time = time.monotonic()
            queue_wait = start_time - task.enqueue_time

            try:
                segments, info = self._model.transcribe(
                    task.audio_path,
                    language=self.language,
                    task=self.task,
                    beam_size=self.beam_size,
                )
                text = "".join(segment.text for segment in segments)
                inference_time = time.monotonic() - start_time

                results.append(TranscriptionResult(
                    text=text,
                    duration=info.duration,
                    inference_time=inference_time,
                    queue_wait_time=queue_wait,
                ))
            except Exception as e:
                logger.error(f"Batch inference error for {task.audio_path}: {e}")
                results.append(e)

        return results

    @property
    def queue_size(self) -> int:
        """현재 큐 대기 수."""
        return self._queue.qsize() if self._queue else 0

    def get_metrics(self) -> dict:
        """서비스 메트릭 반환."""
        return {
            "queue_size": self.queue_size,
            "queue_maxsize": self.queue_maxsize,
            "total_processed": self.total_processed,
            "total_batches": self.total_batches,
            "total_errors": self.total_errors,
            "total_rejected": self.total_rejected,
            "running": self._running,
            "batch_window_ms": self.batch_window_ms,
            "max_batch_size": self.max_batch_size,
        }
