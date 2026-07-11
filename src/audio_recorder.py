"""Accumulate incoming audio and emit one feature row per fixed block."""

from __future__ import annotations

import threading
import queue
from collections.abc import Callable
from typing import Any

import numpy as np

from .audio_features import AudioFeatures, extract_audio_features, normalize_audio
from .sync_clock import SyncClock


class AudioRecorder:
    """Turn arbitrary input chunks into 0.1 second feature blocks."""

    def __init__(
        self,
        clock: SyncClock,
        block_duration_sec: float = 0.1,
        on_features: Callable[[dict], None] | None = None,
    ) -> None:
        self.clock = clock
        self.block_duration_sec = block_duration_sec
        self.on_features = on_features
        self._sample_rate: int | None = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._features: list[dict] = []
        self._stream_lock = threading.RLock()
        self._stream: Any | None = None
        self._stream_queue: queue.Queue | None = None
        self._stream_worker: threading.Thread | None = None
        self._stream_sample_rate = 0
        self._stream_generation = 0
        self._running = False
        self._closed = False
        self._last_error = ""
        self._on_stream_chunk: Callable[[np.ndarray, str], None] | None = None

    @property
    def last_error(self) -> str:
        with self._stream_lock:
            return self._last_error

    def is_running(self) -> bool:
        with self._stream_lock:
            return self._running and self._stream is not None

    def _set_error(self, error: str) -> None:
        with self._stream_lock:
            self._last_error = str(error)

    def start(
        self,
        stream_factory: Callable[[Callable], Any],
        sample_rate: int,
        on_chunk: Callable[[np.ndarray, str], None] | None = None,
    ) -> None:
        """Start one managed input stream with a lightweight native callback."""
        self.stop()
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("音频采样率必须大于 0")

        stream_queue: queue.Queue = queue.Queue(maxsize=32)
        with self._stream_lock:
            self._closed = False
            self._last_error = ""
            self._stream_sample_rate = sample_rate
            self._stream_generation += 1
            generation = self._stream_generation
            self._stream_queue = stream_queue
            self._on_stream_chunk = on_chunk

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            self._stream_callback(generation, indata, status)

        stream = None
        worker = threading.Thread(
            target=self._stream_worker_loop,
            args=(generation, stream_queue),
            daemon=True,
            name="audio-feature-worker",
        )
        try:
            stream = stream_factory(callback)
            with self._stream_lock:
                if generation != self._stream_generation:
                    raise RuntimeError("音频采集启动已被新的请求取代")
                self._stream = stream
                self._stream_worker = worker
                self._running = True
            worker.start()
            stream.start()
        except Exception as exc:
            self._set_error(str(exc))
            self.stop()
            raise

    def _stream_callback(self, generation: int, indata: Any, status: Any) -> None:
        """Copy and enqueue only; never perform FFT or UI/session work here."""
        try:
            with self._stream_lock:
                if (
                    not self._running
                    or self._closed
                    or generation != self._stream_generation
                    or self._stream_queue is None
                ):
                    return
                stream_queue = self._stream_queue
            chunk = np.asarray(indata, dtype=np.float32).copy()
            stream_queue.put_nowait((chunk, str(status) if status else ""))
        except queue.Full:
            self._set_error("音频处理队列已满，已丢弃一个输入块")
        except Exception as exc:
            self._set_error(f"音频 callback 异常：{exc}")

    def _stream_worker_loop(self, generation: int, stream_queue: queue.Queue) -> None:
        while True:
            item = stream_queue.get()
            if item is None:
                return
            chunk, status = item
            with self._stream_lock:
                if generation != self._stream_generation or self._closed:
                    continue
                sample_rate = self._stream_sample_rate
                on_chunk = self._on_stream_chunk
            try:
                if status:
                    self._set_error(status)
                self.add_samples(chunk, sample_rate)
                if on_chunk is not None:
                    on_chunk(chunk, status)
            except Exception as exc:
                self._set_error(f"音频数据处理异常：{exc}")

    def stop(self) -> None:
        """Idempotently stop and close the native stream outside state locks."""
        with self._stream_lock:
            self._running = False
            self._stream_generation += 1
            stream = self._stream
            worker = self._stream_worker
            stream_queue = self._stream_queue
            self._stream = None
            self._stream_worker = None
            self._stream_queue = None
            self._on_stream_chunk = None

        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self._set_error(f"停止音频流失败：{exc}")
            try:
                stream.close()
            except Exception as exc:
                self._set_error(f"关闭音频流失败：{exc}")
        if stream_queue is not None:
            try:
                stream_queue.put_nowait(None)
            except queue.Full:
                try:
                    stream_queue.get_nowait()
                    stream_queue.put_nowait(None)
                except (queue.Empty, queue.Full):
                    pass
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def close(self) -> None:
        self.stop()
        with self._stream_lock:
            self._closed = True

    def __enter__(self) -> "AudioRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def add_samples(
        self, samples: np.ndarray, sample_rate: int
    ) -> list[AudioFeatures]:
        mono = normalize_audio(samples)
        if mono.size == 0:
            return []

        emitted: list[AudioFeatures] = []
        callback_rows: list[dict] = []
        with self._lock:
            if self._sample_rate is not None and self._sample_rate != sample_rate:
                self._buffer = np.empty(0, dtype=np.float32)
            self._sample_rate = sample_rate
            self._buffer = np.concatenate((self._buffer, mono))
            block_size = max(1, int(round(sample_rate * self.block_duration_sec)))
            block_count = self._buffer.size // block_size
            remainder_size = self._buffer.size - block_count * block_size
            latest_block_timestamp = self.clock.now() - remainder_size / sample_rate
            first_block_timestamp = latest_block_timestamp - max(
                0, block_count - 1
            ) * self.block_duration_sec

            for block_index in range(block_count):
                block = self._buffer[:block_size]
                self._buffer = self._buffer[block_size:]
                feature = extract_audio_features(
                    block,
                    sample_rate=sample_rate,
                    timestamp=first_block_timestamp
                    + block_index * self.block_duration_sec,
                )
                row = feature.to_dict()
                self._features.append(row)
                emitted.append(feature)
                callback_rows.append(row)

        if self.on_features is not None:
            for row in callback_rows:
                self.on_features(row)

        return emitted

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [row.copy() for row in self._features]

    def reset(self) -> None:
        self.close()
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)
            self._features.clear()
