"""Accumulate incoming audio and emit one feature row per fixed block."""

from __future__ import annotations

import threading
from collections.abc import Callable

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
        with self._lock:
            self._buffer = np.empty(0, dtype=np.float32)
            self._features.clear()
