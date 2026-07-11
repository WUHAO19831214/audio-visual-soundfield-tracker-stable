import time

import numpy as np

from src.audio_recorder import AudioRecorder
from src.sync_clock import SyncClock


class FakeStream:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def emit(self, samples) -> None:
        frames = len(samples) if hasattr(samples, "__len__") else 0
        self.callback(samples, frames, None, None)


def make_factory(streams: list[FakeStream]):
    def factory(callback):
        stream = FakeStream(callback)
        streams.append(stream)
        return stream

    return factory


def test_stop_and_close_are_idempotent() -> None:
    streams: list[FakeStream] = []
    recorder = AudioRecorder(clock=SyncClock())
    recorder.start(make_factory(streams), sample_rate=16_000)

    recorder.stop()
    recorder.stop()
    recorder.close()
    recorder.close()

    assert streams[0].stopped is True
    assert streams[0].closed is True
    assert recorder.is_running() is False


def test_start_closes_existing_stream_first() -> None:
    streams: list[FakeStream] = []
    recorder = AudioRecorder(clock=SyncClock())
    factory = make_factory(streams)

    recorder.start(factory, sample_rate=16_000)
    recorder.start(factory, sample_rate=16_000)

    assert streams[0].stopped is True
    assert streams[0].closed is True
    assert streams[1].started is True
    recorder.close()


def test_callback_exception_is_contained() -> None:
    streams: list[FakeStream] = []
    recorder = AudioRecorder(clock=SyncClock())
    recorder.start(make_factory(streams), sample_rate=16_000)

    streams[0].emit(object())

    assert "callback" in recorder.last_error
    recorder.close()


def test_processing_exception_is_recorded_not_raised() -> None:
    streams: list[FakeStream] = []

    def broken_features(row) -> None:
        raise RuntimeError("consumer failed")

    recorder = AudioRecorder(clock=SyncClock(), on_features=broken_features)
    recorder.start(make_factory(streams), sample_rate=16_000)
    streams[0].emit(np.ones((1600, 1), dtype=np.float32))

    deadline = time.monotonic() + 1.0
    while "consumer failed" not in recorder.last_error and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "consumer failed" in recorder.last_error
    recorder.close()
