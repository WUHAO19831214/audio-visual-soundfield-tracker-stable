import numpy as np

from src.audio_features import extract_audio_features
from src.audio_recorder import AudioRecorder
from src.sync_clock import SyncClock


def test_extract_audio_features_for_sine_wave() -> None:
    sample_rate = 16_000
    frequency = 440.0
    duration = 0.1
    time_axis = np.arange(int(sample_rate * duration)) / sample_rate
    signal = 0.5 * np.sin(2.0 * np.pi * frequency * time_axis)

    features = extract_audio_features(signal, sample_rate, timestamp=10.0)

    assert features.timestamp == 10.0
    assert abs(features.dominant_frequency_hz - frequency) <= 10.0
    assert 400.0 < features.spectral_centroid_hz < 500.0
    assert -10.0 < features.db < -8.0
    assert 0.04 < features.zero_crossing_rate < 0.07


def test_audio_recorder_emits_tenth_second_blocks() -> None:
    recorder = AudioRecorder(clock=SyncClock(), block_duration_sec=0.1)
    samples = np.zeros(3_200, dtype=np.float32)

    emitted = recorder.add_samples(samples, sample_rate=16_000)

    assert len(emitted) == 2
    assert len(recorder.snapshot()) == 2
    assert abs(emitted[1].timestamp - emitted[0].timestamp - 0.1) < 1e-6
