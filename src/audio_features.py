"""Audio feature extraction for fixed-duration mono signal blocks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AudioFeatures:
    timestamp: float
    rms: float
    db: float
    dominant_frequency_hz: float
    spectral_centroid_hz: float
    zero_crossing_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """Convert integer or floating point audio samples to mono float32."""
    array = np.asarray(samples)
    if array.ndim > 1:
        channel_axis = 0 if array.shape[0] <= 8 else 1
        array = array.mean(axis=channel_axis)

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)

    return np.nan_to_num(array.reshape(-1), copy=False)


def extract_audio_features(
    samples: np.ndarray,
    sample_rate: int,
    timestamp: float,
    db_floor: float = -120.0,
) -> AudioFeatures:
    """Extract relative level and basic spectral features from one block."""
    signal = normalize_audio(samples)
    if signal.size == 0 or sample_rate <= 0:
        return AudioFeatures(timestamp, 0.0, db_floor, 0.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    db = max(db_floor, 20.0 * np.log10(max(rms, 10 ** (db_floor / 20.0))))

    centered = signal - float(np.mean(signal))
    windowed = centered * np.hanning(centered.size)
    magnitudes = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / sample_rate)

    if magnitudes.size > 1:
        magnitudes[0] = 0.0
    magnitude_sum = float(np.sum(magnitudes))
    if magnitude_sum > 0.0:
        dominant_frequency = float(frequencies[int(np.argmax(magnitudes))])
        spectral_centroid = float(np.sum(frequencies * magnitudes) / magnitude_sum)
    else:
        dominant_frequency = 0.0
        spectral_centroid = 0.0

    if centered.size > 1:
        signs = np.signbit(centered)
        zero_crossing_rate = float(np.count_nonzero(signs[1:] != signs[:-1])) / (
            centered.size - 1
        )
    else:
        zero_crossing_rate = 0.0

    return AudioFeatures(
        timestamp=float(timestamp),
        rms=rms,
        db=float(db),
        dominant_frequency_hz=dominant_frequency,
        spectral_centroid_hz=spectral_centroid,
        zero_crossing_rate=zero_crossing_rate,
    )

