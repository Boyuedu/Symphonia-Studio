"""
Audio feature extraction for the Emotional Music Landscape.
Produces dense, time-varying features (loudness, spectral centroid, spectral flux,
harmonic energy, onset strength) using small analysis windows for high temporal resolution.
Works with any duration; configurable window and hop (general, not piece-specific).
"""

import os
from typing import Any, Optional

# Project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_features_from_audio(
    audio_path: str,
    duration_seconds: Optional[float] = None,
    sr_target: int = 22050,
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    window_sec: float = 0.4,
    hop_sec: float = 0.25,
) -> Optional[dict[str, Any]]:
    """
    Extract time-varying audio features with high temporal resolution.
    Uses small windows (e.g. 0.3–0.5 s) and hop (~0.25 s) for a dense sequence.

    Returns dict with:
      - times: list of float (time in seconds per frame)
      - loudness: normalized RMS energy [0–1]
      - spectral_centroid: normalized brightness [0–1]
      - spectral_flux: normalized spectral change [0–1]
      - onset_strength: normalized [0–1]
      - harmonic_energy: normalized [0–1] if available, else zeros
    Returns None if file not found or librosa unavailable.
    """
    if not os.path.isfile(audio_path):
        return None
    try:
        import numpy as np
        import librosa
    except ImportError:
        return None
    try:
        # Derive frame parameters from seconds (general for any sr and duration)
        hop_length = hop_length or max(1, int(hop_sec * sr_target))
        n_fft = n_fft or max(256, int(window_sec * sr_target))
        n_fft = min(n_fft, 2048)

        y, sr = librosa.load(audio_path, sr=sr_target, mono=True, duration=duration_seconds)
        if len(y) == 0:
            return None

        # Loudness / RMS energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        # Spectral centroid (brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        # Spectral flux: frame-to-frame difference in magnitude spectrum
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=0) ** 2, axis=0))
        flux = np.concatenate([[flux[0]], flux])  # same length as other features

        # Band energy: sum magnitude per frequency band (for orchestra explorer: match sound to instrument range)
        # Bands (Hz): low 0-250, mid_low 250-500, mid 500-2000, mid_high 2000-4000, high 4000+
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        nyq = sr / 2.0
        band_defs = [
            ("low", 0, 250),
            ("mid_low", 250, 500),
            ("mid", 500, 2000),
            ("mid_high", 2000, 4000),
            ("high", 4000, nyq + 1),
        ]
        band_energy = {}
        for name, f_lo, f_hi in band_defs:
            mask = (freqs >= f_lo) & (freqs < f_hi)
            band_energy[name] = np.sum(S[mask, :], axis=0)
        # Normalize each band to [0, 1] over time
        for name in band_energy:
            x = band_energy[name]
            mn, mx = x.min(), x.max()
            band_energy[name] = (x - mn) / (mx - mn + 1e-10)
        # Onset strength (activity)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        # Harmonic energy (optional): from HPSS
        try:
            y_harmonic = librosa.effects.harmonic(y, margin=4)
            harm_rms = librosa.feature.rms(y=y_harmonic, frame_length=n_fft, hop_length=hop_length)[0]
            # Match length if needed
            if len(harm_rms) < len(rms):
                harm_rms = np.pad(harm_rms, (0, len(rms) - len(harm_rms)), mode="edge")
            elif len(harm_rms) > len(rms):
                harm_rms = harm_rms[: len(rms)]
        except Exception:
            harm_rms = np.zeros_like(rms)

        times = np.arange(len(rms)) * (hop_length / sr)

        # Normalize each feature to [0, 1] for weighted combination
        def _norm(x):
            x = np.array(x, dtype=float)
            mn, mx = x.min(), x.max()
            if mx - mn < 1e-10:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        out = {
            "times": times.tolist(),
            "loudness": _norm(rms).tolist(),
            "spectral_centroid": _norm(cent).tolist(),
            "spectral_flux": _norm(flux).tolist(),
            "onset_strength": _norm(onset).tolist(),
            "harmonic_energy": _norm(harm_rms).tolist(),
            "energy": _norm(rms).tolist(),
        }
        # Band energy for orchestra explorer: match frequency content to instrument groups
        out["band_energy"] = {k: v.tolist() for k, v in band_energy.items()}

        # Optional: Basic Pitch note/pitch activity (pitch_low, pitch_mid, pitch_high, melody_strength).
        # Combines with band_energy to better estimate which instrument group is playing.
        try:
            from analysis.basic_pitch_features import extract_pitch_activity
            duration_sec = len(y) / float(sr)
            bp = extract_pitch_activity(
                audio_path,
                duration_seconds=duration_sec,
                times_out=times.tolist(),
            )
            if bp and len(bp.get("times", [])) == len(times):
                out["pitch_low"] = bp["pitch_low"]
                out["pitch_mid"] = bp["pitch_mid"]
                out["pitch_high"] = bp["pitch_high"]
                out["melody_strength"] = bp["melody_strength"]
        except Exception:
            pass

        return out
    except Exception:
        return None


def get_audio_path_for_piece(piece_id: str) -> Optional[str]:
    """
    Resolve piece_id to a local audio file path if available.
    Returns None for external URLs or missing audio.
    """
    import sys
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from utils.retrieval import get_audio_info
    info = get_audio_info(piece_id)
    url = (info or {}).get("audio_url", "")
    if not url or url.startswith("http"):
        return None
    if url.startswith("/static/"):
        rel = url[len("/static/"):]
        return os.path.join(_ROOT, "static", rel.lstrip("/"))
    return None
