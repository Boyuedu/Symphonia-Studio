"""
Pitch and melody activity from Spotify Basic Pitch for Orchestra Explorer.
Basic Pitch detects notes and pitch (not instrument type); we combine it with
spectral bands to estimate which instrument group is active.

Outputs time-gridded signals: pitch_low, pitch_mid, pitch_high (activity by
MIDI range), and melody_strength (overall note activity). Map these in
orchestra_explorer_band_weights to instrument groups (e.g. high_strings <- pitch_high + melody).
"""

from __future__ import annotations

import os
from typing import Any, Optional

# MIDI ranges: low (cello/bass), mid (piano mid, brass), high (violin, flute, melody)
MIDI_LOW_MAX = 48   # C3 and below
MIDI_MID_MAX = 72   # C5; above is high
# high: 72-108 (and above if present)


def extract_pitch_activity(
    audio_path: str,
    duration_seconds: Optional[float] = None,
    times_out: Optional[list[float]] = None,
    n_points: int = 400,
) -> Optional[dict[str, Any]]:
    """
    Run Basic Pitch on audio and return pitch-range activity aligned to a time grid.

    Args:
        audio_path: Path to audio file (e.g. .mp3, .wav). Basic Pitch resamples to 22050 Hz.
        duration_seconds: If set, only consider notes within [0, duration_seconds].
        times_out: If provided, output signals are interpolated to these times.
                  Otherwise use n_points uniformly in [0, duration_seconds].

    Returns:
        Dict with:
          - times: list of float (time grid)
          - pitch_low: activity in low MIDI range (<= MIDI_LOW_MAX), normalized [0,1]
          - pitch_mid: activity in mid range (MIDI_LOW_MAX < p <= MIDI_MID_MAX)
          - pitch_high: activity in high range (> MIDI_MID_MAX)
          - melody_strength: total note activity (all pitches), normalized [0,1]
        Returns None if Basic Pitch is not installed or prediction fails.
    """
    if not os.path.isfile(audio_path):
        return None
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        import numpy as np
    except ImportError:
        return None

    try:
        # predict() processes full file; note_events = (start_s, end_s, pitch_midi, amplitude, pitch_bend)
        _, _, note_events = predict(
            audio_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )
    except Exception:
        return None

    if not note_events:
        return _empty_pitch_activity(duration_seconds or 300.0, times_out, n_points)

    duration = duration_seconds
    if duration is None and note_events:
        max_end = max(e[1] for e in note_events)
        duration = max(1.0, max_end)

    if times_out is not None:
        t_grid = np.array(times_out, dtype=float)
    else:
        t_grid = np.linspace(0, duration, n_points)

    # Build per-frame activity: at each t, sum amplitude of notes active at t, by pitch band
    pitch_low = np.zeros_like(t_grid)
    pitch_mid = np.zeros_like(t_grid)
    pitch_high = np.zeros_like(t_grid)
    melody = np.zeros_like(t_grid)

    for (start_s, end_s, pitch_midi, amplitude, _) in note_events:
        if duration is not None and (start_s >= duration or end_s <= 0):
            continue
        # Which grid indices does this note cover?
        mask = (t_grid >= start_s) & (t_grid <= end_s)
        if pitch_midi <= MIDI_LOW_MAX:
            pitch_low[mask] += amplitude
        elif pitch_midi <= MIDI_MID_MAX:
            pitch_mid[mask] += amplitude
        else:
            pitch_high[mask] += amplitude
        melody[mask] += amplitude

    def _norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-10:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    return {
        "times": t_grid.tolist(),
        "pitch_low": _norm(pitch_low).tolist(),
        "pitch_mid": _norm(pitch_mid).tolist(),
        "pitch_high": _norm(pitch_high).tolist(),
        "melody_strength": _norm(melody).tolist(),
    }


def _empty_pitch_activity(
    duration: float,
    times_out: Optional[list[float]],
    n_points: int,
) -> dict[str, Any]:
    """Return zeroed pitch activity when Basic Pitch returns no notes."""
    import numpy as np
    if times_out is not None:
        t_grid = np.array(times_out, dtype=float)
    else:
        t_grid = np.linspace(0, duration, n_points)
    n = len(t_grid)
    return {
        "times": t_grid.tolist(),
        "pitch_low": [0.0] * n,
        "pitch_mid": [0.0] * n,
        "pitch_high": [0.0] * n,
        "melody_strength": [0.0] * n,
    }
