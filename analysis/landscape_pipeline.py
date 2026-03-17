"""
Emotional Music Landscape pipeline: modular generation of emotion curves and
landscape data. Supports high temporal resolution, multi-feature emotion estimation,
smoothing, micro-variations, and continuous color mapping. General for any piece and duration.
"""

import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.landscape_config import get_landscape_config

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ----- Helpers -----

def _tension_from_label(label: str, config: dict) -> float:
    """Map a journey label to a tension value in [0, 1]."""
    table = config.get("tension_from_label") or {}
    neutral = config.get("tension_neutral", 0.5)
    lower = (label or "").lower()
    for keyword, value in table.items():
        if keyword in lower:
            return float(value)
    return neutral


def _interp(t: float, t_vals: list[float], v_vals: list[float]) -> float:
    """Linear interpolation. t_vals must be increasing."""
    if t <= t_vals[0]:
        return v_vals[0]
    if t >= t_vals[-1]:
        return v_vals[-1]
    for i in range(len(t_vals) - 1):
        if t_vals[i] <= t <= t_vals[i + 1]:
            a = (t - t_vals[i]) / (t_vals[i + 1] - t_vals[i])
            return v_vals[i] * (1 - a) + v_vals[i + 1] * a
    return v_vals[-1]


def _default_journey_points(duration_seconds: float) -> list[dict[str, Any]]:
    """Minimal journey when no LLM journey is provided (general, any duration)."""
    step = max(1, duration_seconds / 5)
    return [
        {"time_sec": 0, "label": "Opening"},
        {"time_sec": step, "label": "Developing"},
        {"time_sec": 2 * step, "label": "Middle"},
        {"time_sec": 3 * step, "label": "Return"},
        {"time_sec": duration_seconds, "label": "Closing"},
    ]


# ----- 1. Extract audio features (high temporal resolution) -----

def extract_audio_features(
    audio_path: Optional[str],
    duration_seconds: float,
    config: Optional[dict] = None,
) -> Optional[dict[str, Any]]:
    """
    Extract dense audio features using small windows and hop.
    window_sec ~0.3–0.5 s, hop_sec ~0.25 s for fine-grained emotion estimates.
    Returns None if no audio or extraction fails.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    config = config or get_landscape_config()
    window_sec = config.get("feature_window_sec", 0.4)
    hop_sec = config.get("feature_hop_sec", 0.25)
    from analysis.audio_features import extract_features_from_audio as _extract
    return _extract(
        audio_path,
        duration_seconds=duration_seconds,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )


# ----- 2. Compute raw emotion curve (multi-feature weighted) -----

def compute_emotion_curve(
    features: Optional[dict[str, Any]],
    journey: Optional[list[dict[str, Any]]],
    duration_seconds: float,
    config: Optional[dict] = None,
) -> tuple[list[float], list[float]]:
    """
    Compute raw emotion (tension) values over time.
    - If features: emotion = w1*loudness + w2*centroid + w3*flux + w4*onset (configurable weights).
    - If no features: dense grid from journey interpolation + multiple small peaks (mountain range).
    Returns (times, raw_emotion) both as lists.
    """
    config = config or get_landscape_config()
    duration_seconds = max(1.0, float(duration_seconds))

    if features and HAS_NUMPY:
        t = np.array(features["times"])
        w1 = config.get("emotion_weight_loudness", 0.35)
        w2 = config.get("emotion_weight_centroid", 0.20)
        w3 = config.get("emotion_weight_flux", 0.25)
        w4 = config.get("emotion_weight_onset", 0.20)
        w5 = config.get("emotion_weight_harmonic", 0.0)
        # Normalize weights to sum to 1
        s = w1 + w2 + w3 + w4 + w5
        if s < 1e-6:
            s = 1.0
        w1, w2, w3, w4, w5 = w1 / s, w2 / s, w3 / s, w4 / s, w5 / s
        loud = np.array(features.get("loudness") or features.get("energy", []))
        cent = np.array(features.get("spectral_centroid", []))
        flux = np.array(features.get("spectral_flux", []))
        onset = np.array(features.get("onset_strength", []))
        harm = np.array(features.get("harmonic_energy", np.zeros_like(loud)))
        n = min(len(t), len(loud), len(cent), len(flux), len(onset))
        if n == 0:
            t = np.linspace(0, duration_seconds, max(10, int(duration_seconds / 0.25)))
            raw = 0.5 * np.ones_like(t)
            return t.tolist(), raw.tolist()
        t, loud, cent, flux, onset = t[:n], loud[:n], cent[:n], flux[:n], onset[:n]
        harm = harm[:n] if len(harm) >= n else np.zeros(n)
        raw = w1 * loud + w2 * cent + w3 * flux + w4 * onset + w5 * harm
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        return t.tolist(), raw.tolist()

    # No audio: dense grid from journey; smooth tendency only (no extra peaks)
    num_points = config.get("landscape_num_points", 400)
    if not journey:
        journey = _default_journey_points(duration_seconds)
    times_j = [p["time_sec"] for p in journey]
    tension_j = [_tension_from_label(p.get("label", ""), config) for p in journey]
    if HAS_NUMPY:
        t_unif = np.linspace(0, duration_seconds, num_points)
        base = np.interp(t_unif, times_j, tension_j)
        base = np.clip(base, 0, 1)
        return t_unif.tolist(), base.tolist()
    t_unif = [duration_seconds * i / (num_points - 1) for i in range(num_points)]
    base = [_interp(t, times_j, tension_j) for t in t_unif]
    return t_unif, base


# ----- 3. Smooth the emotion curve -----

def smooth_emotion_curve(
    values: list[float],
    times: list[float],
    duration_seconds: float,
    config: Optional[dict] = None,
) -> list[float]:
    """
    Smooth raw emotion to remove abrupt jumps while preserving gradual variation.
    Uses Gaussian smoothing by default; optional Savitzky-Golay if configured.
    """
    config = config or get_landscape_config()
    if not values or len(values) < 3:
        return list(values)
    if HAS_NUMPY:
        a = np.array(values, dtype=float)
        savgol = config.get("savgol_window_length", 0)
        if savgol and savgol >= 3 and len(a) > savgol:
            try:
                from scipy.signal import savgol_filter
                pol = min(3, savgol - 1)
                return savgol_filter(a, savgol, pol).tolist()
            except Exception:
                pass
        sigma_sec = config.get("landscape_gaussian_sigma_sec", 4.0)
        n = len(a)
        sigma_pts = sigma_sec * n / max(1, duration_seconds)
        sigma_pts = max(1.0, sigma_pts)
        half = min(n // 2, max(3, int(4 * sigma_pts)))
        x = np.arange(-half, half + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma_pts) ** 2)
        kernel /= kernel.sum()
        out = np.convolve(a, kernel, mode="same")
        return out.tolist()
    return list(values)


# ----- 4. Add micro emotional variations -----

def add_micro_variations(
    smoothed: list[float],
    times: list[float],
    config: Optional[dict] = None,
) -> list[float]:
    """
    Add subtle micro-variations so the curve feels natural (small amplitude).
    emotion_final = emotion_smoothed + small_sinusoidal + optional_noise.
    Values clamped to [0, 1].
    """
    config = config or get_landscape_config()
    amp = config.get("micro_variation_amplitude", 0.03)
    freq_hz = config.get("micro_variation_freq_hz", 0.4)
    noise_scale = config.get("micro_variation_noise_scale", 0.015)
    if not smoothed or not times:
        return list(smoothed)
    if HAS_NUMPY:
        a = np.array(smoothed, dtype=float)
        t = np.array(times, dtype=float)
        # Slow sinusoidal component
        a = a + amp * np.sin(2 * np.pi * freq_hz * t)
        # Optional small noise
        if noise_scale > 0:
            rng = np.random.default_rng(42)
            a = a + noise_scale * (rng.random(len(a)) * 2 - 1)
        a = np.clip(a, 0, 1)
        return a.tolist()
    return list(smoothed)


# ----- 5. Height to continuous color (gradient) -----

def _height_to_color(h: float, config: dict) -> str:
    """
    Map tension/height in [0, 1] to a continuous gradient:
    low -> cool (blue/cyan), medium -> green, high -> warm (orange/red).
    Returns hex color string.
    """
    def _parse_hex(hex_str: str) -> tuple[int, int, int]:
        s = (hex_str or "#000000").strip().lstrip("#")
        if len(s) == 6:
            return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
        return 0, 0, 0

    low = _parse_hex(config.get("color_low_tension", "#2e86ab"))
    mid = _parse_hex(config.get("color_mid_tension", "#4a7c59"))
    high = _parse_hex(config.get("color_high_tension", "#c75b39"))
    h = max(0, min(1, h))
    if h <= 0.5:
        t = h * 2  # 0->1 over first half
        r = int(low[0] + (mid[0] - low[0]) * t)
        g = int(low[1] + (mid[1] - low[1]) * t)
        b = int(low[2] + (mid[2] - low[2]) * t)
    else:
        t = (h - 0.5) * 2
        r = int(mid[0] + (high[0] - mid[0]) * t)
        g = int(mid[1] + (high[1] - mid[1]) * t)
        b = int(mid[2] + (high[2] - mid[2]) * t)
    r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ----- 6. Generate landscape data (orchestrator) -----

def generate_landscape_data(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    duration_seconds: float,
    journey: Optional[list[dict[str, Any]]] = None,
    section_markers: Optional[list[dict[str, Any]]] = None,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Full pipeline: extract features (if audio) -> compute emotion curve ->
    smooth -> add micro variations -> normalize -> build points with color.
    Returns:
      points: list of { time, height, color } for visualization
      times, tension, emotional_tone: for legacy API (same length as points)
      duration_seconds, section_markers
    """
    config = get_landscape_config()
    duration_seconds = max(1.0, float(duration_seconds))

    # Step 1: Extract audio features (high resolution) or use journey
    features = extract_audio_features(audio_path, duration_seconds, config)

    # Step 2: Compute raw emotion curve (multi-feature or journey-based mountain range)
    times, raw = compute_emotion_curve(features, journey, duration_seconds, config)
    if not journey and not features:
        journey = _default_journey_points(duration_seconds)
        times, raw = compute_emotion_curve(None, journey, duration_seconds, config)

    # Step 3: Smooth (Gaussian or Savitzky-Golay)
    smoothed = smooth_emotion_curve(raw, times, duration_seconds, config)

    # Step 4: Add micro variations
    with_variations = add_micro_variations(smoothed, times, config)

    # Normalize to [0, 1] after all steps
    if HAS_NUMPY:
        a = np.array(with_variations, dtype=float)
        mn, mx = a.min(), a.max()
        if mx - mn > 1e-10:
            a = (a - mn) / (mx - mn)
        a = np.clip(a, 0, 1)
        tension_final = a.tolist()
    else:
        mn, mx = min(with_variations), max(with_variations)
        r = mx - mn + 1e-10
        tension_final = [max(0, min(1, (x - mn) / r)) for x in with_variations]

    # Emotional tone for legacy API: use same curve or slight offset for color variety
    if HAS_NUMPY:
        tone = np.roll(np.array(tension_final), max(1, len(tension_final) // 20))
        tone = (tone - tone.min()) / (tone.max() - tone.min() + 1e-8)
        emotional_tone = tone.tolist()
    else:
        emotional_tone = tension_final

    # Build points with continuous color gradient
    points = []
    for i, t in enumerate(times):
        h = tension_final[i] if i < len(tension_final) else 0.5
        points.append({
            "time": float(t),
            "height": h,
            "color": _height_to_color(h, config),
        })

    return {
        "points": points,
        "times": times,
        "tension": tension_final,
        "emotional_tone": emotional_tone,
        "duration_seconds": duration_seconds,
        "section_markers": section_markers or [],
    }


# ----- 7. Main entry: compute_emotional_landscape (API-compatible) -----

def compute_emotional_landscape(
    piece_id: str,
    piece_title: str,
    composer_name: str,
    duration_seconds: float,
    journey: Optional[list[dict[str, Any]]] = None,
    section_markers: Optional[list[dict[str, Any]]] = None,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compute emotional landscape; returns both new points format and legacy times/tension/emotional_tone.
    """
    return generate_landscape_data(
        piece_id, piece_title, composer_name, duration_seconds,
        journey=journey, section_markers=section_markers, audio_path=audio_path,
    )


# ----- Instrument landscape (unchanged logic, uses config num_points) -----

def compute_instrument_landscape(
    piece_id: str,
    duration_seconds: float,
    tension_curve: Optional[list[float]] = None,
    times_curve: Optional[list[float]] = None,
    instrument_groups: Optional[list[str]] = None,
    has_orchestra: bool = False,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Estimate instrument-group activity over time. Same visual metaphor:
    mountains = more activity/tension, valleys = calmer.
    """
    config = get_landscape_config()
    groups = instrument_groups or config.get("instrument_groups", [])
    phase_weight = config.get("group_phase_weight") or []
    num_points = config.get("landscape_num_points", 400)
    duration_seconds = max(1.0, float(duration_seconds))

    if audio_path and HAS_NUMPY:
        from analysis.audio_features import extract_features_from_audio
        feats = extract_features_from_audio(
            audio_path, duration_seconds=duration_seconds,
            window_sec=config.get("feature_window_sec", 0.4),
            hop_sec=config.get("feature_hop_sec", 0.25),
        )
        if feats:
            times_audio = np.array(feats["times"])
            energy = np.array(feats.get("energy") or feats.get("loudness", []))
            n = len(times_audio)
            ng = len(groups)
            activity = []
            for g in range(ng):
                phase = phase_weight[g][0] if g < len(phase_weight) else g / max(1, ng)
                shift = int(phase * n * 0.3) % n
                band = np.roll(energy, shift) * (phase_weight[g][1] if g < len(phase_weight) else 0.8)
                activity.append(band.tolist())
            t_unif = np.linspace(0, duration_seconds, num_points)
            out_groups = []
            for g, name in enumerate(groups):
                if g < len(activity):
                    act = np.interp(t_unif, times_audio, activity[g]).tolist()
                else:
                    act = [0.5] * num_points
                out_groups.append({"name": name, "activity": act})
            return {
                "times": t_unif.tolist(),
                "groups": out_groups,
                "duration_seconds": duration_seconds,
                "disclaimer": config.get("instrument_disclaimer", ""),
            }
    if not tension_curve or not times_curve:
        journey = _default_journey_points(duration_seconds)
        times_j = [p["time_sec"] for p in journey]
        tension_j = [_tension_from_label(p.get("label", ""), config) for p in journey]
        if HAS_NUMPY:
            t_unif = np.linspace(0, duration_seconds, num_points)
            tension_curve = np.interp(t_unif, times_j, tension_j).tolist()
        else:
            t_unif = [duration_seconds * i / (num_points - 1) for i in range(num_points)]
            tension_curve = [_interp(t, times_j, tension_j) for t in t_unif]
        times_curve = t_unif
    n = len(tension_curve)
    out_groups = []
    for g, name in enumerate(groups):
        phase, weight = (phase_weight[g] if g < len(phase_weight) else (0.0, 1.0))
        if HAS_NUMPY:
            t_curve = np.array(times_curve)
            tension_arr = np.array(tension_curve)
            t_shifted = np.clip(t_curve - phase * duration_seconds, 0, duration_seconds)
            act = np.interp(t_curve, t_shifted, tension_arr) * weight
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            out_groups.append({"name": name, "activity": act.tolist()})
        else:
            shifted = [_interp(max(0, t - phase * duration_seconds), times_curve, tension_curve) * weight for t in times_curve]
            mn, mx = min(shifted), max(shifted)
            act = [(x - mn) / (mx - mn + 1e-8) for x in shifted]
            out_groups.append({"name": name, "activity": act})
    return {
        "times": times_curve,
        "groups": out_groups,
        "duration_seconds": duration_seconds,
        "disclaimer": config.get("instrument_disclaimer", ""),
    }


# ----- Orchestra Explorer: real-time “who is playing” (extensible; mock = phase/weight) -----
# To replace with real source separation: return activity arrays from your model per group id.


def compute_orchestra_explorer_activity(
    piece_id: str,
    duration_seconds: float,
    tension_curve: Optional[list[float]] = None,
    times_curve: Optional[list[float]] = None,
    has_orchestra: bool = True,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Compute per-group activity for Orchestra Explorer (sync with playback).
    Uses configurable 6 groups (high strings, low strings, woodwinds, brass, percussion, piano).
    Placeholder: activity from tension curve + phase/weight per group.
    Replace tension_curve/times_curve with real per-group activity from source separation when available.
    """
    config = get_landscape_config()
    explorer_groups = config.get("orchestra_explorer_groups") or []
    phase_weight = config.get("orchestra_explorer_phase_weight") or []
    num_points = config.get("landscape_num_points", 400)
    duration_seconds = max(1.0, float(duration_seconds))

    # When audio is available: use band energy + band weights so activity matches sound to instrument.
    # Each group's activity = weighted sum of frequency bands (and optionally onset/harmonic).
    if audio_path and HAS_NUMPY:
        from analysis.audio_features import extract_features_from_audio
        feats = extract_features_from_audio(
            audio_path, duration_seconds=duration_seconds,
            window_sec=config.get("feature_window_sec", 0.4),
            hop_sec=config.get("feature_hop_sec", 0.25),
        )
        if feats:
            times_audio = np.array(feats["times"])
            t_unif = np.linspace(0, duration_seconds, num_points)
            band_weights_list = config.get("orchestra_explorer_band_weights") or []
            band_energy = feats.get("band_energy")

            if band_energy and isinstance(band_energy, dict) and len(band_energy) > 0:
                # Build signals for each band (and onset, harmonic) interpolated to t_unif
                signals = {}
                for band_name, values in band_energy.items():
                    if isinstance(values, (list, tuple)) and len(values) == len(times_audio):
                        signals[band_name] = np.interp(t_unif, times_audio, np.array(values, dtype=float))
                if feats.get("onset_strength") and len(feats["onset_strength"]) == len(times_audio):
                    signals["onset"] = np.interp(t_unif, times_audio, np.array(feats["onset_strength"], dtype=float))
                if feats.get("harmonic_energy") and len(feats["harmonic_energy"]) == len(times_audio):
                    signals["harmonic"] = np.interp(t_unif, times_audio, np.array(feats["harmonic_energy"], dtype=float))
                # Basic Pitch: pitch-range and melody activity (when available)
                for key, feat_key in [("pitch_low", "pitch_low"), ("pitch_mid", "pitch_mid"), ("pitch_high", "pitch_high"), ("melody", "melody_strength")]:
                    if feat_key in feats and len(feats[feat_key]) == len(times_audio):
                        signals[key] = np.interp(t_unif, times_audio, np.array(feats[feat_key], dtype=float))

                # Piano/solo-only piece: only piano_solo group gets activity; others stay at 0
                piano_solo_id = "piano_solo"
                out = []
                for g, grp in enumerate(explorer_groups):
                    gid = grp.get("id", f"g{g}")
                    if not has_orchestra and gid != piano_solo_id:
                        out.append({
                            "id": gid,
                            "name": grp.get("name", ""),
                            "tooltip": grp.get("tooltip", ""),
                            "activity": [0.0] * len(t_unif),
                        })
                        continue
                    weights = band_weights_list[g] if g < len(band_weights_list) else {}
                    act = np.zeros_like(t_unif)
                    for key, w in (weights or {}).items():
                        if key in signals and w != 0:
                            act = act + w * signals[key]
                    if act.max() > act.min():
                        act = (act - act.min()) / (act.max() - act.min())
                    else:
                        act = np.clip(act, 0, 1)
                    out.append({
                        "id": gid,
                        "name": grp.get("name", ""),
                        "tooltip": grp.get("tooltip", ""),
                        "activity": act.tolist(),
                    })
                return {
                    "times": t_unif.tolist(),
                    "groups": out,
                    "duration_seconds": duration_seconds,
                    "disclaimer": config.get("instrument_disclaimer", ""),
                }

            # Fallback: no band_energy (e.g. old extract), use phase-shifted energy per group
            energy = np.array(feats.get("energy") or feats.get("loudness", []))
            n = len(times_audio)
            ng = len(explorer_groups)
            energy_curve = np.interp(t_unif, times_audio, energy)
            if energy_curve.max() > energy_curve.min():
                energy_curve = (energy_curve - energy_curve.min()) / (energy_curve.max() - energy_curve.min() + 1e-8)
            out = []
            piano_solo_id = "piano_solo"
            for g in range(ng):
                grp = explorer_groups[g] if g < len(explorer_groups) else {"id": f"g{g}", "name": f"Group {g}", "tooltip": ""}
                gid = grp.get("id", f"g{g}")
                if not has_orchestra and gid != piano_solo_id:
                    out.append({"id": gid, "name": grp.get("name", ""), "tooltip": grp.get("tooltip", ""), "activity": [0.0] * len(t_unif)})
                    continue
                if not has_orchestra and gid == piano_solo_id:
                    act = energy_curve.copy()
                else:
                    phase, weight = (phase_weight[g] if g < len(phase_weight) else (g / max(1, ng), 0.8))
                    shift = int(phase * n * 0.3) % n
                    band = np.roll(energy, shift) * weight
                    act = np.interp(t_unif, times_audio, band)
                    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
                out.append({
                    "id": gid,
                    "name": grp.get("name", ""),
                    "tooltip": grp.get("tooltip", ""),
                    "activity": act.tolist(),
                })
            return {
                "times": t_unif.tolist(),
                "groups": out,
                "duration_seconds": duration_seconds,
                "disclaimer": config.get("instrument_disclaimer", ""),
            }

    # No audio or extraction failed: use tension curve + phase/weight (mock)
    if not tension_curve or not times_curve:
        journey = _default_journey_points(duration_seconds)
        times_j = [p["time_sec"] for p in journey]
        tension_j = [_tension_from_label(p.get("label", ""), config) for p in journey]
        if HAS_NUMPY:
            t_unif = np.linspace(0, duration_seconds, num_points)
            tension_curve = np.interp(t_unif, times_j, tension_j).tolist()
        else:
            t_unif = [duration_seconds * i / (num_points - 1) for i in range(num_points)]
            tension_curve = [_interp(t, times_j, tension_j) for t in t_unif]
        times_curve = t_unif
    piano_solo_id = "piano_solo"
    out = []
    for g, grp in enumerate(explorer_groups):
        gid = grp.get("id", f"g{g}")
        if not has_orchestra and gid != piano_solo_id:
            n_pts = len(times_curve)
            out.append({
                "id": gid,
                "name": grp.get("name", ""),
                "tooltip": grp.get("tooltip", ""),
                "activity": [0.0] * n_pts,
            })
            continue
        phase, weight = (phase_weight[g] if g < len(phase_weight) else (0.0, 1.0))
        if HAS_NUMPY:
            t_curve = np.array(times_curve)
            tension_arr = np.array(tension_curve)
            if not has_orchestra and gid == piano_solo_id:
                act = tension_arr.copy()
            else:
                t_shifted = np.clip(t_curve - phase * duration_seconds, 0, duration_seconds)
                act = np.interp(t_curve, t_shifted, tension_arr) * weight
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            out.append({
                "id": gid,
                "name": grp.get("name", ""),
                "tooltip": grp.get("tooltip", ""),
                "activity": act.tolist(),
            })
        else:
            if not has_orchestra and gid == piano_solo_id:
                shifted = tension_curve
            else:
                shifted = [_interp(max(0, t - phase * duration_seconds), times_curve, tension_curve) * weight for t in times_curve]
            mn, mx = min(shifted), max(shifted)
            act = [(x - mn) / (mx - mn + 1e-8) for x in shifted]
            out.append({
                "id": gid,
                "name": grp.get("name", ""),
                "tooltip": grp.get("tooltip", ""),
                "activity": act,
            })
    return {
        "times": times_curve,
        "groups": out,
        "duration_seconds": duration_seconds,
        "disclaimer": config.get("instrument_disclaimer", ""),
    }
