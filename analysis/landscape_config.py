"""
Configurable defaults for the Emotional Music Landscape and Instrument Landscape.
No hardcoded piece/composer/emotion names—all names and weights are configurable.
Extension point: load from data/landscape_config.json for per-deployment overrides.
"""

from typing import Any

# --- Emotional landscape: tension & emotional tone ---
# Tension: 0 = calm/release, 1 = intensity. Used for terrain height (mountains/valleys).
# Emotional tone: 0 = darker/cooler, 1 = brighter/warmer. Used for color mapping.

# High temporal resolution: small windows and hop for dense emotion estimates (general, not piece-specific)
FEATURE_WINDOW_SEC = 0.4        # Analysis window size in seconds (0.3–0.5 recommended)
FEATURE_HOP_SEC = 0.25          # Hop length in seconds (~0.25 for dense sequence)

# Multi-feature emotion weights: emotion = w1*loudness + w2*spectral_centroid + w3*spectral_flux + w4*onset
# Configurable so the system can be tuned; must sum to 1.0 for normalized output
EMOTION_WEIGHT_LOUDNESS = 0.35
EMOTION_WEIGHT_CENTROID = 0.20
EMOTION_WEIGHT_FLUX = 0.25
EMOTION_WEIGHT_ONSET = 0.20
# When harmonic energy is available it can be added; placeholder weight 0 for "use other four"
EMOTION_WEIGHT_HARMONIC = 0.0

# Smoothing: strong smoothing for overall smooth tendency (not fine detail)
# Larger sigma = smoother curve; keeps broad rises and falls only
LANDSCAPE_GAUSSIAN_SIGMA_SEC = 14.0
# Savitzky-Golay alternative: window length in points (odd). If > 0, can be used instead of Gaussian.
SAVGOL_WINDOW_LENGTH = 0        # 0 = use Gaussian

# Micro emotional variations: set to 0 for a clean overall tendency; increase slightly if you want subtle texture
MICRO_VARIATION_AMPLITUDE = 0.0    # No extra wobble — smooth tendency only
MICRO_VARIATION_FREQ_HZ = 0.4
MICRO_VARIATION_NOISE_SCALE = 0.0

# Fallback when no audio: number of points for journey-based curve (dense for mountain-range shape)
LANDSCAPE_NUM_POINTS = 400
TENSION_SMOOTH_WINDOW_SEC = 18.0
EMOTIONAL_TONE_SMOOTH_WINDOW_SEC = 22.0
# Label -> rough tension value when deriving from LLM journey (0–1)
# Keys are lowercased substrings; first match wins. Add more as needed.
DEFAULT_TENSION_FROM_LABEL = {
    "calm": 0.15,
    "opening": 0.2,
    "gentle": 0.25,
    "peaceful": 0.2,
    "developing": 0.45,
    "middle": 0.6,
    "peak": 0.95,
    "intensity": 0.9,
    "building": 0.7,
    "tension": 0.85,
    "climax": 0.95,
    "return": 0.5,
    "release": 0.3,
    "closing": 0.25,
    "ending": 0.2,
}
# Default tension when no keyword matches
DEFAULT_TENSION_NEUTRAL = 0.5

# --- Instrument landscape: group names and frequency-band mapping ---
# Default instrument groups for "instrument activity" view (estimates only).
# For piano-only pieces, a reduced set can be used (see instrument_analysis).
DEFAULT_INSTRUMENT_GROUPS = [
    "Strings",
    "Woodwinds",
    "Brass",
    "Piano",
    "Percussion",
]
# When no audio: approximate activity from tension curve with per-group phase/weight.
# Each entry: (phase_offset_0_to_1, weight). Activity = weight * tension(t + phase * duration)
DEFAULT_GROUP_ESTIMATE_PHASE_WEIGHT = [
    (0.0, 1.0),   # Strings: follows main tension
    (0.15, 0.85), # Woodwinds: slightly delayed
    (0.25, 0.7),  # Brass: later peaks
    (0.05, 0.9),  # Piano: close to main
    (0.2, 0.6),   # Percussion: accents
]
# Disclaimer shown in UI and in export
INSTRUMENT_DISCLAIMER = (
    "This is an interpretive estimate of instrument-group activity, "
    "not a precise separation. Use it as a listening guide."
)

# --- Orchestra Explorer: display groups (extensible; not hardcoded to one piece) ---
# Used for real-time “who is playing” visualization. Each group has id, name, tooltip.
# To replace mock detection with real source separation, plug in activity from your model here.
ORCHESTRA_EXPLORER_GROUPS = [
    {"id": "high_strings", "name": "Violins / high strings", "tooltip": "Violins and violas carry melody and shimmer. Listen for the main theme."},
    {"id": "low_strings", "name": "Cellos / low strings", "tooltip": "Cellos and basses add depth and rhythm. They often lead into big moments."},
    {"id": "woodwinds", "name": "Woodwinds", "tooltip": "Flutes, oboes, clarinets, bassoons add color and dialogue. Notice when they answer the strings."},
    {"id": "brass", "name": "Brass", "tooltip": "Horns, trumpets, trombones bring power and climaxes. Listen for them in the loudest sections."},
    {"id": "percussion", "name": "Percussion", "tooltip": "Timpani, drums, and other percussion mark rhythm and emphasis."},
    {"id": "piano_solo", "name": "Piano / solo", "tooltip": "Piano or solo instrument takes the spotlight in concertos and featured passages."},
]
# Phase/weight for each explorer group (same order as above). Mock: activity from tension curve.
# Replace with per-group activity from source separation when available.
ORCHESTRA_EXPLORER_PHASE_WEIGHT = [
    (0.0, 1.0),   # High strings
    (0.08, 0.9),  # Low strings
    (0.15, 0.85), # Woodwinds
    (0.25, 0.7),  # Brass
    (0.2, 0.6),   # Percussion
    (0.05, 0.9),  # Piano / solo
]

# Band weights for matching sound to instrument (when audio is available).
# Spectral: "low", "mid_low", "mid", "mid_high", "high" (Hz bands); "onset", "harmonic".
# Basic Pitch (when installed): "pitch_low" (MIDI ≤48), "pitch_mid" (48–72), "pitch_high" (>72), "melody" (note activity).
# One dict per explorer group (same order as ORCHESTRA_EXPLORER_GROUPS).
ORCHESTRA_EXPLORER_BAND_WEIGHTS = [
    {"mid_high": 0.4, "high": 0.4, "pitch_high": 0.15, "melody": 0.05},   # high_strings
    {"low": 0.4, "mid_low": 0.4, "pitch_low": 0.2},                      # low_strings
    {"mid": 0.4, "mid_high": 0.4, "pitch_high": 0.1, "melody": 0.1},    # woodwinds
    {"mid_low": 0.4, "mid": 0.4, "pitch_mid": 0.2},                     # brass
    {"onset": 0.6, "low": 0.2, "high": 0.2},                            # percussion
    {"mid": 0.25, "mid_high": 0.25, "harmonic": 0.25, "melody": 0.25},  # piano_solo: melody from Basic Pitch
]

# --- Section overlay (optional) ---
# If section markers are provided, show as light vertical lines or labels
SHOW_SECTION_MARKERS = True
SECTION_MARKER_ALPHA = 0.35

# --- Visualization: continuous emotional gradient ---
# low tension (0) -> cool (blue/cyan), medium (0.5) -> green, high (1) -> warm (orange/red)
# Values as CSS-style hex or named; used to interpolate color along timeline
COLOR_LOW_TENSION = "#2e86ab"   # cool blue
COLOR_MID_TENSION = "#4a7c59"   # green
COLOR_HIGH_TENSION = "#c75b39"  # warm orange/red
# Colormap name for matplotlib export (fallback)
EMOTIONAL_COLORMAP = "coolwarm"  # blue-green-red
# Terrain fill alpha
TERRAIN_FILL_ALPHA = 0.65
# Figure size (inches) for PNG export
FIGURE_SIZE_EMOTION = (12, 4)
FIGURE_SIZE_INSTRUMENT = (12, 6)


def get_landscape_config() -> dict[str, Any]:
    """
    Return full config dict. Can be extended to load from data/landscape_config.json.
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(root, "data", "landscape_config.json")
    out = {
        "feature_window_sec": FEATURE_WINDOW_SEC,
        "feature_hop_sec": FEATURE_HOP_SEC,
        "emotion_weight_loudness": EMOTION_WEIGHT_LOUDNESS,
        "emotion_weight_centroid": EMOTION_WEIGHT_CENTROID,
        "emotion_weight_flux": EMOTION_WEIGHT_FLUX,
        "emotion_weight_onset": EMOTION_WEIGHT_ONSET,
        "emotion_weight_harmonic": EMOTION_WEIGHT_HARMONIC,
        "landscape_gaussian_sigma_sec": LANDSCAPE_GAUSSIAN_SIGMA_SEC,
        "savgol_window_length": SAVGOL_WINDOW_LENGTH,
        "micro_variation_amplitude": MICRO_VARIATION_AMPLITUDE,
        "micro_variation_freq_hz": MICRO_VARIATION_FREQ_HZ,
        "micro_variation_noise_scale": MICRO_VARIATION_NOISE_SCALE,
        "tension_smooth_window_sec": TENSION_SMOOTH_WINDOW_SEC,
        "emotional_tone_smooth_window_sec": EMOTIONAL_TONE_SMOOTH_WINDOW_SEC,
        "landscape_num_points": LANDSCAPE_NUM_POINTS,
        "color_low_tension": COLOR_LOW_TENSION,
        "color_mid_tension": COLOR_MID_TENSION,
        "color_high_tension": COLOR_HIGH_TENSION,
        "tension_from_label": dict(DEFAULT_TENSION_FROM_LABEL),
        "tension_neutral": DEFAULT_TENSION_NEUTRAL,
        "instrument_groups": list(DEFAULT_INSTRUMENT_GROUPS),
        "group_phase_weight": list(DEFAULT_GROUP_ESTIMATE_PHASE_WEIGHT),
        "orchestra_explorer_groups": list(ORCHESTRA_EXPLORER_GROUPS),
        "orchestra_explorer_phase_weight": list(ORCHESTRA_EXPLORER_PHASE_WEIGHT),
        "orchestra_explorer_band_weights": list(ORCHESTRA_EXPLORER_BAND_WEIGHTS),
        "instrument_disclaimer": INSTRUMENT_DISCLAIMER,
        "show_section_markers": SHOW_SECTION_MARKERS,
        "section_marker_alpha": SECTION_MARKER_ALPHA,
        "emotional_colormap": EMOTIONAL_COLORMAP,
        "terrain_fill_alpha": TERRAIN_FILL_ALPHA,
        "figure_size_emotion": list(FIGURE_SIZE_EMOTION),
        "figure_size_instrument": list(FIGURE_SIZE_INSTRUMENT),
    }
    try:
        import json
        if os.path.isfile(json_path):
            with open(json_path, encoding="utf-8") as f:
                overrides = json.load(f)
            for k, v in overrides.items():
                if k in out:
                    out[k] = v
    except Exception:
        pass
    return out
