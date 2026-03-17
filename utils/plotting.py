"""
Emotion landscape / music journey plotting.
Produces saveable images (PNG) for:
- Emotional Music Landscape: terrain (mountains = tension, color = emotional tone)
- Instrument Landscape: stacked or layered activity curves (estimate).
"""

import io
import os
import sys
from typing import Any, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    np = None


def plot_emotion_landscape(
    journey: list[dict[str, Any]],
    piece_title: str,
    composer_name: str,
    duration_seconds: int,
    out_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot emotional journey as a landscape (time vs. "height" / intensity).
    journey: list of {time_sec, label, description}
    Returns path to saved image, or None if matplotlib unavailable or error.
    """
    if not HAS_MATPLOTLIB or not journey:
        return None

    times = [p["time_sec"] for p in journey]
    labels = [p.get("label", "") for p in journey]
    # Use index as proxy for "emotional height" so the line has shape; could later use sentiment/valence
    y = list(range(len(times)))
    if duration_seconds <= 0:
        duration_seconds = max(times) if times else 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(times, 0, y + [y[-1]], alpha=0.4, color="steelblue")
    ax.plot(times, y + [y[-1]], color="navy", linewidth=2)
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(0, max(y) + 1 if y else 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Emotional journey")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"Music landscape: {piece_title}\n{composer_name}", fontsize=12)
    fig.tight_layout()

    if out_path:
        try:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return out_path
        except Exception:
            plt.close(fig)
            return None
    plt.close(fig)
    return None


def plot_emotion_landscape_to_bytes(
    journey: list[dict[str, Any]],
    piece_title: str,
    composer_name: str,
    duration_seconds: int,
) -> Optional[bytes]:
    """Same as plot_emotion_landscape but returns PNG bytes (for in-memory serve or download)."""
    if not HAS_MATPLOTLIB or not journey:
        return None
    import io
    buf = io.BytesIO()
    times = [p["time_sec"] for p in journey]
    labels = [p.get("label", "") for p in journey]
    y = list(range(len(times)))
    if duration_seconds <= 0:
        duration_seconds = max(times) if times else 1
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(times, 0, y + [y[-1]], alpha=0.4, color="steelblue")
    ax.plot(times, y + [y[-1]], color="navy", linewidth=2)
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(0, max(y) + 1 if y else 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Emotional journey")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"Music landscape: {piece_title}\n{composer_name}", fontsize=12)
    fig.tight_layout()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        out = buf.getvalue()
        plt.close(fig)
        return out
    except Exception:
        plt.close(fig)
        return None


# --- Emotional Music Landscape: terrain (tension = height, color = emotional tone) ---

def _get_landscape_config() -> dict:
    try:
        from analysis.landscape_config import get_landscape_config
        return get_landscape_config()
    except Exception:
        return {}


def render_emotion_landscape_terrain_to_bytes(
    times: list[float],
    tension: list[float],
    emotional_tone: list[float],
    duration_seconds: float,
    piece_title: str = "",
    composer_name: str = "",
    section_markers: Optional[list[dict]] = None,
) -> Optional[bytes]:
    """
    Render the emotional landscape as a terrain: x = time, y = tension (mountains/valleys),
    color = emotional tone (darker to brighter). Returns PNG bytes for download/display.
    """
    if not HAS_MATPLOTLIB or not times or len(times) != len(tension) or len(times) != len(emotional_tone):
        return None
    config = _get_landscape_config()
    figsize = tuple(config.get("figure_size_emotion", (12, 4)))
    fill_alpha = config.get("terrain_fill_alpha", 0.65)
    cmap_name = config.get("emotional_colormap", "viridis")
    show_markers = config.get("show_section_markers", True)
    marker_alpha = config.get("section_marker_alpha", 0.35)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#fafaf8")
    fig.patch.set_facecolor("#fafaf8")
    t = np.array(times)
    y = np.array(tension)
    tone = np.array(emotional_tone)
    # Normalize y to 0–1 for display
    if y.max() > y.min():
        y = (y - y.min()) / (y.max() - y.min())
    # Fill under curve with a soft color; then line colored by emotional tone
    ax.fill_between(t, 0, y, alpha=fill_alpha, color="lavender", interpolate=True)
    # Line segments colored by emotional_tone (0 = cool, 1 = warm)
    points = np.stack([t, y], axis=1)
    segs = np.concatenate([points[:-1], points[1:]], axis=1).reshape(-1, 2, 2)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(tone[:-1])
    lc = LineCollection(segs, colors=colors, linewidth=2.5, capstyle="round")
    ax.add_collection(lc)
    # Section markers (optional)
    if show_markers and section_markers:
        for m in section_markers:
            ts = m.get("time_sec")
            if ts is not None and 0 <= ts <= duration_seconds:
                ax.axvline(x=ts, color="gray", linestyle="--", alpha=marker_alpha, linewidth=1)
    ax.set_xlim(0, max(duration_seconds, t.max()))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (min:sec)", fontsize=10)
    ax.set_ylabel("Tension / intensity", fontsize=10)
    # Format x-axis as min:sec
    def _fmt_sec(x):
        m = int(x // 60)
        s = int(x % 60)
        return f"{m}:{s:02d}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: _fmt_sec(x)))
    title = f"{piece_title}" if piece_title else "Emotional landscape"
    if composer_name:
        title += f" — {composer_name}"
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        out = buf.getvalue()
        plt.close(fig)
        return out
    except Exception:
        plt.close(fig)
        return None


def render_instrument_landscape_to_bytes(
    times: list[float],
    groups: list[dict[str, Any]],
    duration_seconds: float,
    piece_title: str = "",
    composer_name: str = "",
) -> Optional[bytes]:
    """
    Render instrument-group activity as stacked mini-landscapes (same metaphor:
    mountains = activity, valleys = calm). Each group gets a row. Clearly interpretive.
    """
    if not HAS_MATPLOTLIB or not times or not groups:
        return None
    config = _get_landscape_config()
    figsize = tuple(config.get("figure_size_instrument", (12, 6)))
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#fafaf8")
    t = np.array(times)
    for i, grp in enumerate(groups):
        ax = axes[i]
        ax.set_facecolor("#fafaf8")
        act = np.array(grp.get("activity", []))
        if len(act) == 0:
            act = np.zeros_like(t)
        elif len(act) != len(t):
            act = np.interp(np.linspace(0, len(act) - 1, len(t)), np.arange(len(act)), act)
        name = grp.get("name", f"Group {i+1}")
        ax.fill_between(t, 0, act, alpha=0.6, color="steelblue")
        ax.plot(t, act, color="navy", linewidth=1.5)
        ax.set_ylabel(name, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(duration_seconds, t.max()))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time (min:sec)", fontsize=10)
    def _fmt_sec(x, p):
        m = int(x // 60)
        s = int(x % 60)
        return f"{m}:{s:02d}"
    axes[-1].xaxis.set_major_formatter(plt.FuncFormatter(_fmt_sec))
    title = "Instrument activity (interpretive estimate)"
    if piece_title:
        title = f"{piece_title} — " + title
    if composer_name:
        title += f" — {composer_name}"
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        out = buf.getvalue()
        plt.close(fig)
        return out
    except Exception:
        plt.close(fig)
        return None
