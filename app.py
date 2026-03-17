"""
Symphonia Studio: AI-powered classical music discovery and visualization.
Flask app: API + web UI for full workflow (match → top3 → fun facts → guided listening → landscape → instruments).
"""

import io
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, render_template_string, send_file, redirect, Response
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

from utils.retrieval import (
    match_pop_to_composer,
    get_pieces_for_composer,
    get_composer_name,
    get_piece_by_id,
    get_raw_facts_for_fun_facts,
    get_audio_info,
)
from recommendation.ranking import user_taste_from_step1, select_diverse_top_k
from recommendation.explanations import build_explanations
from utils.plotting import (
    plot_emotion_landscape_to_bytes,
    render_emotion_landscape_terrain_to_bytes,
    render_instrument_landscape_to_bytes,
)
from analysis.structure_analysis import get_structure_timeline, structure_timeline_for_llm
from analysis.emotion_analysis import get_emotion_journey
from analysis.instrument_analysis import get_instrument_interaction
from analysis.landscape_pipeline import (
    compute_emotional_landscape,
    compute_instrument_landscape,
    compute_orchestra_explorer_activity,
)
from analysis.audio_features import get_audio_path_for_piece

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["JSON_AS_ASCII"] = False

# ----- Optional LLM: match unknown artist, top3 explanation, fun facts, guided listening, emotion -----
def _llm_match_artist(artist_input: str):
    try:
        from llm.nemotron_client import complete_json
        from llm.prompts import MATCH_ARTIST_TO_COMPOSER
        out = complete_json(
            "You output only valid JSON. No markdown.",
            MATCH_ARTIST_TO_COMPOSER.format(artist_input=artist_input or ""),
            temperature=0.6,
        )
        return out
    except Exception:
        return None


def _llm_top3_explanation(composer_name: str, match_reason: str, pieces: list):
    try:
        from llm.nemotron_client import complete_json
        from llm.prompts import TOP3_PIECES_EXPLANATION
        pieces_list = "\n".join(
            f"- {p.get('piece_id', '')}: {p.get('title', '')} ({p.get('subtitle', '')})"
            for p in pieces
        )
        out = complete_json(
            "You output only valid JSON. No markdown.",
            TOP3_PIECES_EXPLANATION.format(
                composer_name=composer_name,
                match_reason=match_reason,
                pieces_list=pieces_list,
            ),
            temperature=0.6,
        )
        return out
    except Exception:
        return None


def _llm_fun_facts(raw_facts: list):
    if not raw_facts:
        return {"facts": ["This piece has a rich history—explore it with your ears!"]}
    try:
        from llm.nemotron_client import complete_json
        from llm.prompts import FUN_FACTS_GENERATION
        out = complete_json(
            "You output only valid JSON. No markdown.",
            FUN_FACTS_GENERATION.format(raw_facts="\n".join(f"- {f}" for f in raw_facts)),
            temperature=0.7,
        )
        return out or {"facts": raw_facts[:3]}
    except Exception:
        return {"facts": raw_facts[:3]}


def _llm_guided_listening(piece_title: str, composer_name: str, piece_id: str):
    struct = get_structure_timeline(piece_id)
    piece = get_piece_by_id(piece_id)
    duration = (
        (struct.get("duration_seconds") if struct else None)
        or (piece.get("duration_seconds") if piece else None)
    ) or 300
    timeline = structure_timeline_for_llm(piece_id) if struct else None

    try:
        from llm.nemotron_client import complete_json
        from llm.prompts import GUIDED_LISTENING_STRUCTURE, GUIDED_LISTENING_DURATION_ONLY
        if timeline:
            prompt = GUIDED_LISTENING_STRUCTURE.format(
                piece_title=piece_title,
                composer_name=composer_name,
                duration_seconds=duration,
                structure_timeline=timeline,
            )
        else:
            prompt = GUIDED_LISTENING_DURATION_ONLY.format(
                piece_title=piece_title,
                composer_name=composer_name,
                duration_seconds=duration,
            )
        out = complete_json(
            "You output only valid JSON. No markdown.",
            prompt,
            temperature=0.5,
        )
        if out and isinstance(out.get("sections"), list):
            for s in out["sections"]:
                if "description" not in s:
                    s["description"] = ""
            return out["sections"]
    except Exception:
        pass
    return _default_guided_sections(piece_id)


def _default_guided_sections(piece_id: str):
    struct = get_structure_timeline(piece_id)
    piece = get_piece_by_id(piece_id)
    duration = (
        (struct.get("duration_seconds") if struct else None)
        or (piece.get("duration_seconds") if piece else None)
    ) or 300

    if not struct or not struct.get("sections"):
        # No structure data: use a small number of listener-friendly segments
        step = max(30, duration // 4)
        return [
            {"label": "Opening", "description": "The piece introduces its mood and main idea.", "start_sec": 0, "end_sec": min(step, duration), "cues": ["Notice how the piece begins—what mood do you get?", "Listen for the main melody when it first appears."]},
            {"label": "Theme A", "description": "The main melody is in the foreground.", "start_sec": step, "end_sec": min(2 * step, duration), "cues": ["Does the melody feel like it's floating or moving forward?", "Listen for the accompaniment underneath."]},
            {"label": "Contrast or build", "description": "Something shifts—texture, energy, or mood.", "start_sec": 2 * step, "end_sec": min(3 * step, duration), "cues": ["Does the texture or dynamics change?", "Compare this to the opening."]},
            {"label": "Return and ending", "description": "The music comes back or winds down.", "start_sec": 3 * step, "end_sec": duration, "cues": ["Does a familiar idea return?", "Notice how the piece closes."]},
        ]

    sections = []
    for name, start, end in struct["sections"]:
        if end <= start:
            continue
        sections.append({
            "label": name.replace("_", " ").title(),
            "description": "Notice what changes in this section—melody, texture, or mood.",
            "start_sec": start,
            "end_sec": end,
            "cues": [
                "Notice what changes in this section.",
                "Listen for melody, then accompaniment.",
            ],
        })
    return sections


# ----- API routes -----

@app.route("/api/match", methods=["GET", "POST"])
def api_match():
    """Match pop artist to classical composer. POST/GET: artist=..."""
    artist = (request.json or {}).get("artist") if request.is_json else request.args.get("artist", "")
    artist = (artist or "").strip()
    result = match_pop_to_composer(artist)
    # If unknown, optionally call LLM to still suggest a composer
    if result.get("suggestion") and artist:
        llm = _llm_match_artist(artist)
        if isinstance(llm, dict) and llm.get("composer_id"):
            result["composer_id"] = llm.get("composer_id")
            result["composer_name"] = llm.get("composer_name", result["composer_name"])
            result["reason"] = llm.get("reason", result["reason"])
    return jsonify(result)


def _gather_fun_facts_for_pieces(piece_ids: list[str], composer_id: str, use_llm: bool = True) -> dict[str, list[str]]:
    """Fetch or generate fun facts for each piece. Returns piece_id -> list of fact strings."""
    result: dict[str, list[str]] = {}
    for piece_id in piece_ids:
        raw = get_raw_facts_for_fun_facts(piece_id, composer_id)
        if use_llm and raw:
            data = _llm_fun_facts(raw)
            result[piece_id] = (data or {}).get("facts", raw[:3])
        else:
            result[piece_id] = raw[:3] if raw else ["A beloved piece from the repertoire—worth discovering."]
    return result


@app.route("/api/top3", methods=["GET"])
def api_top3():
    """
    Step 2: Recommend top 3 pieces (fine matching).
    Input: composer_id, match_reason from Step 1; optional taste (JSON) for user profile.
    Output: Ranked, diverse top 3 with intro and per-piece tagline + fun_fact.
    """
    composer_id = request.args.get("composer_id", "")
    match_reason = request.args.get("match_reason", "")
    use_llm = request.args.get("use_llm", "1") == "1"
    taste_param = request.args.get("taste", "")

    composer_name = get_composer_name(composer_id)
    candidates = get_pieces_for_composer(composer_id)
    if not candidates:
        return jsonify({"pieces": [], "intro": "", "composer_name": composer_name})

    step1_result = {"reason": match_reason, "composer_id": composer_id}
    if taste_param:
        try:
            import json as _json
            step1_result["style_profile"] = _json.loads(taste_param)
        except Exception:
            pass
    user_taste = user_taste_from_step1(step1_result)
    top3 = select_diverse_top_k(candidates, user_taste, k=3)

    fun_facts_by_id = _gather_fun_facts_for_pieces(
        [p.get("piece_id", "") for p in top3],
        composer_id,
        use_llm=use_llm,
    )
    explained = build_explanations(top3, composer_name, match_reason, fun_facts_by_id)

    for p in explained["pieces"]:
        pid = p.get("piece_id", "")
        audio = get_audio_info(pid)
        raw_url = audio.get("audio_url", "")
        # Use proxy for external URLs (e.g. Google Drive) so playback works despite CORS/redirects
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            p["audio_url"] = f"/api/audio/{pid}"
        else:
            p["audio_url"] = raw_url
        p["cover_art_url"] = audio.get("cover_art_url", "")
        p["availability_status"] = audio.get("availability_status", "unavailable")

    return jsonify({
        "pieces": explained["pieces"],
        "intro": explained["intro"],
        "composer_name": composer_name,
    })


@app.route("/api/audio/<piece_id>")
def api_audio_stream(piece_id):
    """Stream audio for a piece. Proxies external URLs (e.g. Google Drive) so the browser can play them."""
    audio = get_audio_info(piece_id)
    raw_url = (audio or {}).get("audio_url", "")
    if not raw_url:
        return jsonify({"error": "No audio URL for this piece"}), 404
    if raw_url.startswith("http://") or raw_url.startswith("https://"):
        try:
            resp = urlopen(raw_url, timeout=30)
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "audio/mpeg").split(";")[0].strip()
            if not content_type or content_type == "text/html":
                content_type = "audio/mpeg"
            return Response(data, mimetype=content_type)
        except (URLError, HTTPError, OSError) as e:
            return jsonify({"error": "Failed to fetch audio", "detail": str(e)}), 502
    # Local path: redirect so Flask serves from static
    if raw_url.startswith("/"):
        return redirect(raw_url)
    return jsonify({"error": "Invalid audio URL"}), 400


@app.route("/api/fun-facts", methods=["GET"])
def api_fun_facts():
    """Get fun facts for a piece. Optional: use_llm=1."""
    piece_id = request.args.get("piece_id", "")
    use_llm = request.args.get("use_llm", "1") == "1"
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"facts": []})
    composer_id = piece.get("composer_id", "")
    raw = get_raw_facts_for_fun_facts(piece_id, composer_id)
    if use_llm:
        data = _llm_fun_facts(raw)
    else:
        data = {"facts": raw[:3]}
    return jsonify(data)


@app.route("/api/guided-listening", methods=["GET"])
def api_guided_listening():
    """Get section timeline and listening cues for a piece."""
    piece_id = request.args.get("piece_id", "")
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"sections": [], "duration_seconds": 0})
    sections = _llm_guided_listening(
        piece.get("title", ""),
        piece.get("composer_name", ""),
        piece_id,
    )
    struct = get_structure_timeline(piece_id)
    duration = (
        (struct.get("duration_seconds") if struct else None)
        or piece.get("duration_seconds")
    ) or 0
    return jsonify({"sections": sections, "duration_seconds": duration})


@app.route("/api/emotion-landscape", methods=["GET"])
def api_emotion_landscape():
    """Get emotion journey for visualization. Optional: use_llm=1."""
    piece_id = request.args.get("piece_id", "")
    use_llm = request.args.get("use_llm", "1") == "1"
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"journey": []})
    journey = get_emotion_journey(
        piece_id,
        piece.get("title", ""),
        piece.get("composer_name", ""),
        use_llm=use_llm,
    )
    return jsonify({"journey": journey, "duration_seconds": piece.get("duration_seconds", 0)})


@app.route("/api/emotion-landscape/image", methods=["GET"])
def api_emotion_landscape_image():
    """Return emotion or instrument landscape as PNG (save to device). mode=emotion|instrument."""
    piece_id = request.args.get("piece_id", "")
    mode = (request.args.get("mode") or "emotion").strip().lower()
    if mode not in ("emotion", "instrument"):
        mode = "emotion"
    piece = get_piece_by_id(piece_id)
    if not piece:
        return "", 404
    title = piece.get("title", "")
    composer = piece.get("composer_name", "")
    duration = piece.get("duration_seconds") or 300
    use_llm = request.args.get("use_llm", "1") == "1"
    journey = get_emotion_journey(piece_id, title, composer, use_llm=use_llm)
    audio_path = get_audio_path_for_piece(piece_id)
    section_markers = []
    struct = get_structure_timeline(piece_id)
    if struct and struct.get("sections"):
        for name, start, end in struct["sections"]:
            if end > start:
                section_markers.append({"time_sec": start, "label": name.replace("_", " ").title()})

    if mode == "instrument":
        emotion_data = compute_emotional_landscape(
            piece_id, title, composer, duration,
            journey=journey, section_markers=section_markers or None, audio_path=audio_path,
        )
        inst_data = compute_instrument_landscape(
            piece_id, duration,
            tension_curve=emotion_data.get("tension"),
            times_curve=emotion_data.get("times"),
            has_orchestra=piece.get("has_orchestra", False),
            audio_path=audio_path,
        )
        png = render_instrument_landscape_to_bytes(
            inst_data["times"], inst_data["groups"], duration, title, composer,
        )
        download_name = "symphonia-instrument-landscape.png"
    else:
        landscape_data = compute_emotional_landscape(
            piece_id, title, composer, duration,
            journey=journey, section_markers=section_markers or None, audio_path=audio_path,
        )
        png = render_emotion_landscape_terrain_to_bytes(
            landscape_data["times"], landscape_data["tension"], landscape_data["emotional_tone"],
            duration, title, composer, section_markers=landscape_data.get("section_markers"),
        )
        if not png:
            png = plot_emotion_landscape_to_bytes(journey, title, composer, duration)
        download_name = "symphonia-emotional-landscape.png"
    if not png:
        return "", 404
    return send_file(io.BytesIO(png), mimetype="image/png", download_name=download_name)


@app.route("/api/emotion-landscape/data", methods=["GET"])
def api_emotion_landscape_data():
    """Return emotional landscape time series for frontend (terrain + playhead)."""
    piece_id = request.args.get("piece_id", "")
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"times": [], "tension": [], "emotional_tone": [], "points": [], "duration_seconds": 0, "section_markers": []})
    use_llm = request.args.get("use_llm", "1") == "1"
    journey = get_emotion_journey(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), use_llm=use_llm,
    )
    duration = piece.get("duration_seconds") or 300
    audio_path = get_audio_path_for_piece(piece_id)
    section_markers = []
    struct = get_structure_timeline(piece_id)
    if struct and struct.get("sections"):
        for name, start, end in struct["sections"]:
            if end > start:
                section_markers.append({"time_sec": start, "label": name.replace("_", " ").title()})
    data = compute_emotional_landscape(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), duration,
        journey=journey, section_markers=section_markers or None, audio_path=audio_path,
    )
    return jsonify({
        "times": data["times"],
        "tension": data["tension"],
        "emotional_tone": data["emotional_tone"],
        "points": data.get("points", []),
        "duration_seconds": data["duration_seconds"],
        "section_markers": data.get("section_markers", []),
    })


@app.route("/api/emotion-landscape/instrument-data", methods=["GET"])
def api_emotion_landscape_instrument_data():
    """Return instrument-group activity estimate for frontend (interpretive)."""
    piece_id = request.args.get("piece_id", "")
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"times": [], "groups": [], "duration_seconds": 0, "disclaimer": ""})
    duration = piece.get("duration_seconds") or 300
    use_llm = request.args.get("use_llm", "1") == "1"
    journey = get_emotion_journey(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), use_llm=use_llm,
    )
    audio_path = get_audio_path_for_piece(piece_id)
    emotion_data = compute_emotional_landscape(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), duration,
        journey=journey, audio_path=audio_path,
    )
    inst_data = compute_instrument_landscape(
        piece_id, duration,
        tension_curve=emotion_data.get("tension"),
        times_curve=emotion_data.get("times"),
        has_orchestra=piece.get("has_orchestra", False),
        audio_path=audio_path,
    )
    return jsonify({
        "times": inst_data["times"],
        "groups": inst_data["groups"],
        "duration_seconds": inst_data["duration_seconds"],
        "disclaimer": inst_data.get("disclaimer", ""),
    })


@app.route("/api/orchestra-explorer", methods=["GET"])
def api_orchestra_explorer():
    """
    Orchestra Explorer: time-varying activity per instrument group for real-time visualization.
    Returns groups with id, name, tooltip, activity[] and times[]. Sync with playback to light up icons.
    Placeholder/mock: activity from tension curve + phase/weight. Replace with real source separation when available.
    """
    piece_id = request.args.get("piece_id", "")
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"times": [], "groups": [], "duration_seconds": 0, "disclaimer": ""})
    duration = piece.get("duration_seconds") or 300
    use_llm = request.args.get("use_llm", "1") == "1"
    journey = get_emotion_journey(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), use_llm=use_llm,
    )
    audio_path = get_audio_path_for_piece(piece_id)
    emotion_data = compute_emotional_landscape(
        piece_id, piece.get("title", ""), piece.get("composer_name", ""), duration,
        journey=journey, audio_path=audio_path,
    )
    data = compute_orchestra_explorer_activity(
        piece_id, duration,
        tension_curve=emotion_data.get("tension"),
        times_curve=emotion_data.get("times"),
        has_orchestra=piece.get("has_orchestra", False),
        audio_path=audio_path,
    )
    return jsonify({
        "times": data["times"],
        "groups": data["groups"],
        "duration_seconds": data["duration_seconds"],
        "disclaimer": data.get("disclaimer", ""),
    })


@app.route("/api/instruments", methods=["GET"])
def api_instruments():
    """Get instrument groups and roles for a piece. Optional: use_llm=1."""
    piece_id = request.args.get("piece_id", "")
    use_llm = request.args.get("use_llm", "1") == "1"
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"groups": [], "message": ""})
    has_orchestra = piece.get("has_orchestra", False)
    data = get_instrument_interaction(
        piece_id,
        piece.get("title", ""),
        piece.get("composer_name", ""),
        has_orchestra,
        use_llm=use_llm,
    )
    return jsonify(data)


@app.route("/api/piece/<piece_id>", methods=["GET"])
def api_piece(piece_id):
    """Get full piece metadata."""
    piece = get_piece_by_id(piece_id)
    if not piece:
        return jsonify({"error": "Not found"}), 404
    return jsonify(piece)


# ----- Serve minimal single-page UI -----
INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Symphonia Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d0d0f;
      --surface: #16161a;
      --border: #2a2a30;
      --text: #e8e6e3;
      --muted: #8b8685;
      --accent: #c9a86c;
      --accent-dim: #a68b52;
      --wood-dark: #4a3828;
      --wood-mid: #5c4535;
      --wood-warm: #6d5542;
      --wood-light: #8b6f52;
      --wood-grain-a: rgba(0,0,0,0.06);
      --wood-grain-b: rgba(0,0,0,0.04);
      --wood-grain-c: rgba(255,255,255,0.03);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      min-height: 100vh;
      line-height: 1.5;
      color: var(--text);
      background:
        radial-gradient(
          ellipse 120% 100% at 20% 20%,
          rgba(255,248,240,0.12) 0%,
          transparent 50%
        ),
        repeating-linear-gradient(
          0deg,
          transparent 0px,
          transparent 1px,
          var(--wood-grain-a) 1px,
          var(--wood-grain-a) 2px
        ),
        repeating-linear-gradient(
          0deg,
          transparent 0px,
          transparent 3px,
          var(--wood-grain-b) 3px,
          var(--wood-grain-b) 4px
        ),
        repeating-linear-gradient(
          0deg,
          transparent 0px,
          transparent 47px,
          var(--wood-grain-b) 47px,
          var(--wood-grain-b) 51px
        ),
        repeating-linear-gradient(
          0deg,
          transparent 0px,
          transparent 23px,
          var(--wood-grain-c) 23px,
          var(--wood-grain-c) 24px
        ),
        linear-gradient(
          165deg,
          var(--wood-light) 0%,
          var(--wood-warm) 22%,
          var(--wood-mid) 45%,
          var(--wood-warm) 70%,
          var(--wood-dark) 100%
        );
      background-attachment: fixed;
    }
    h1, h2 { font-family: 'Cormorant Garamond', serif; font-weight: 600; }
    .container { max-width: 720px; margin: 0 auto; padding: 1.5rem; }
    .step { display: none; }
    .step.active { display: block; animation: fade 0.3s ease; }
    @keyframes fade { from { opacity: 0; } to { opacity: 1; } }
    .hero {
      text-align: center;
      padding: 2rem 0 2.5rem;
      border-bottom: 1px solid var(--border);
    }
    .hero h1 { font-size: 2rem; color: var(--accent); margin-bottom: 0.5rem; }
    .hero p { color: var(--muted); font-weight: 300; font-size: 0.95rem; }
    input[type="text"] {
      width: 100%;
      padding: 0.9rem 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text);
      font-size: 1rem;
      font-family: inherit;
    }
    input::placeholder { color: var(--muted); }
    .btn {
      display: inline-block;
      padding: 0.75rem 1.5rem;
      background: var(--accent);
      color: var(--bg);
      border: none;
      border-radius: 8px;
      font-family: inherit;
      font-weight: 500;
      cursor: pointer;
      font-size: 0.95rem;
    }
    .btn:hover { background: var(--accent-dim); }
    .btn.secondary { background: var(--surface); color: var(--accent); border: 1px solid var(--border); }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.25rem;
      margin-bottom: 1rem;
      cursor: pointer;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover { border-color: var(--accent-dim); }
    .card.selected { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .card h3 { font-family: 'Cormorant Garamond', serif; font-size: 1.2rem; margin-bottom: 0.35rem; }
    .card .tagline { color: var(--muted); font-size: 0.9rem; }
    .card .fun-fact { font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }
    .recommendation-cards {
      display: flex;
      flex-wrap: wrap;
      gap: 1.25rem;
      margin-top: 1rem;
      justify-content: center;
    }
    .recommendation-card {
      flex: 1 1 200px;
      min-width: 200px;
      max-width: 260px;
      aspect-ratio: 2.5 / 3.5;
      cursor: pointer;
      opacity: 0;
      transform: translateY(14px);
      transition: opacity 0.45s ease-out, transform 0.45s ease-out;
      perspective: 1000px;
    }
    .recommendation-card--visible {
      opacity: 1;
      transform: translateY(0);
    }
    .recommendation-card:hover .recommendation-card__inner { box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
    .recommendation-card.selected .recommendation-card__inner { box-shadow: 0 0 0 2px var(--accent), 0 6px 20px rgba(0,0,0,0.3); }
    .recommendation-card__inner {
      position: relative;
      width: 100%;
      height: 100%;
      border-radius: 16px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 14px rgba(0,0,0,0.25);
      transition: transform 1s ease-in-out, box-shadow 0.25s;
      transform-style: preserve-3d;
    }
    .recommendation-card--flipped .recommendation-card__inner {
      transform: rotateY(180deg);
    }
    .recommendation-card__back,
    .recommendation-card__front {
      position: absolute;
      inset: 0;
      border-radius: 16px;
      backface-visibility: hidden;
      -webkit-backface-visibility: hidden;
    }
    .recommendation-card__back {
      background: linear-gradient(165deg, #1e1c1a 0%, #151412 100%);
      border: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .recommendation-card__back-num {
      font-family: 'Cormorant Garamond', serif;
      font-size: 3.5rem;
      font-weight: 600;
      color: var(--accent);
      opacity: 0.6;
    }
    .recommendation-card__front {
      background: linear-gradient(165deg, var(--surface) 0%, #1a1918 100%);
      border: 1px solid var(--border);
      padding: 1.25rem 1.35rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transform: rotateY(180deg);
    }
    .recommendation-card__title {
      font-family: 'Cormorant Garamond', serif;
      font-size: 1.2rem;
      font-weight: 600;
      margin: 0 0 0.75rem;
      color: var(--text);
      line-height: 1.3;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.6rem;
    }
    .recommendation-card__fact {
      font-size: 0.875rem;
      color: var(--muted);
      line-height: 1.5;
      margin: 0;
      flex: 1;
    }
    .recommendation-card__fact-label {
      font-weight: 700;
      display: block;
      margin-bottom: 0.35rem;
    }
    .recommendation-card__fact-content {
      display: block;
    }
    .match-result {
      background: var(--surface);
      border-radius: 12px;
      padding: 1.25rem;
      margin: 1.25rem 0;
      border-left: 4px solid var(--accent);
    }
    .match-result .reason { color: var(--muted); margin-top: 0.5rem; font-size: 0.95rem; }
    .suggestion-msg { color: var(--muted); font-size: 0.9rem; margin-top: 0.75rem; }
    .section-cue { margin: 0.35em 0; font-size: 0.9rem; color: var(--muted); }
    .landscape-img { width: 100%; border-radius: 8px; margin-top: 1rem; }
    .instrument-group { margin: 1rem 0; padding: 1rem; background: var(--bg); border-radius: 8px; transition: border-color 0.2s; }
    .instrument-group.active { border-left: 3px solid var(--accent); }
    .instrument-group h4 { color: var(--accent); font-size: 0.95rem; margin-bottom: 0.35rem; }
    .instrument-group p { font-size: 0.9rem; color: var(--muted); }
    .orchestra-explorer { margin-top: 0.5rem; }
    .orchestra-explorer-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 0.75rem; }
    .orchestra-card {
      display: flex; flex-direction: column; align-items: center; padding: 0.75rem 0.5rem;
      background: var(--bg); border-radius: 10px; border: 1px solid var(--border);
      transition: opacity 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease;
      cursor: default;
    }
    .orchestra-card.dim { opacity: 0.4; }
    .orchestra-card.active { opacity: 1; border-color: var(--accent-dim); box-shadow: 0 0 12px rgba(201, 168, 108, 0.25); }
    .orchestra-card.strong { box-shadow: 0 0 18px rgba(201, 168, 108, 0.4); animation: orchestra-pulse 1.5s ease-in-out infinite; }
    @keyframes orchestra-pulse { 0%, 100% { box-shadow: 0 0 18px rgba(201, 168, 108, 0.4); } 50% { box-shadow: 0 0 22px rgba(201, 168, 108, 0.55); } }
    .orchestra-card__icon { width: 40px; height: 40px; margin-bottom: 0.4rem; stroke: currentColor; fill: none; stroke-width: 1.5; stroke-linecap: round; }
    .orchestra-card__label { font-size: 0.75rem; color: var(--text); text-align: center; line-height: 1.2; }
    .orchestra-explorer-disclaimer { font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; }
    .nav-row { display: flex; gap: 0.75rem; margin-top: 1.5rem; flex-wrap: wrap; }
    .step-back {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.6rem 1rem;
      margin-bottom: 1.25rem;
      color: var(--accent);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: inherit;
      font-size: 0.95rem;
      font-weight: 500;
      cursor: pointer;
      transition: border-color 0.2s, color 0.2s;
    }
    .step-back:hover {
      color: var(--text);
      border-color: var(--accent-dim);
    }
    .step-back .arrow {
      width: 0;
      height: 0;
      border-top: 5px solid transparent;
      border-bottom: 5px solid transparent;
      border-right: 8px solid currentColor;
    }
    .guided-timeline { margin: 1rem 0; display: flex; flex-wrap: wrap; gap: 6px; }
    .timeline-segment {
      height: 36px; padding: 0 0.85rem; border-radius: 8px; background: var(--surface);
      display: flex; align-items: center; font-size: 0.9rem; border: 1px solid var(--border);
      transition: background 0.2s, border-color 0.2s, color 0.2s;
    }
    .timeline-segment.active { background: var(--accent-dim); color: var(--bg); border-color: var(--accent); }
    .guided-current-section {
      margin-top: 1.25rem; padding: 1rem 1.1rem; border-radius: 10px; background: var(--surface);
      border: 1px solid var(--border);
    }
    .guided-section-title { margin: 0 0 0.35em; font-size: 1.05rem; font-weight: 600; }
    .guided-section-desc { margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--text); line-height: 1.4; }
    .guided-cues-list { margin: 0; padding-left: 1.2rem; font-size: 0.9rem; color: var(--muted); line-height: 1.5; }
    .guided-cues-list li { margin: 0.35em 0; }
    .landscape-intro { font-size: 0.9rem; color: var(--muted); margin-bottom: 0.75rem; font-style: italic; }
    .landscape-canvas-wrap { position: relative; width: 100%; max-width: 800px; margin: 0 auto; border-radius: 10px; overflow: hidden; background: var(--surface); border: 1px solid var(--border); }
    .landscape-canvas-wrap canvas { display: block; max-width: 100%; height: auto; }
    .landscape-save-row { margin-top: 0.75rem; display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
    .recommendation-card__actions { margin-top: 0.75rem; display: flex; align-items: center; gap: 0.5rem; }
    .audio-btn {
      display: inline-flex; align-items: center; justify-content: center;
      width: 2.5rem; height: 2.5rem; border-radius: 50%;
      border: 1px solid var(--border); background: var(--bg);
      color: var(--accent); cursor: pointer; transition: background 0.2s, border-color 0.2s;
    }
    .audio-btn:hover:not(:disabled) { background: var(--surface); border-color: var(--accent-dim); }
    .audio-btn:disabled { opacity: 0.5; cursor: not-allowed; color: var(--muted); }
    .audio-btn.loading { pointer-events: none; opacity: 0.8; }
    .audio-unavailable { font-size: 0.8rem; color: var(--muted); }
    .audio-player-bar {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: var(--surface); border-top: 1px solid var(--border);
      padding: 0.75rem 1rem; display: none; align-items: center; gap: 1rem;
      z-index: 100; max-width: 100%;
    }
    .audio-player-bar.visible { display: flex; flex-wrap: wrap; }
    .audio-player-bar .now-playing { flex: 0 0 auto; font-size: 0.9rem; color: var(--muted); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .audio-player-bar .controls { display: flex; align-items: center; gap: 0.5rem; }
    .audio-player-bar .progress-wrap { flex: 1; min-width: 120px; display: flex; align-items: center; gap: 0.5rem; }
    .audio-player-bar .progress-wrap input[type="range"] { flex: 1; accent-color: var(--accent); }
    .audio-player-bar .time { font-size: 0.8rem; color: var(--muted); min-width: 4rem; text-align: right; }
    .audio-player-bar .volume-wrap { display: flex; align-items: center; gap: 0.25rem; }
    .audio-player-bar .volume-wrap input[type="range"] { width: 70px; accent-color: var(--accent); }
    .audio-player-bar .error-msg { font-size: 0.85rem; color: #c66; }
    .audio-player-bar .audio-disclaimer { width: 100%; margin: 0.5rem 0 0; padding: 0; font-size: 0.8rem; color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <header class="hero">
      <h1>Symphonia Studio</h1>
      <p>Discover classical music through the artists you already love</p>
    </header>

    <!-- Step 1: Enter artist -->
    <section id="step1" class="step active">
      <h2 style="margin-bottom: 1rem;">Who's your favorite artist?</h2>
      <input type="text" id="artistInput" placeholder="e.g. Coldplay, Taylor Swift, Billie Eilish…" autocomplete="off">
      <div class="nav-row">
        <button class="btn" onclick="doMatch()">Find my composer</button>
      </div>
    </section>

    <!-- Step 2: Match result + Top 3 pieces -->
    <section id="step2" class="step">
      <button type="button" class="step-back" onclick="showStep(1)" aria-label="Back to artist">
        <span class="arrow"></span> Back
      </button>
      <div id="matchResult" class="match-result"></div>
      <p id="suggestionMsg" class="suggestion-msg" style="display:none;"></p>
      <h2 style="margin: 1.5rem 0 0.5rem;">Top 3 pieces for you</h2>
      <p id="top3Intro" style="color: var(--muted); margin-bottom: 1.25rem; font-size: 0.95rem;"></p>
      <div id="pieceCards" class="recommendation-cards" aria-live="polite"></div>
      <p class="audio-disclaimer-step2" style="margin-top: 1.25rem; margin-bottom: 0.75rem; font-size: 0.85rem; color: var(--muted);">Due to time limit and copyright, some audios are unavailable at this time.</p>
      <div class="nav-row">
        <button class="btn secondary" onclick="showStep(1)">Change artist</button>
      </div>
    </section>

    <!-- Step 3: Guided listening + Landscape + Instruments -->
    <section id="step3" class="step">
      <button type="button" class="step-back" onclick="showStep(2)" aria-label="Back to piece selection">
        <span class="arrow"></span> Back
      </button>
      <h2 id="pieceTitle" style="margin-bottom: 0.5rem;"></h2>
      <p id="pieceSubtitle" style="color: var(--muted); font-size: 0.95rem;"></p>
      <div class="nav-row" style="margin-top: 0.5rem;">
        <button type="button" class="audio-btn" id="step3PlayBtn" style="width: auto; padding: 0 1rem;" aria-label="Play this piece">&#9658; Listen</button>
      </div>

      <h3 style="margin: 1.5rem 0 0.75rem;">Guided listening</h3>
      <p style="color: var(--muted); font-size: 0.9rem;">Follow the music: the timeline highlights where you are. Use the cues below as a friendly guide.</p>
      <div id="timeline" class="guided-timeline" aria-label="Section timeline"></div>
      <div id="guidedCurrentSection" class="guided-current-section" aria-live="polite">
        <h4 id="guidedSectionTitle" class="guided-section-title"></h4>
        <p id="guidedSectionDescription" class="guided-section-desc"></p>
        <ul id="guidedSectionCues" class="guided-cues-list"></ul>
      </div>

      <h3 style="margin: 1.5rem 0 0.75rem;">Emotional Music Landscape</h3>
      <p class="landscape-intro">This is an emotional journey you might experience while listening.</p>
      <div id="journeyChart" class="landscape-canvas-wrap" style="min-height: 200px;">
        <canvas id="landscapeCanvas" width="800" height="240" aria-label="Emotional landscape"></canvas>
      </div>
      <div class="landscape-save-row">
        <a id="downloadLandscape" href="#" class="btn secondary" download="symphonia-landscape.png" style="display:none;">Save as image</a>
      </div>

      <h3 style="margin: 1.5rem 0 0.75rem;">Orchestra Explorer</h3>
      <p style="color: var(--muted); font-size: 0.9rem;">See which instrument groups are active as the music plays. Icons light up when that part of the orchestra is in the foreground.</p>
      <div id="orchestraExplorer" class="orchestra-explorer">
        <div id="orchestraExplorerGrid" class="orchestra-explorer-grid"></div>
        <p id="orchestraExplorerDisclaimer" class="orchestra-explorer-disclaimer"></p>
      </div>

      <div class="nav-row">
        <button class="btn secondary" onclick="showStep(2)">Choose another piece</button>
      </div>
    </section>
  </div>

  <audio id="globalAudio" preload="metadata"></audio>
  <div id="audioPlayerBar" class="audio-player-bar" aria-label="Audio player">
    <span class="now-playing" id="audioNowPlaying"></span>
    <div class="controls">
      <button type="button" class="audio-btn" id="audioPrev" aria-label="Previous">&#9664;</button>
      <button type="button" class="audio-btn" id="audioPlayPause" aria-label="Play">&#9658;</button>
      <button type="button" class="audio-btn" id="audioNext" aria-label="Next">&#9654;</button>
    </div>
    <div class="progress-wrap">
      <input type="range" id="audioProgress" min="0" max="100" value="0" step="0.1" aria-label="Seek">
      <span class="time" id="audioTime">0:00 / 0:00</span>
    </div>
    <div class="volume-wrap">
      <span aria-hidden="true">&#9834;</span>
      <input type="range" id="audioVolume" min="0" max="100" value="80" aria-label="Volume">
    </div>
    <p class="error-msg" id="audioError" style="display:none;"></p>
    <p class="audio-disclaimer">Due to time limit and copyright, some audios are unavailable at this time.</p>
  </div>

  <script>
    const CARD_REVEAL_DELAY_MS = 400;
    const CARD_FLIP_DELAY_MS = 550;

    let state = {
      artist: '', composerId: '', composerName: '', matchReason: '', pieces: [], selectedPieceId: null,
      landscapeData: null,
      landscapeAudioAvailable: false
    };

    const audioState = {
      currentIndex: -1,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      volume: 0.8,
      loading: false,
      error: null
    };

    function formatTime(sec) {
      if (!Number.isFinite(sec) || sec < 0) return '0:00';
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return m + ':' + (s < 10 ? '0' : '') + s;
    }

    function getPlaylist() { return state.pieces || []; }

    function updatePlayerUI() {
      const bar = document.getElementById('audioPlayerBar');
      const nowEl = document.getElementById('audioNowPlaying');
      const timeEl = document.getElementById('audioTime');
      const progress = document.getElementById('audioProgress');
      const playBtn = document.getElementById('audioPlayPause');
      const errEl = document.getElementById('audioError');
      const playlist = getPlaylist();
      const piece = playlist[audioState.currentIndex];

      if (!piece || playlist.length === 0) {
        bar.classList.remove('visible');
        return;
      }
      bar.classList.add('visible');
      errEl.style.display = audioState.error ? 'block' : 'none';
      errEl.textContent = audioState.error || '';
      nowEl.textContent = piece.title || 'Now playing';
      timeEl.textContent = formatTime(audioState.currentTime) + ' / ' + formatTime(audioState.duration);
      progress.value = audioState.duration ? (audioState.currentTime / audioState.duration) * 100 : 0;
      playBtn.textContent = audioState.isPlaying ? '\u275A\u275A' : '\u25B6';
      playBtn.disabled = !!audioState.error;
      document.querySelectorAll('#pieceCards .audio-btn[data-piece-id]').forEach(btn => {
        const pid = btn.dataset.pieceId;
        const isCurrent = piece.piece_id === pid;
        btn.disabled = btn.dataset.available !== 'true';
        btn.classList.toggle('loading', isCurrent && audioState.loading);
        btn.textContent = (isCurrent && audioState.isPlaying) ? '\u275A\u275A' : '\u25B6';
      });
      const step3Btn = document.getElementById('step3PlayBtn');
      if (step3Btn && state.selectedPieceId === piece.piece_id) {
        step3Btn.textContent = audioState.isPlaying ? '\u275A\u275A Pause' : '\u25B6 Listen';
      }
    }

    function updateGuidedCurrentSection(timeSec) {
      const sections = state.guidedSections || [];
      const titleEl = document.getElementById('guidedSectionTitle');
      const descEl = document.getElementById('guidedSectionDescription');
      const cuesEl = document.getElementById('guidedSectionCues');
      if (!titleEl || !descEl || !cuesEl) return;
      let section = null;
      for (let i = 0; i < sections.length; i++) {
        const s = sections[i];
        const start = parseFloat(s.start_sec);
        const end = parseFloat(s.end_sec);
        if (timeSec >= start && timeSec < end) { section = s; break; }
      }
      if (!section && sections.length) {
        if (timeSec >= (parseFloat(sections[sections.length - 1].end_sec) || 0)) section = sections[sections.length - 1];
        else section = sections[0];
      }
      if (section) {
        titleEl.textContent = section.label || '';
        descEl.textContent = section.description || '';
        descEl.style.display = section.description ? '' : 'none';
        cuesEl.innerHTML = (section.cues || []).map(c => '<li>' + escapeHtml(c) + '</li>').join('');
      } else {
        titleEl.textContent = '';
        descEl.textContent = '';
        cuesEl.innerHTML = '';
      }
    }
    function escapeHtml(s) {
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }
    function setPlaybackIndicator(timeSec) {
      state.audioCurrentTime = timeSec;
      document.querySelectorAll('#timeline .timeline-segment').forEach(seg => {
        const start = parseFloat(seg.dataset.start);
        const end = parseFloat(seg.dataset.end);
        seg.classList.toggle('active', timeSec >= start && timeSec < end);
      });
      updateGuidedCurrentSection(timeSec);
      drawLandscapePlayhead(timeSec);
      updateOrchestraExplorer(timeSec);
    }

    function getOrchestraIcon(groupId) {
      const icons = {
        high_strings: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M12 8 L12 32 M12 12 L28 10 M12 20 L28 18 M12 28 L28 26" stroke-width="1.2"/><ellipse cx="28" cy="10" rx="4" ry="5"/></svg>',
        low_strings: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M14 6 L14 34 M14 10 L26 12 M14 20 L26 22 M14 30 L26 28" stroke-width="1.4"/><ellipse cx="26" cy="12" rx="5" ry="6"/></svg>',
        woodwinds: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><rect x="8" y="16" width="24" height="4" rx="1"/><circle cx="14" cy="18" r="1.5"/><circle cx="20" cy="18" r="1.5"/><circle cx="26" cy="18" r="1.5"/></svg>',
        brass: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M10 20 Q20 8 30 20 Q20 32 10 20" stroke-width="1.2"/><path d="M30 20 L36 18" stroke-width="1"/></svg>',
        percussion: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><ellipse cx="20" cy="20" rx="12" ry="6"/><path d="M8 20 L32 20" stroke-width="1"/></svg>',
        piano_solo: '<svg class="orchestra-card__icon" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="14" width="28" height="14" rx="1" stroke-width="1.2"/><path d="M10 14 L10 28 M14 14 L14 28 M18 14 L18 28 M22 14 L22 28 M26 14 L26 28" stroke-width="0.8"/></svg>'
      };
      return icons[groupId] || icons.high_strings;
    }

    function renderOrchestraExplorer() {
      const grid = document.getElementById('orchestraExplorerGrid');
      if (!grid) return;
      const data = state.orchestraExplorerData;
      if (!data || !data.groups || !data.groups.length) {
        grid.innerHTML = '<p style="color: var(--muted); font-size: 0.9rem;">Orchestra activity is not available for this piece.</p>';
        return;
      }
      grid.innerHTML = data.groups.map(g => {
        const tip = (g.tooltip || g.name || '').replace(/"/g, '&quot;');
        return '<div class="orchestra-card" data-orchestra-group-id="' + escapeHtml(g.id) + '" title="' + tip + '">' +
          getOrchestraIcon(g.id) +
          '<span class="orchestra-card__label">' + escapeHtml(g.name || g.id) + '</span></div>';
      }).join('');
    }

    function updateOrchestraExplorer(timeSec) {
      const data = state.orchestraExplorerData;
      if (!data || !data.times || !data.groups || !data.times.length) return;
      const times = data.times;
      const t = Math.max(0, Math.min(timeSec, (data.duration_seconds || times[times.length - 1]) + 1));
      document.querySelectorAll('.orchestra-card').forEach(card => {
        const gid = card.dataset.orchestraGroupId;
        const g = data.groups.find(gr => gr.id === gid);
        if (!g || !g.activity || !Array.isArray(g.activity)) return;
        let a = 0;
        if (t <= times[0]) a = g.activity[0];
        else if (t >= times[times.length - 1]) a = g.activity[g.activity.length - 1];
        else {
          let i = 0;
          while (i < times.length - 1 && times[i + 1] < t) i++;
          const t0 = times[i], t1 = times[i + 1];
          const v0 = g.activity[i], v1 = g.activity[i + 1];
          const frac = (t1 - t0) > 0 ? (t - t0) / (t1 - t0) : 0;
          a = v0 + frac * (v1 - v0);
        }
        const thresh = 0.2;
        card.classList.toggle('dim', a < thresh);
        card.classList.toggle('active', a >= thresh);
        card.classList.toggle('strong', a >= 0.6);
      });
    }

    function landscapeTimeToX(t, duration, width) {
      if (!duration || duration <= 0) return 0;
      return Math.max(0, Math.min(width, (t / duration) * width));
    }
    function emotionToColor(tone) {
      const t = Math.max(0, Math.min(1, tone || 0));
      const h = 260 - t * 260;
      const s = 65;
      const l = 35 + t * 25;
      return 'hsl(' + h + ',' + s + '%,' + l + '%)';
    }
    function drawEmotionLandscape() {
      const canvas = document.getElementById('landscapeCanvas');
      if (!canvas) return;
      const wrap = canvas.parentElement;
      const maxW = 800;
      const minW = 280;
      const aspect = 240 / 800;
      const containerW = wrap && wrap.clientWidth > 0 ? wrap.clientWidth : maxW;
      const w = Math.max(minW, Math.min(containerW, maxW));
      const h = Math.round(w * aspect);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      var bgGrad = ctx.createLinearGradient(0, 0, 0, h);
      bgGrad.addColorStop(0, '#eef2f6');
      bgGrad.addColorStop(0.5, '#f4f4f2');
      bgGrad.addColorStop(1, '#f0eeeb');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);
      if (!state.landscapeAudioAvailable || !state.landscapeData) return;
      const data = state.landscapeData;
      const points = data.points && data.points.length ? data.points : null;
      const times = data.times || [];
      const tension = data.tension || [];
      const tone = data.emotional_tone || tension;
      if (!points && (!times.length || !tension.length)) return;
      const pad = { top: 24, right: 16, bottom: 28, left: 16 };
      const W = w - pad.left - pad.right;
      const H = h - pad.top - pad.bottom;
      const duration = data.duration_seconds || 1;
      var path = [];
      if (points) {
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          const x = pad.left + (p.time / duration) * W;
          const y = pad.top + H - (p.height || 0) * H;
          path.push({ x, y, color: p.color || emotionToColor(p.height) });
        }
      } else {
        for (let i = 0; i < times.length; i++) {
          const x = pad.left + (times[i] / duration) * W;
          const y = pad.top + H - (tension[i] || 0) * H;
          path.push({ x, y, color: emotionToColor(tone[i]) });
        }
      }
      if (path.length === 0) return;
      for (let i = 0; i < path.length - 1; i++) {
        const a = path[i];
        const b = path[i + 1];
        ctx.beginPath();
        ctx.moveTo(a.x, pad.top + H);
        ctx.lineTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.lineTo(b.x, pad.top + H);
        ctx.closePath();
        ctx.fillStyle = a.color;
        ctx.globalAlpha = 0.78;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      for (let i = 1; i < path.length; i++) ctx.lineTo(path[i].x, path[i].y);
      ctx.strokeStyle = 'rgba(0,0,0,0.18)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
      state._landscapeLastDraw = 'emotion';
    }
    function drawInstrumentLandscape() {
      const canvas = document.getElementById('landscapeCanvas');
      const data = state.landscapeInstrumentData;
      if (!canvas || !data || !data.groups || data.groups.length === 0) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.width;
      const h = canvas.height;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.scale(dpr, dpr);
      const pad = { top: 20, right: 16, bottom: 24, left: 16 };
      const W = w - pad.left - pad.right;
      const H = h - pad.top - pad.bottom;
      const duration = data.duration_seconds || 1;
      const times = data.times || [];
      const n = data.groups.length;
      const rowH = H / n;
      ctx.clearRect(0, 0, w, h);
      var bgGrad = ctx.createLinearGradient(0, 0, 0, h);
      bgGrad.addColorStop(0, '#eef2f6');
      bgGrad.addColorStop(0.5, '#f4f4f2');
      bgGrad.addColorStop(1, '#f0eeeb');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, w, h);
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#555';
      data.groups.forEach((grp, idx) => {
        const act = grp.activity || [];
        const y0 = pad.top + idx * rowH;
        const yH = rowH - 4;
        ctx.fillText(grp.name || ('Group ' + (idx + 1)), pad.left, y0 + 10);
        if (act.length === 0) return;
        ctx.beginPath();
        ctx.moveTo(pad.left, y0 + yH);
        for (let i = 0; i < times.length; i++) {
          const x = pad.left + (times[i] / duration) * W;
          const y = y0 + yH - (act[i] || 0) * (yH - 2);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.lineTo(pad.left + W, y0 + yH);
        ctx.closePath();
        ctx.fillStyle = 'rgba(70, 130, 180, 0.5)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 50, 100, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, y0 + yH - (act[0] || 0) * (yH - 2));
        for (let i = 1; i < times.length; i++) {
          const x = pad.left + (times[i] / duration) * W;
          const y = y0 + yH - (act[i] || 0) * (yH - 2);
          ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
      state._landscapeLastDraw = 'instrument';
    }
    function drawLandscapePlayhead(timeSec) {
      const canvas = document.getElementById('landscapeCanvas');
      if (!canvas || !state.landscapeAudioAvailable || !state.landscapeData) return;
      drawEmotionLandscape();
      const data = state.landscapeData;
      const duration = data.duration_seconds || 1;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = canvas.width / dpr;
      const ch = canvas.height / dpr;
      const pad = { left: 16, right: 16, top: 24, bottom: 28 };
      const W = cw - pad.left - pad.right;
      const x = pad.left + (timeSec / duration) * W;
      const ctx = canvas.getContext('2d');
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = 'rgba(200, 80, 80, 0.9)';
      ctx.fillRect(x, 0, 2, ch);
      ctx.restore();
    }

    function playPieceByIndex(index) {
      const playlist = getPlaylist();
      if (index < 0 || index >= playlist.length) return;
      const piece = playlist[index];
      const url = piece.audio_url;
      const status = piece.availability_status;
      if (!url || status !== 'available') {
        audioState.error = 'Audio unavailable';
        audioState.currentIndex = index;
        updatePlayerUI();
        return;
      }
      const audio = document.getElementById('globalAudio');
      audioState.currentIndex = index;
      audioState.error = null;
      audioState.loading = true;
      updatePlayerUI();
      audio.src = url;
      audio.volume = audioState.volume;
      audio.load();
      audio.play().then(() => {
        audioState.loading = false;
        audioState.isPlaying = true;
        updatePlayerUI();
      }).catch(err => {
        audioState.loading = false;
        audioState.error = 'Playback failed';
        updatePlayerUI();
      });
    }

    function togglePlayPause() {
      const audio = document.getElementById('globalAudio');
      if (audioState.error) return;
      if (audioState.isPlaying) {
        audio.pause();
        audioState.isPlaying = false;
      } else {
        if (audioState.currentIndex >= 0) audio.play().catch(() => {});
        else playPieceByIndex(0);
        audioState.isPlaying = true;
      }
      updatePlayerUI();
    }

    function initAudioPlayer() {
      const audio = document.getElementById('globalAudio');
      const progress = document.getElementById('audioProgress');
      progress.addEventListener('input', () => {
        const pct = parseFloat(progress.value) / 100;
        const dur = audio.duration;
        if (Number.isFinite(dur) && dur > 0) audio.currentTime = pct * dur;
      });
      progress.addEventListener('change', () => { audioState.currentTime = audio.currentTime; updatePlayerUI(); });
      audio.addEventListener('timeupdate', () => {
        audioState.currentTime = audio.currentTime;
        audioState.duration = audio.duration;
        updatePlayerUI();
        setPlaybackIndicator(audio.currentTime);
      });
      audio.addEventListener('durationchange', () => { audioState.duration = audio.duration; updatePlayerUI(); });
      audio.addEventListener('ended', () => {
        audioState.isPlaying = false;
        const next = audioState.currentIndex + 1;
        if (next < getPlaylist().length) playPieceByIndex(next);
        else updatePlayerUI();
      });
      audio.addEventListener('error', () => {
        audioState.loading = false;
        audioState.error = 'Failed to load audio';
        updatePlayerUI();
      });
      document.getElementById('audioPlayPause').addEventListener('click', togglePlayPause);
      document.getElementById('audioPrev').addEventListener('click', () => {
        if (audioState.currentIndex <= 0) return;
        playPieceByIndex(audioState.currentIndex - 1);
      });
      document.getElementById('audioNext').addEventListener('click', () => {
        const playlist = getPlaylist();
        if (audioState.currentIndex >= playlist.length - 1) return;
        playPieceByIndex(audioState.currentIndex + 1);
      });
      document.getElementById('audioVolume').addEventListener('input', e => {
        audioState.volume = parseInt(e.target.value, 10) / 100;
        audio.volume = audioState.volume;
      });
    }

    function showStep(n) {
      document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
      document.getElementById('step' + n).classList.add('active');
    }

    function buildRecommendations(pieces) {
      if (!pieces || pieces.length === 0) return Promise.resolve([]);
      if (pieces.every(p => p.fun_fact)) {
        return Promise.resolve(pieces.map(p => ({
          piece_id: p.piece_id,
          title: p.title || '',
          funFact: p.fun_fact || (p.tagline || 'A beloved piece from the repertoire.'),
          audio_url: p.audio_url || '',
          availability_status: p.availability_status || 'unavailable',
          duration_seconds: p.duration_seconds
        })));
      }
      const funFactPromises = pieces.map(p =>
        fetch('/api/fun-facts?piece_id=' + encodeURIComponent(p.piece_id)).then(r => r.json())
      );
      return Promise.all(funFactPromises).then(results =>
        pieces.map((p, i) => {
          const facts = results[i] && results[i].facts;
          const funFact = (facts && facts[0]) ? facts[0] : (p.fun_fact || p.tagline || 'A beloved piece from the repertoire.');
          return {
            piece_id: p.piece_id,
            title: p.title || '',
            funFact: funFact,
            audio_url: p.audio_url || '',
            availability_status: p.availability_status || 'unavailable',
            duration_seconds: p.duration_seconds
          };
        })
      );
    }

    function renderRecommendationCards(recommendations) {
      const container = document.getElementById('pieceCards');
      container.innerHTML = '';
      recommendations.forEach((rec, index) => {
        const card = document.createElement('div');
        card.className = 'recommendation-card';
        card.dataset.pieceId = rec.piece_id;
        card.setAttribute('role', 'button');
        card.tabIndex = 0;
        card.addEventListener('click', () => selectPiece(rec.piece_id));
        card.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectPiece(rec.piece_id); } });

        const inner = document.createElement('div');
        inner.className = 'recommendation-card__inner';

        const back = document.createElement('div');
        back.className = 'recommendation-card__back';
        const backNum = document.createElement('span');
        backNum.className = 'recommendation-card__back-num';
        backNum.textContent = String(index + 1);
        back.appendChild(backNum);

        const front = document.createElement('div');
        front.className = 'recommendation-card__front';
        const titleEl = document.createElement('h3');
        titleEl.className = 'recommendation-card__title';
        titleEl.textContent = rec.title;
        const factEl = document.createElement('p');
        factEl.className = 'recommendation-card__fact';
        const factText = (rec.funFact || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        factEl.innerHTML = '<span class="recommendation-card__fact-label">Fun Facts:</span><span class="recommendation-card__fact-content">' + factText + '</span>';
        front.appendChild(titleEl);
        front.appendChild(factEl);

        const actions = document.createElement('div');
        actions.className = 'recommendation-card__actions';
        const canPlay = rec.availability_status === 'available' && rec.audio_url;
        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'audio-btn';
        playBtn.setAttribute('aria-label', canPlay ? 'Play' : 'Audio unavailable');
        playBtn.dataset.pieceId = rec.piece_id;
        playBtn.dataset.available = canPlay ? 'true' : 'false';
        playBtn.textContent = '\u25B6';
        playBtn.disabled = !canPlay;
        playBtn.addEventListener('click', e => { e.stopPropagation(); e.preventDefault(); if (canPlay) playPieceByIndex(index); });
        actions.appendChild(playBtn);
        if (!canPlay) {
          const unav = document.createElement('span');
          unav.className = 'audio-unavailable';
          unav.textContent = 'Audio unavailable';
          actions.appendChild(unav);
        }
        front.appendChild(actions);

        inner.appendChild(back);
        inner.appendChild(front);
        card.appendChild(inner);
        container.appendChild(card);

        setTimeout(() => card.classList.add('recommendation-card--visible'), index * CARD_REVEAL_DELAY_MS);
        setTimeout(() => card.classList.add('recommendation-card--flipped'), index * CARD_REVEAL_DELAY_MS + CARD_FLIP_DELAY_MS);
      });
    }

    async function doMatch() {
      const artist = document.getElementById('artistInput').value.trim();
      if (!artist) return;
      const r = await fetch('/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ artist })
      }).then(res => res.json());
      state.artist = artist;
      state.composerId = r.composer_id;
      state.composerName = r.composer_name;
      state.matchReason = r.reason || '';
      document.getElementById('matchResult').innerHTML = '<strong>' + r.composer_name + '</strong><p class="reason">' + (r.reason || '') + '</p>';
      document.getElementById('suggestionMsg').style.display = r.suggestion ? 'block' : 'none';
      document.getElementById('suggestionMsg').textContent = r.suggestions_message || '';

      let top3Url = '/api/top3?composer_id=' + encodeURIComponent(state.composerId) + '&match_reason=' + encodeURIComponent(state.matchReason);
      if (r.style_profile && typeof r.style_profile === 'object') {
        try { top3Url += '&taste=' + encodeURIComponent(JSON.stringify(r.style_profile)); } catch (e) {}
      }
      const top3 = await fetch(top3Url).then(res => res.json());
      state.pieces = top3.pieces || [];
      document.getElementById('top3Intro').textContent = top3.intro || 'Pick a piece to explore.';

      const audio = document.getElementById('globalAudio');
      if (audio) { audio.pause(); audio.src = ''; }
      audioState.currentIndex = -1;
      audioState.isPlaying = false;
      audioState.error = null;
      updatePlayerUI();

      const recommendations = await buildRecommendations(state.pieces);
      renderRecommendationCards(recommendations);
      showStep(2);
    }

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initAudioPlayer);
    } else {
      initAudioPlayer();
    }

    async function selectPiece(pieceId) {
      state.selectedPieceId = pieceId;
      document.querySelectorAll('#pieceCards .recommendation-card').forEach(c => { c.classList.toggle('selected', c.dataset.pieceId === pieceId); });
      const piece = state.pieces.find(p => p.piece_id === pieceId) || {};
      document.getElementById('pieceTitle').textContent = piece.title || '';
      document.getElementById('pieceSubtitle').textContent = piece.subtitle || '';

      const step3PlayBtn = document.getElementById('step3PlayBtn');
      const canPlay = piece.availability_status === 'available' && piece.audio_url;
      if (step3PlayBtn) {
        step3PlayBtn.style.display = 'inline-flex';
        step3PlayBtn.disabled = !canPlay;
        step3PlayBtn.onclick = () => {
          if (!canPlay) return;
          const idx = state.pieces.findIndex(p => p.piece_id === pieceId);
          if (idx < 0) return;
          if (audioState.currentIndex === idx && audioState.isPlaying) togglePlayPause();
          else playPieceByIndex(idx);
        };
      }

      const [guided, emotion, landscapeDataRes, orchestraRes] = await Promise.all([
        fetch('/api/guided-listening?piece_id=' + encodeURIComponent(pieceId)).then(r => r.json()),
        fetch('/api/emotion-landscape?piece_id=' + encodeURIComponent(pieceId)).then(r => r.json()),
        fetch('/api/emotion-landscape/data?piece_id=' + encodeURIComponent(pieceId)).then(r => r.json()),
        fetch('/api/orchestra-explorer?piece_id=' + encodeURIComponent(pieceId)).then(r => r.json())
      ]);

      state.guidedSections = guided.sections || [];
      const tl = document.getElementById('timeline');
      tl.innerHTML = state.guidedSections.map((s, i) => '<div class="timeline-segment" data-start="' + s.start_sec + '" data-end="' + s.end_sec + '" data-section-index="' + i + '"><span>' + escapeHtml(s.label || '') + '</span></div>').join('');
      const currentTime = (state.selectedPieceId === (getPlaylist()[audioState.currentIndex] || {}).piece_id) ? (audioState.currentTime || 0) : 0;
      updateGuidedCurrentSection(currentTime);

      state.landscapeAudioAvailable = canPlay;
      state.landscapeData = landscapeDataRes.times && landscapeDataRes.times.length ? landscapeDataRes : null;
      drawEmotionLandscape();
      if (state.landscapeAudioAvailable && state.landscapeData) {
        drawLandscapePlayhead(currentTime);
      }
      const downloadBtn = document.getElementById('downloadLandscape');
      if (canPlay) {
        downloadBtn.style.display = 'inline-block';
        downloadBtn.href = '/api/emotion-landscape/image?piece_id=' + encodeURIComponent(pieceId) + '&mode=emotion';
      } else {
        downloadBtn.style.display = 'none';
      }

      state.orchestraExplorerData = orchestraRes.times && orchestraRes.groups && orchestraRes.groups.length ? orchestraRes : null;
      renderOrchestraExplorer();
      if (state.orchestraExplorerData) {
        updateOrchestraExplorer(currentTime);
        document.getElementById('orchestraExplorerDisclaimer').textContent = orchestraRes.disclaimer || '';
      } else {
        document.getElementById('orchestraExplorerDisclaimer').textContent = '';
      }
      showStep(3);
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
