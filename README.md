# Symphonia Studio

AI-powered classical music discovery and visualization: from your favorite pop artist to guided listening, emotion landscape, and orchestra explorer.

## Features

- **Pop → Classical matching**: Enter a pop artist; get a matched classical composer and a memorable “why” (with fallback for unknown artists).
- **Top 3 pieces**: Recommended pieces with natural-language taglines and fun facts.
- **Fun facts**: Short, engaging facts (no dry academia), generated from collected info in multiple dimensions.
- **Guided listening**: Timeline (Intro → Theme A/B → Development → Recap) with specific listening cues.
- **Emotion landscape**: “Emotional journey you might experience” as a visual map; save as PNG.
- **Orchestra explorer**: Instrument groups and roles; listening cues for orchestral or piano pieces.

## Setup

```bash
cd symphonia_studio
pip install -r requirements.txt
```

### LLM (optional)

For matching unknown artists, taglines, fun facts, guided cues, and emotion/instrument text, set one of:

- **OpenAI**: `OPENAI_API_KEY`, optionally `OPENAI_BASE_URL`, `OPENAI_MODEL` (default `gpt-4o-mini`).
- **Nemotron / OpenAI-compatible**: `NEMOTRON_API_KEY`, `NEMOTRON_BASE_URL`, `NEMOTRON_MODEL`.

Without an API key, the app still runs using only the data in `data/` (predefined mappings, default composer, and fallback text).

## Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Project layout

```
symphonia_studio/
├── app.py                 # Flask app: API + UI
├── data/
│   ├── artists_to_composers.json
│   ├── composer_pieces.json
│   ├── fun_facts.json
│   ├── piece_metadata.csv
│   └── audio/              # Optional: place audio files or stems here
├── llm/
│   ├── prompts.py          # All LLM prompt templates
│   └── nemotron_client.py  # Generic OpenAI-compatible / Nemotron client
├── analysis/
│   ├── emotion_analysis.py
│   ├── structure_analysis.py
│   └── instrument_analysis.py
├── utils/
│   ├── retrieval.py        # Artist→composer, top3, fun facts data
│   ├── plotting.py        # Emotion landscape PNG
│   └── helpers.py
└── requirements.txt
```

## Extending

- **New artists**: Add entries to `data/artists_to_composers.json` under `mappings` (artist_key, composer_id, reason, weight).
- **New composers/pieces**: Edit `data/composer_pieces.json` and `data/piece_metadata.csv`; add raw facts in `data/fun_facts.json`.
- **Guided listening**: Section times come from `piece_metadata.csv`; cues can be refined in `llm/prompts.py` (GUIDED_LISTENING_STRUCTURE).
- **Orchestra**: For pieces with `has_orchestra: true`, instrument groups and roles are in `analysis/instrument_analysis.py`; add per-piece orchestration in data later if needed.

Design is data-driven and extensible: no hardcoded artist or piece names in matching or recommendation logic.
