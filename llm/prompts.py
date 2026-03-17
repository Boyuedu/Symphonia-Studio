"""
Prompt templates for Symphonia Studio LLM pipeline.
All prompts are designed to be general and work with any composer/piece data.
"""

# --- Pop-to-classical matching (used when artist is not in predefined list) ---
MATCH_ARTIST_TO_COMPOSER = """You are a music expert who helps listeners discover classical music based on their pop preferences.

Your task is to match a pop or contemporary artist to ONE classical composer with a similar artistic "vibe".

Use the following reasoning framework.

--------------------------------
Step 1: Identify the artist style
--------------------------------
Estimate the artist's typical characteristics in terms of:

• emotional tone (romantic, melancholic, dreamy, dramatic, etc.)
• atmosphere (intimate, cinematic, atmospheric, dark, minimal)
• musical texture (simple piano, lush orchestration, layered sound)
• energy level (quiet, moderate, powerful)

--------------------------------
Step 2: Compare with composers
--------------------------------

Use the following typical style associations:

dreamy + atmospheric + subtle → Debussy  
intimate + lyrical + emotional → Chopin  
minimal + haunting + quiet → Satie  
dramatic + grand + emotionally intense → Tchaikovsky or Rachmaninoff  
melodic + narrative + romantic → Schubert  
dark + tense + powerful → Shostakovich  

--------------------------------
Step 3: Optional scoring method
--------------------------------

Estimate a match score using these weights:

emotional tone match: 40%  
atmosphere match: 30%  
texture match: 15%  
energy match: 15%

score = 0.4 * tone_score + 0.3 * atmosphere_score + 0.15 * texture_score + 0.15 * energy_score

Select the composer with the best overall match.

--------------------------------
Output format
--------------------------------

Respond with valid JSON only, no markdown or extra text:

{
  "composer_id": "<lowercase_id e.g. debussy, chopin, satie, rachmaninoff, schubert, shostakovich>",
  "composer_name": "<Full name>",
  "reason": "<2-3 sentences explaining why this composer fits fans of the artist. Make it memorable and specific.>"
}

If the artist name is empty, gibberish, or not a real artist, respond with:

{
  " Try again with a different artist.
}
"""

Artist name: {artist_input}
"""

# --- Top 3 pieces: natural language explanation ---
TOP3_PIECES_EXPLANATION = """You are helping a listener choose their first classical piece from a composer they were matched to.

Composer: {composer_name}
The listener was matched to this composer because: {match_reason}

Here are the top 3 recommended pieces (in order). For each, write a single short sentence that:
- Describes the piece's character in plain language (e.g. "dreamy and atmospheric", "light and flowing")
- Is non-academic and inviting
- Is different from the other two (vary the angle: mood, texture, or feel)

Pieces:
{pieces_list}

Respond with valid JSON only:
{
  "intro": "<1 sentence: e.g. 'Here are three pieces that might fit your taste.'>",
  "pieces": [
    {{ "piece_id": "<id>", "tagline": "<one short descriptive sentence>" }},
    {{ "piece_id": "<id>", "tagline": "<one short descriptive sentence>" }},
    {{ "piece_id": "<id>", "tagline": "<one short descriptive sentence>" }}
  ]
}
"""

FUN_FACTS_GENERATION = """
You are a friendly guide making classical music approachable with fun facts.

We have collected the following raw information about the piece and/or composer:
{raw_facts}

Generate exactly 3 short fun facts for the listener.

Requirements:
- Use only the information contained in the raw facts above.
- Do not introduce outside facts.
- Do not copy the raw facts verbatim.
- Rewrite them into more vivid, interesting, listener-friendly language.
- It is okay to combine closely related raw facts into one stronger fun fact.
- Each fact should feel different in style or angle (for example: historical, surprising, cultural, romantic, funny).
- Keep each fact to 1–2 sentences.
- Tone: engaging, curious, warm, and non-academic.
- Avoid jargon and avoid sounding like a textbook.
- Do not repeat the same idea in different words.

Respond with valid JSON only:
{
  "facts": [
    "<first fun fact>",
    "<second fun fact>",
    "<third fun fact>"
  ]
}
"""

# --- Guided listening: section labels, one-sentence description, and listening cues ---
GUIDED_LISTENING_STRUCTURE = """You are a friendly companion helping someone listen to a classical piece for the first time. Use short, concrete, human-friendly language. Focus on what they can actually hear—not academic or theory-heavy analysis.

Piece: {piece_title}
Composer: {composer_name}
Duration: {duration_seconds} seconds.

The piece has the following rough structure (times in seconds):
{structure_timeline}

For each section, provide:
1. A short section label (e.g. "Opening", "Theme A", "Contrast", "Build-up", "Return", "Ending").
2. One short sentence describing what is happening in this section—what changes or what the listener is hearing.
3. One to three specific listening cues. Cues must point to things the listener can directly hear, e.g.:
   - melody moving higher or lower, or feeling like it's floating
   - texture becoming thicker or thinner
   - accompaniment becoming more flowing or more tense
   - mood becoming brighter, darker, calmer, warmer, heavier, lighter
   - a familiar idea returning in a changed form
   Avoid dense music-theory terms, long paragraphs, and overclaiming emotions as objective fact.

Respond with valid JSON only:
{
  "sections": [
    {
      "label": "<section label>",
      "description": "<one short sentence: what is happening or what changes>",
      "start_sec": <number>,
      "end_sec": <number>,
      "cues": ["<cue 1>", "<cue 2>", "<cue 3>"]
    }
  ]
}
"""

# When we have no pre-defined structure, ask the LLM to divide the piece into listener-friendly sections by time.
GUIDED_LISTENING_DURATION_ONLY = """You are a friendly companion helping someone listen to a classical piece for the first time. Use short, concrete, human-friendly language. Focus on what they can actually hear.

Piece: {piece_title}
Composer: {composer_name}
Duration: {duration_seconds} seconds. There is no pre-defined structure—divide the piece into 4 to 6 listener-friendly sections (e.g. Opening, Theme A, Contrast, Build-up, Return, Ending). Use round numbers for start_sec and end_sec.

For each section, provide:
1. A short section label.
2. One short sentence describing what is happening or what changes in this part of the piece.
3. One to three specific listening cues—things the listener can directly hear (melody, texture, mood, accompaniment, familiar idea returning). No academic jargon.

Respond with valid JSON only:
{
  "sections": [
    {
      "label": "<section label>",
      "description": "<one short sentence>",
      "start_sec": <number>,
      "end_sec": <number>,
      "cues": ["<cue 1>", "<cue 2>", "<cue 3>"]
    }
  ]
}
"""

# --- Emotion landscape: "emotional journey you might experience" ---
EMOTION_LANDSCAPE = """You are describing the emotional journey a listener might experience during a piece—not claiming "the piece's emotion," but an interpretive map.

Piece: {piece_title}
Composer: {composer_name}
Duration: {duration_seconds} seconds.
Structure (approximate): {structure_summary}

Generate 5-8 time points (in seconds) that mark clear emotional shifts or moments. For each point give:
- time_sec: number
- label: short emotion or moment (e.g. "Calm opening", "Building tension", "Peak intensity", "Gentle release")
- description: one short sentence—what you might feel or notice at this moment. Frame as "you might feel..." or "this is a moment of..."

Respond with valid JSON only:
{
  "journey": [
    {{ "time_sec": <number>, "label": "<label>", "description": "<short description>" }},
    ...
  ]
}
"""

# --- Instrument roles / Orchestra conversation (for orchestral pieces) ---
INSTRUMENT_INTERACTION = """You are explaining how different parts of the orchestra contribute to a piece, for a listener who may not distinguish instruments easily.

Piece: {piece_title}
Composer: {composer_name}
This piece uses: {instrument_groups}.

For each instrument group listed, provide:
1. A one-sentence description of its role in this piece (melody, harmony, rhythm, color, dialogue with another section).
2. One specific moment or quality to listen for (e.g. "Listen for the strings when the theme returns—they carry the melody while the piano answers.").

If the piece is piano-only or not orchestral, respond with:
{ "groups": [], "message": "This piece is for piano (or small ensemble) only. The main dialogue is between the hands or between melody and accompaniment." }

Otherwise respond with valid JSON only:
{
  "groups": [
    {
      "name": "<e.g. Strings>",
      "role": "<one sentence>",
      "listen_for": "<one specific cue>"
    }
  ],
  "message": "<optional one sentence: e.g. 'Try muting strings to hear how the winds carry the theme.'>"
}
"""


def fill(template: str, **kwargs) -> str:
    """Safe format: only replace keys that exist; leave others as placeholders if missing."""
    return template.format(**{k: (v if v is not None else "{" + k + "}") for k, v in kwargs.items()})
