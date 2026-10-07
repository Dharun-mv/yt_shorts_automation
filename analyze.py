#!/usr/bin/env python3
"""
YouTube Shorts Content Analyzer with Word-Level Transcriptions,
Multi-Provider LLM Fallbacks, B-Roll Cue Extraction, Karaoke Subtitles,
and SQLite Checkpointing.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TORCH_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import multiprocessing
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

torch.set_num_threads(1)
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


# =====================================================================
# Data Models
# =====================================================================

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None


class SegmentTimestamp(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = Field(default_factory=list)


class ClipSegment(BaseModel):
    clip_id: int = Field(description="Sequential identifier for the clip starting from 1")
    start: float = Field(description="Start timestamp in seconds")
    end: float = Field(description="End timestamp in seconds")
    title: str = Field(description="Viral, high-CTR Shorts title with emoji")
    description: str = Field(description="YouTube Shorts description with #Shorts")


# Backward compatibility alias
ShortsAnalysisResult = ClipSegment



# =====================================================================
# SQLite Checkpointing
# =====================================================================

class CheckpointDB:
    """Manages SQLite checkpoints for transcriptions and LLM analysis."""

    def __init__(self, db_path: str = "checkpoints.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Table for transcription cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcription_checkpoints (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT,
                    file_size INTEGER,
                    full_text TEXT,
                    segments_json TEXT,
                    words_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table for LLM analysis cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_checkpoints (
                    file_hash TEXT PRIMARY KEY,
                    provider TEXT,
                    model TEXT,
                    result_json TEXT,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Computes SHA256 checksum of the audio file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_transcription(self, file_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT full_text, segments_json, words_json FROM transcription_checkpoints WHERE file_hash = ?",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "full_text": row[0],
                    "segments": json.loads(row[1]),
                    "words": json.loads(row[2]),
                }
        return None

    def save_transcription(
        self,
        file_hash: str,
        file_path: str,
        file_size: int,
        full_text: str,
        segments: List[Dict[str, Any]],
        words: List[Dict[str, Any]],
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO transcription_checkpoints
                (file_hash, file_path, file_size, full_text, segments_json, words_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    file_hash,
                    file_path,
                    file_size,
                    full_text,
                    json.dumps(segments),
                    json.dumps(words),
                ),
            )
            conn.commit()

    def get_analysis(self, file_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT result_json FROM analysis_checkpoints WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def save_analysis(
        self,
        file_hash: str,
        provider: str,
        model: str,
        result_data: Any,
        status: str = "success",
    ):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO analysis_checkpoints
                (file_hash, provider, model, result_json, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_hash, provider, model, json.dumps(result_data), status),
            )
            conn.commit()


# =====================================================================
# Whisper Transcription Engine (Single-Threaded CPU, VAD Disabled)
# =====================================================================

def transcribe_audio_file(
    audio_path: str,
    db: CheckpointDB,
    model_size: str = "base",
    force: bool = False,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Transcribes audio using faster-whisper with word-level timestamps.
    Strict single-threaded CPU execution with VAD disabled.
    Utilizes SQLite checkpointing to avoid redundant computation.
    """
    audio_path = os.path.abspath(audio_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_hash = db.calculate_file_hash(audio_path)
    file_size = os.path.getsize(audio_path)

    # Check cached checkpoint
    if not force:
        cached = db.get_transcription(file_hash)
        if cached:
            print(f"[✓] Loaded existing transcription from SQLite checkpoint (Hash: {file_hash[:10]}...)")
            return cached["full_text"], cached["segments"], cached["words"], file_hash

    print(f"[*] Initializing faster-whisper model ({model_size}, single-threaded CPU, VAD disabled)...")
    # Strict single-threaded CPU configuration (cpu_threads=1)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=1)

    print(f"[*] Transcribing '{os.path.basename(audio_path)}' with word-level timestamps (VAD disabled)...")
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=False,
        clip_timestamps="0",
        condition_on_previous_text=False,
    )

    print(f"[+] Detected language: '{info.language}' with probability {info.language_probability:.2f}")

    segments_list = []
    all_words = []
    text_chunks = []

    for seg_idx, segment in enumerate(segments):
        seg_words = []
        if segment.words:
            for w in segment.words:
                word_dict = {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 3) if w.probability is not None else None,
                }
                seg_words.append(word_dict)
                all_words.append(word_dict)

        seg_data = {
            "id": seg_idx,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text.strip(),
            "words": seg_words,
        }
        segments_list.append(seg_data)
        text_chunks.append(segment.text.strip())

    full_text = " ".join(text_chunks)

    # Save to SQLite checkpoint
    db.save_transcription(file_hash, audio_path, file_size, full_text, segments_list, all_words)
    print(f"[✓] Checkpointed transcription to SQLite (Total words: {len(all_words)}, Segments: {len(segments_list)})")

    return full_text, segments_list, all_words, file_hash


# =====================================================================
# Karaoke ASS Subtitle Generation
# =====================================================================

def format_ass_time(seconds: float) -> str:
    """Converts seconds into ASS timestamp format: H:MM:SS.cs"""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int(round((seconds - int(seconds)) * 100))
    if centisecs >= 100:
        secs += 1
        centisecs = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def generate_karaoke_ass_subtitles(
    words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    output_ass_path: str,
    font_name: str = "FreeSans",
    font_size: int = 65,
    max_words_per_line: int = 4,
    max_pause_sec: float = 0.45,
) -> str:
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file using a 'karaoke' effect:
    - The active word being spoken is highlighted in yellow (#00FFFF in ASS format, &H0000FFFF)
      and slightly larger (scale 115%).
    - The rest of the sentence remains white (&H00FFFFFF).

    Args:
        words: List of word dicts with 'word', 'start', 'end'.
        clip_start: Start timestamp of the video clip in seconds.
        clip_end: End timestamp of the video clip in seconds.
        output_ass_path: Output file path for the .ass subtitle.
        font_name: Font family name (default: FreeSans).
        font_size: Base font size.
        max_words_per_line: Max words grouped per subtitle phrase.
        max_pause_sec: Maximum pause duration before splitting phrase.

    Returns:
        str: Absolute path of generated .ass file.
    """
    clip_words = []
    for w in words:
        w_start = float(w["start"])
        w_end = float(w["end"])
        if w_end > clip_start and w_start < clip_end:
            rel_start = max(0.0, w_start - clip_start)
            rel_end = max(rel_start + 0.05, min(w_end - clip_start, clip_end - clip_start))
            clip_words.append({
                "word": w["word"].strip().upper(),
                "start": rel_start,
                "end": rel_end,
            })

    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []

    for w in clip_words:
        if not current_chunk:
            current_chunk.append(w)
            continue

        prev_w = current_chunk[-1]
        pause = w["start"] - prev_w["end"]

        if len(current_chunk) >= max_words_per_line or pause > max_pause_sec:
            chunks.append(current_chunk)
            current_chunk = [w]
        else:
            current_chunk.append(w)

    if current_chunk:
        chunks.append(current_chunk)

    dialogue_lines = []

    for chunk in chunks:
        for i, active_w in enumerate(chunk):
            event_start = active_w["start"]
            if i + 1 < len(chunk):
                event_end = chunk[i + 1]["start"]
            else:
                event_end = active_w["end"]

            event_end = max(event_start + 0.05, event_end)

            # Build line text: active word highlighted yellow (#00FFFF) + enlarged, rest white
            formatted_words = []
            for j, w in enumerate(chunk):
                word_text = w["word"].strip()
                if j == i:
                    formatted_words.append(r"{\c&H0000FFFF&\fscx115\fscy115\b1}" + word_text + r"{\r}")
                else:
                    formatted_words.append(r"{\c&H00FFFFFF&\fscx100\fscy100\b1}" + word_text + r"{\r}")

            dialogue_text = " ".join(formatted_words)
            dialogue_lines.append(
                f"Dialogue: 0,{format_ass_time(event_start)},{format_ass_time(event_end)},Karaoke,,0,0,0,,{dialogue_text}"
            )

    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,5.0,2.5,2,40,40,360,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(dialogue_lines) + "\n"

    Path(output_ass_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_ass_path


# Alias for backward compatibility
generate_dynamic_ass_subtitles = generate_karaoke_ass_subtitles


# =====================================================================
# Multi-Provider LLM Fallback Engine (with B-Roll Cues)
# =====================================================================

class MultiProviderLLM:
    """
    Robust multi-provider LLM client with automatic fallback across:
    1. Google Gemini
    2. Groq (Llama 3.3/3.1)
    3. OpenAI (GPT-4o-mini)
    4. Anthropic (Claude 3.5 Haiku)
    5. OpenRouter
    6. Ollama (Local)
    7. Heuristic fallback (Algorithm)
    """

    def __init__(self):
        self.http_client = httpx.Client(timeout=45.0)

    def _build_prompt(self, full_text: str, segments: List[Dict[str, Any]]) -> str:
        timecoded_summary = []
        for s in segments:
            timecoded_summary.append(f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}")
        timecoded_text = "\n".join(timecoded_summary[:400])

        return f"""You are an elite viral YouTube Shorts editor and social media strategist.
Analyze the following timestamped video transcript and extract 3 to 5 distinct viral segments (each strictly between 30.0 and 60.0 seconds in duration, unless the entire video is shorter than 30s).

STRICT FILTERING RULES:
1. IGNORE HOUSEKEEPING, SPONSORS & LULLS: Strictly ignore housekeeping (intros, channel updates, like/subscribe begging), sponsor reads, brand promotions, and conversational lulls or awkward pauses.
2. STRONG HOOKS: Only select segments that start with strong, high-retention hooks (such as negative curiosity, contrarian paradoxes, shocking counter-intuitive statements, or high-stakes questions).
3. FAST PACING & DIRECTNESS: Dialogue must get to the point quickly without sounding like an essay, rambling context, or unnecessary fluff.
4. CLEAR PAYOFF: Each segment must resolve with a clear payoff, punchline, insight, or actionable takeaway before concluding.
5. DISTINCT & NON-OVERLAPPING: Provide 3 to 5 distinct segments covering separate highlights from across the transcript with no overlapping timestamps.
6. PRECISE TIMESTAMPS: Align 'start' and 'end' timestamps precisely with the provided transcript timestamps.

Transcript with timestamps:
---
{timecoded_text}
---

OUTPUT FORMAT:
Return ONLY a valid JSON array of clip objects strictly matching this schema, with no conversational preamble, markdown wrappers, backticks, or extra commentary:
[
  {{
    "clip_id": 1,
    "start": 15.5,
    "end": 55.0,
    "title": "Why You're Doing This Completely Wrong 🤯 #Shorts",
    "description": "Discover the shocking truth behind this contrarian strategy. #Shorts #Viral #Trending"
  }},
  {{
    "clip_id": 2,
    "start": 120.0,
    "end": 165.5,
    "title": "The Hidden Secret Nobody Talks About 🔥 #Shorts",
    "description": "The exact reason this paradox works every single time. #Shorts #Insight #Tips"
  }}
]"""

    def _extract_json(self, raw_text: str) -> List[Dict[str, Any]]:
        """Safely extracts a JSON array of clip objects from markdown or raw LLM output."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        parsed: Any = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match_arr = re.search(r"(\[.*\])", cleaned, re.DOTALL)
            if match_arr:
                try:
                    parsed = json.loads(match_arr.group(1))
                except json.JSONDecodeError:
                    parsed = None

            if parsed is None:
                match_obj = re.search(r"(\{.*\})", cleaned, re.DOTALL)
                if match_obj:
                    try:
                        parsed = json.loads(match_obj.group(1))
                    except json.JSONDecodeError:
                        parsed = None

            if parsed is None:
                raise ValueError(f"Could not parse valid JSON from LLM response:\n{raw_text}")

        raw_clips: List[Any] = []
        if isinstance(parsed, list):
            raw_clips = parsed
        elif isinstance(parsed, dict):
            for key in ["clips", "segments", "viral_segments", "results", "data", "items"]:
                if key in parsed and isinstance(parsed[key], list):
                    raw_clips = parsed[key]
                    break
            if not raw_clips:
                for val in parsed.values():
                    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        raw_clips = val
                        break
            if not raw_clips:
                raw_clips = [parsed]
        else:
            raise ValueError(f"Unexpected JSON structure from LLM response:\n{raw_text}")

        normalized_clips: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_clips, start=1):
            if not isinstance(item, dict):
                continue
            clip_id = int(item.get("clip_id", item.get("id", idx)))

            start_val = item.get("start", item.get("start_time", 0.0))
            try:
                start_f = float(start_val)
            except (ValueError, TypeError):
                start_f = 0.0

            end_val = item.get("end", item.get("end_time", start_f + 30.0))
            try:
                end_f = float(end_val)
            except (ValueError, TypeError):
                end_f = start_f + 30.0

            if end_f <= start_f:
                end_f = start_f + 30.0

            title = str(item.get("title", f"Viral Moment #{clip_id} 🔥 #Shorts")).strip()
            desc = str(item.get("description", f"Check out this viral moment! #Shorts #Trending")).strip()

            normalized_clips.append({
                "clip_id": clip_id,
                "start": round(start_f, 2),
                "end": round(end_f, 2),
                "title": title,
                "description": desc,
            })

        if not normalized_clips:
            raise ValueError(f"No valid clip objects found in parsed JSON:\n{raw_text}")

        return normalized_clips

    def _call_gemini(self, prompt: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.4,
            },
        }
        res = self.http_client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
        candidate_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return self._extract_json(candidate_text)

    def _call_groq(self, prompt: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an elite video editor and viral strategist. You only output a valid JSON array of viral clip objects conforming to the requested schema, with no markdown or additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        res = self.http_client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def _call_openai(self, prompt: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an elite video editor and viral strategist. You only output a valid JSON array of viral clip objects conforming to the requested schema, with no markdown or additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        res = self.http_client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def _call_anthropic(self, prompt: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment.")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 2048,
            "system": "You are an elite video editor and viral strategist. You only output a valid JSON array of viral clip objects conforming to the requested schema, with no markdown or additional text.",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        res = self.http_client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        content = res.json()["content"][0]["text"]
        return self._extract_json(content)

    def _call_openrouter(self, prompt: str) -> List[Dict[str, Any]]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": "meta-llama/llama-3.3-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an elite video editor and viral strategist. Always output a strict JSON array of clip objects conforming to the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }
        res = self.http_client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return self._extract_json(content)

    def _call_ollama(self, prompt: str) -> List[Dict[str, Any]]:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        url = f"{host.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
        }
        res = self.http_client.post(url, json=payload, timeout=30.0)
        res.raise_for_status()
        content = res.json().get("response", "")
        return self._extract_json(content)

    def _heuristic_fallback(self, full_text: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deterministic rule-based fallback if no LLM APIs are accessible.
        Selects 3 distinct 30-50 second windows across the transcript.
        """
        print("[!] No LLM API keys available or all requests failed. Running rule-based heuristic fallback...")
        if not segments:
            return [
                {
                    "clip_id": 1,
                    "start": 0.0,
                    "end": 35.0,
                    "title": "Unbelievable Moment You Won't Believe! 🤯 #Shorts",
                    "description": "Check out this incredible highlight! Don't forget to like and subscribe! #Shorts #Viral #Trending",
                }
            ]

        total_duration = segments[-1]["end"]
        num_clips = 3
        clip_target_duration = 40.0
        step = max(total_duration / (num_clips + 1), 35.0)

        clips = []
        for i in range(num_clips):
            desired_start = i * step
            if desired_start >= total_duration - 15.0 and i > 0:
                break

            matching_segments = [s for s in segments if s["start"] >= desired_start]
            if not matching_segments:
                matching_segments = [s for s in segments if s["end"] > desired_start]
            if not matching_segments:
                matching_segments = segments

            start_t = matching_segments[0]["start"]
            target_end = min(start_t + clip_target_duration, total_duration)

            end_t = target_end
            for s in matching_segments:
                if s["end"] >= start_t + 30.0:
                    end_t = s["end"]
                    if end_t - start_t >= 38.0 or s["end"] >= start_t + 55.0:
                        break

            clip_text = " ".join(
                s["text"] for s in segments if s["start"] >= start_t and s["end"] <= end_t
            ).strip()
            if not clip_text:
                clip_text = full_text[:200]

            first_words = " ".join(clip_text.split()[:7])
            title = f"{first_words.capitalize()}... 🤯 #Shorts" if first_words else f"Viral Moment #{i+1} 🔥 #Shorts"
            description = f"{clip_text[:120]}...\n\nSubscribe for more! #Shorts #Viral #Trending"

            clips.append({
                "clip_id": i + 1,
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "title": title,
                "description": description,
            })

        return clips

    def analyze(self, full_text: str, segments: List[Dict[str, Any]]) -> List[ClipSegment]:
        prompt = self._build_prompt(full_text, segments)

        providers = [
            ("Google Gemini", self._call_gemini),
            ("Groq", self._call_groq),
            ("OpenAI", self._call_openai),
            ("Anthropic", self._call_anthropic),
            ("OpenRouter", self._call_openrouter),
            ("Ollama", self._call_ollama),
        ]

        errors = []
        for provider_name, func in providers:
            try:
                print(f"[*] Attempting analysis with {provider_name}...")
                raw_clips = func(prompt)
                clips = [ClipSegment(**item) for item in raw_clips]
                print(f"[✓] Successfully extracted {len(clips)} viral clips via {provider_name}!")
                return clips
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")

        print(f"[-] All LLM providers bypassed or failed: {'; '.join(errors)}")
        heuristic_data = self._heuristic_fallback(full_text, segments)
        return [ClipSegment(**item) for item in heuristic_data]


# =====================================================================
# Main Analysis Pipeline
# =====================================================================

def analyze_audio_for_shorts(
    audio_path: str,
    db_path: str = "checkpoints.db",
    model_size: str = "base",
    force_transcribe: bool = False,
    force_llm: bool = False,
    output_json: Optional[str] = None,
    output_ass: Optional[str] = None,
) -> List[ClipSegment]:
    """
    Complete resilient pipeline:
    1. Check SQLite for cached transcription.
    2. Run faster-whisper with word-level timestamps (single-threaded CPU, VAD disabled).
    3. Check SQLite for cached LLM analysis.
    4. Run multi-provider LLM fallback analysis (extracts 3-5 distinct viral segments).
    5. Save JSON array of clip objects to checkpoint and output_json (temp/analysis.json).
    """
    db = CheckpointDB(db_path)

    # Step 1 & 2: Whisper Transcription with SQLite Checkpoint
    full_text, segments, words, file_hash = transcribe_audio_file(
        audio_path=audio_path,
        db=db,
        model_size=model_size,
        force=force_transcribe,
    )

    # Step 3: Check cached analysis
    clips: Optional[List[ClipSegment]] = None
    if not force_llm:
        cached_analysis = db.get_analysis(file_hash)
        if cached_analysis:
            try:
                if isinstance(cached_analysis, list):
                    clips = [ClipSegment(**item) for item in cached_analysis]
                    print(f"[✓] Loaded {len(clips)} existing clips from SQLite checkpoint.")
                elif isinstance(cached_analysis, dict):
                    start_t = float(cached_analysis.get("start", cached_analysis.get("start_time", 0.0)))
                    end_t = float(cached_analysis.get("end", cached_analysis.get("end_time", start_t + 30.0)))
                    clips = [
                        ClipSegment(
                            clip_id=1,
                            start=start_t,
                            end=end_t,
                            title=cached_analysis.get("title", "Viral Short 🔥 #Shorts"),
                            description=cached_analysis.get("description", "Check out this clip! #Shorts"),
                        )
                    ]
                    print(f"[✓] Loaded legacy single-clip analysis from SQLite checkpoint.")
            except Exception:
                clips = None

    if clips is None:
        # Step 4: Multi-Provider LLM Analysis
        llm_engine = MultiProviderLLM()
        clips = llm_engine.analyze(full_text, segments)

        # Step 5: Save Analysis Checkpoint
        clips_data = [c.model_dump() for c in clips]
        db.save_analysis(
            file_hash=file_hash,
            provider="MultiProviderLLM",
            model="standard",
            result_data=clips_data,
            status="success",
        )

    # Step 6: Save JSON array of clip objects to output_json
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clips_json_data = [c.model_dump() for c in clips]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(clips_json_data, f, indent=2)
        print(f"[+] Saved analysis JSON array ({len(clips)} clips) to: {output_json}")

    # Generate Karaoke ASS Subtitles if requested (for the first clip)
    if output_ass and clips:
        generate_karaoke_ass_subtitles(
            words=words,
            clip_start=clips[0].start,
            clip_end=clips[0].end,
            output_ass_path=output_ass,
        )
        print(f"[+] Generated karaoke ASS subtitles for Clip #1 at: {output_ass}")

    return clips


def find_latest_audio(temp_dir: str = "temp") -> str:
    """Finds the most recently modified audio file in the temp folder."""
    temp_path = Path(temp_dir).resolve()
    if not temp_path.exists():
        raise FileNotFoundError(f"Directory '{temp_dir}' does not exist.")
    audio_files = list(temp_path.glob("*.wav")) + list(temp_path.glob("*.mp3")) + list(temp_path.glob("*.m4a"))
    if not audio_files:
        raise FileNotFoundError(f"No audio files (.wav, .mp3, .m4a) found in '{temp_dir}'. Run download.py first!")
    audio_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(audio_files[0])


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with faster-whisper and analyze 3-5 viral Shorts segments with LLM."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Path to the WAV/audio file (defaults to latest audio in temp/).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=os.getenv("WHISPER_MODEL_SIZE", "base"),
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: base).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="checkpoints.db",
        help="SQLite checkpoint database path (default: checkpoints.db).",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Force re-transcription even if checkpoint exists in SQLite.",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force re-running LLM analysis even if checkpoint exists.",
    )
    parser.add_argument(
        "-o", "--output-json",
        type=str,
        default=None,
        help="Optional path to output analysis JSON file (defaults to temp/analysis.json).",
    )
    parser.add_argument(
        "--output-ass",
        type=str,
        default=None,
        help="Optional path to output karaoke .ass subtitle file for Clip #1.",
    )

    args = parser.parse_args()

    try:
        audio_path = args.audio_path or find_latest_audio()
        output_json = args.output_json
        if output_json is None:
            output_json = str(Path("temp") / "analysis.json")

        clips = analyze_audio_for_shorts(
            audio_path=audio_path,
            db_path=args.db_path,
            model_size=args.model_size,
            force_transcribe=args.force_transcribe,
            force_llm=args.force_llm,
            output_json=output_json,
            output_ass=args.output_ass,
        )

        print("\n" + "=" * 60)
        print(f"🎯 VIRAL SHORTS CLIPS EXTRACTED ({len(clips)} clips)")
        print("=" * 60)
        for clip in clips:
            print(f"🎬 Clip #{clip.clip_id}:")
            print(f"   ⏱  Time Range : {clip.start:.2f}s -> {clip.end:.2f}s (Duration: {clip.end - clip.start:.2f}s)")
            print(f"   🔥 Title      : {clip.title}")
            print(f"   📝 Description:\n      {clip.description}")
            print("-" * 60)
        print("=" * 60)

    except Exception as e:
        print(f"\n[!] Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

