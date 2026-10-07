#!/usr/bin/env python3
"""
YouTube Shorts Video Processor with Dynamic Face Tracking, Pixabay B-Roll Overlays,
Karaoke ASS Subtitles, and Looped Background Music Mixing.
- Analyzes video frames using OpenCV and MediaPipe Face Detection.
- Computes a dynamic, smoothly interpolated 9:16 vertical crop window following the speaker.
- Fetches and standardizes 3-second vertical B-roll clips from Pixabay Video API using b_roll_cues keywords.
- Selects a random background music MP3 from the local music/ folder, loops it, reduces volume (0.10),
  and mixes it underneath the spoken audio track.
- Overlays B-roll clips at exact timestamps with FFmpeg and burns karaoke ASS subtitles on top.
- Outputs the finalized YouTube Short to ./ready folder.
"""

import argparse
import io
import json
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
import requests

# Load environment variables
load_dotenv()

# Initialize MediaPipe Solutions / Tasks compatibility
import mediapipe as mp


# =====================================================================
# MediaPipe Face Detection Wrapper
# =====================================================================

class FaceDetectorWrapper:
    """
    Wrapper for MediaPipe Face Detection supporting both legacy solutions
    and modern Tasks API across MediaPipe versions.
    """

    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self.detector = None
        self._mode = "none"

        # Check if mp.solutions.face_detection exists
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
            try:
                self.detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=model_selection,
                    min_detection_confidence=min_confidence,
                )
                self._mode = "solutions"
            except Exception:
                self.detector = None

        # Fallback to Tasks API
        if self.detector is None:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = self._ensure_blazeface_model()
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceDetectorOptions(
                    base_options=base_options,
                    min_detection_confidence=min_confidence,
                )
                self.detector = vision.FaceDetector.create_from_options(options)
                self._mode = "tasks"
            except Exception as e:
                print(f"[!] Warning: Failed to initialize MediaPipe FaceDetector: {e}", file=sys.stderr)
                self._mode = "fallback"

    @staticmethod
    def _ensure_blazeface_model() -> str:
        """Downloads the BlazeFace short range model if not present locally."""
        model_dir = Path("temp")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "blaze_face_short_range.tflite"

        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            )
            print("[*] Downloading MediaPipe BlazeFace detection model...")
            urllib.request.urlretrieve(url, str(model_path))

        return str(model_path)

    def detect_face_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detects the most prominent face in a BGR frame.

        Returns:
            Optional[Tuple[float, float, float, float]]: (center_x, center_y, width, height) in pixels, or None.
        """
        if self._mode == "fallback" or self.detector is None:
            return None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        try:
            if self._mode == "solutions":
                results = self.detector.process(rgb)
                if results and results.detections:
                    best_detection = max(
                        results.detections,
                        key=lambda d: d.location_data.relative_bounding_box.width
                        * d.location_data.relative_bounding_box.height,
                    )
                    rbox = best_detection.location_data.relative_bounding_box
                    cx = (rbox.xmin + rbox.width / 2.0) * w
                    cy = (rbox.ymin + rbox.height / 2.0) * h
                    bw = rbox.width * w
                    bh = rbox.height * h
                    return (cx, cy, bw, bh)

            elif self._mode == "tasks":
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self.detector.detect(mp_img)
                if results and results.detections:
                    best_detection = max(
                        results.detections,
                        key=lambda d: d.bounding_box.width * d.bounding_box.height,
                    )
                    box = best_detection.bounding_box
                    cx = box.origin_x + box.width / 2.0
                    cy = box.origin_y + box.height / 2.0
                    return (cx, cy, float(box.width), float(box.height))

        except Exception:
            pass

        return None

    def close(self):
        if self._mode == "solutions" and hasattr(self.detector, "close"):
            self.detector.close()
        elif self._mode == "tasks" and hasattr(self.detector, "close"):
            self.detector.close()


# Ensure mediapipe.solutions.face_detection is available if requested directly
if not hasattr(mp, "solutions"):
    class SolutionsContainer:
        pass
    mp.solutions = SolutionsContainer()

if not hasattr(mp.solutions, "face_detection"):
    class FaceDetectionModule:
        FaceDetection = FaceDetectorWrapper
    mp.solutions.face_detection = FaceDetectionModule()


# =====================================================================
# Encoder Helper
# =====================================================================

def detect_best_h264_encoder() -> str:
    """Detects the best available H.264 video encoder on the system."""
    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        encoders = res.stdout
        if "h264_nvenc" in encoders:
            test_res = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if test_res.returncode == 0:
                return "h264_nvenc"
        if "libx264" in encoders:
            test_x264 = subprocess.run(
                ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1", "-c:v", "libx264", "-f", "null", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if test_x264.returncode == 0:
                return "libx264"
        if "libopenh264" in encoders:
            return "libopenh264"
    except Exception:
        pass
    return "libopenh264"


# =====================================================================
# Dynamic 9:16 Face-Tracking Crop Engine
# =====================================================================

def calculate_smooth_crop_centers(
    video_path: str,
    start_time: float,
    end_time: float,
    target_aspect: float = 9.0 / 16.0,
    sample_stride: int = 2,
    smoothing_factor: float = 0.08,
) -> Tuple[List[int], int, int, int, int, float]:
    """
    Analyzes video frames with MediaPipe Face Detection to determine a smoothly
    interpolated horizontal crop window (9:16 aspect ratio) tracking the speaker.

    Returns:
        Tuple[List[int], int, int, int, int, float]:
            (smoothed_x_coords, orig_w, orig_h, crop_w, crop_h, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate 9:16 crop window dimensions
    crop_h = orig_h
    crop_w = int(orig_h * target_aspect)
    # Ensure even dimensions for video codecs
    if crop_w % 2 != 0:
        crop_w -= 1

    max_x = max(0, orig_w - crop_w)
    default_x = max_x / 2.0

    start_frame = max(0, int(start_time * fps))
    end_frame = int(end_time * fps)
    total_frames = max(1, end_frame - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detector = FaceDetectorWrapper(min_confidence=0.45)

    raw_target_x: List[float] = []
    last_detected_x = default_x

    print(f"[*] Analyzing {total_frames} frames for speaker face tracking...")

    current_frame_idx = 0
    while current_frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run face detection on sampled frames for high throughput
        if current_frame_idx % sample_stride == 0:
            face_bbox = detector.detect_face_bbox(frame)
            if face_bbox is not None:
                cx, cy, bw, bh = face_bbox
                # Center the 9:16 crop window around the face center X
                desired_x = cx - (crop_w / 2.0)
                # Clamp within frame boundaries
                clamped_x = max(0.0, min(float(max_x), desired_x))
                last_detected_x = clamped_x
            else:
                # Retain last known position with gentle drift towards center if lost
                last_detected_x = 0.95 * last_detected_x + 0.05 * default_x

        raw_target_x.append(last_detected_x)
        current_frame_idx += 1

    cap.release()
    detector.close()

    # Fill if video ended early
    while len(raw_target_x) < total_frames:
        raw_target_x.append(last_detected_x)

    # Apply Exponential Moving Average (EMA) and forward-backward smoothing for cinematic motion
    smoothed_x: List[float] = []
    curr = raw_target_x[0] if raw_target_x else default_x
    for val in raw_target_x:
        curr = curr * (1.0 - smoothing_factor) + val * smoothing_factor
        smoothed_x.append(curr)

    # Backward pass for zero-phase smoothness
    curr_back = smoothed_x[-1]
    for i in range(len(smoothed_x) - 1, -1, -1):
        curr_back = curr_back * (1.0 - smoothing_factor) + smoothed_x[i] * smoothing_factor
        smoothed_x[i] = (smoothed_x[i] + curr_back) / 2.0

    # Convert to integer pixel coordinates clamped to bounds
    int_crop_x = [int(max(0, min(max_x, round(x)))) for x in smoothed_x]

    print(f"[✓] Completed face tracking across {len(int_crop_x)} frames (Range X: {min(int_crop_x)} -> {max(int_crop_x)})")
    return int_crop_x, orig_w, orig_h, crop_w, crop_h, fps


def crop_video_with_face_tracking(
    video_path: str,
    start_time: float,
    end_time: float,
    output_cropped_path: str,
    target_width: int = 1080,
    target_height: int = 1920,
) -> str:
    """
    Slices the video segment, applies the dynamic face-tracking 9:16 crop,
    and encodes the cropped base video with audio intact.
    """
    crop_x_coords, orig_w, orig_h, crop_w, crop_h, fps = calculate_smooth_crop_centers(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
    )

    encoder = detect_best_h264_encoder()
    duration = max(1.0, end_time - start_time)

    # Extract slice audio to temporary AAC file
    temp_audio_path = str(Path(output_cropped_path).with_suffix(".audio.aac"))
    cmd_extract_audio = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", video_path,
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        temp_audio_path,
    ]
    subprocess.run(cmd_extract_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Open video capture and pipe cropped frames directly to FFmpeg
    cap = cv2.VideoCapture(video_path)
    start_frame = max(0, int(start_time * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cmd_ffmpeg_pipe = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{target_width}x{target_height}",
        "-r", f"{fps}",
        "-i", "-",  # Video from stdin
        "-i", temp_audio_path,  # Original audio
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-shortest",
        output_cropped_path,
    ]

    print(f"[*] Rendering dynamic face-tracked 9:16 crop ({target_width}x{target_height})...")
    proc = subprocess.Popen(
        cmd_ffmpeg_pipe,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_idx = 0
    total_to_process = len(crop_x_coords)

    try:
        while frame_idx < total_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            x_offset = crop_x_coords[frame_idx]
            # Crop 9:16 region
            cropped_frame = frame[0:crop_h, x_offset : x_offset + crop_w]
            # Resize to target 1080x1920
            resized_frame = cv2.resize(
                cropped_frame,
                (target_width, target_height),
                interpolation=cv2.INTER_LINEAR,
            )

            # Write raw frame bytes to FFmpeg stdin
            proc.stdin.write(resized_frame.tobytes())
            frame_idx += 1

    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    print(f"[✓] Dynamic face-tracked video rendered: {output_cropped_path}")
    return output_cropped_path


# =====================================================================
# Pixabay Video API B-Roll Fetcher & Standardizer
# =====================================================================

def fetch_pixabay_broll_clip(
    keyword: str,
    output_path: str,
    api_key: Optional[str] = None,
    duration_sec: float = 3.0,
    target_width: int = 1080,
    target_height: int = 1920,
    timeout: int = 20,
) -> Optional[str]:
    """
    Searches Pixabay Video API for video matching keyword, extracts the highest
    resolution MP4 from the first result's videos dictionary, downloads it to temp/,
    and standardizes it to a 3-second 1080x1920 MP4 clip.

    Args:
        keyword: Search term (e.g., 'robot', 'ocean').
        output_path: Final standardized MP4 output path in temp/.
        api_key: Pixabay API key (reads PIXABAY_API_KEY from environment if None).
        duration_sec: Duration in seconds (default: 3.0).
        target_width: Vertical width (default: 1080).
        target_height: Vertical height (default: 1920).
        timeout: HTTP request timeout in seconds.

    Returns:
        Optional[str]: Path to the standardized B-roll clip, or None if failed.
    """
    api_key = api_key or os.getenv("PIXABAY_API_KEY")
    if not api_key:
        print(f"[!] Warning: PIXABAY_API_KEY not found in environment. Skipping B-roll for '{keyword}'.")
        return None

    # Construct request URL and parameters
    encoded_query = urllib.parse.quote_plus(keyword.strip())
    url = f"https://pixabay.com/api/videos/?key={api_key}&q={encoded_query}&video_type=film"

    try:
        print(f"[*] Searching Pixabay for B-roll keyword: '{keyword}'...")
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])

        if not hits:
            # If film video_type had 0 hits, try without video_type filter as fallback
            fallback_url = f"https://pixabay.com/api/videos/?key={api_key}&q={encoded_query}"
            resp_fallback = requests.get(fallback_url, timeout=timeout)
            if resp_fallback.status_code == 200:
                hits = resp_fallback.json().get("hits", [])

        if not hits:
            print(f"[-] No Pixabay videos found for keyword '{keyword}'.")
            return None

        # 3. Extract highest resolution MP4 URL from first result's 'videos' dictionary
        first_hit = hits[0]
        videos_dict = first_hit.get("videos", {})
        if not videos_dict:
            print(f"[-] No videos dictionary found in first Pixabay result for '{keyword}'.")
            return None

        best_download_url = None
        highest_res = -1

        for quality_name, video_data in videos_dict.items():
            if isinstance(video_data, dict) and video_data.get("url"):
                v_w = int(video_data.get("width", 0))
                v_h = int(video_data.get("height", 0))
                res = v_w * v_h
                if res > highest_res:
                    highest_res = res
                    best_download_url = video_data.get("url")

        # Fallback if dimensions weren't populated
        if not best_download_url:
            for q in ["large", "medium", "small", "tiny"]:
                if q in videos_dict and isinstance(videos_dict[q], dict) and videos_dict[q].get("url"):
                    best_download_url = videos_dict[q]["url"]
                    break

        if not best_download_url:
            print(f"[-] No valid video URL found in Pixabay videos dict for '{keyword}'.")
            return None

        print(f"[+] Selected highest resolution Pixabay MP4 ({highest_res} px) for '{keyword}'")

        # Download raw video to temp/
        raw_temp = str(Path(output_path).with_suffix(".raw.mp4"))
        Path(raw_temp).parent.mkdir(parents=True, exist_ok=True)

        with requests.get(best_download_url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(raw_temp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)

        # Standardize video: Exactly duration_sec seconds, vertical 1080x1920, 30fps, no audio
        encoder = detect_best_h264_encoder()
        cmd_standardize = [
            "ffmpeg", "-y",
            "-i", raw_temp,
            "-t", str(duration_sec),
            "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,crop={target_width}:{target_height},setsar=1,fps=30",
            "-c:v", encoder,
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]
        subprocess.run(cmd_standardize, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(raw_temp):
            os.remove(raw_temp)

        print(f"[✓] Downloaded and standardized 3s Pixabay B-roll for '{keyword}': {output_path}")
        return output_path

    except Exception as e:
        print(f"[!] Error fetching Pixabay B-roll for '{keyword}': {e}", file=sys.stderr)
        return None


# Alias for backward compatibility
fetch_pexels_broll_clip = fetch_pixabay_broll_clip


def fetch_all_broll_cues(
    b_roll_cues: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    temp_dir: str = "temp",
    clip_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetches 3-second B-roll clips for all cues in the analysis JSON using Pixabay Video API.

    Returns:
        List[Dict[str, Any]]: List of dicts with 'file_path', 'start_time', 'end_time', 'keyword'.
    """
    if not b_roll_cues:
        return []

    clip_duration = max(1.0, clip_end - clip_start)
    downloaded_brolls: List[Dict[str, Any]] = []
    temp_path = Path(temp_dir).resolve()
    temp_path.mkdir(parents=True, exist_ok=True)

    for idx, cue in enumerate(b_roll_cues):
        keyword = str(cue.get("keyword", "")).strip()
        if not keyword:
            continue

        raw_ts = float(cue.get("timestamp", clip_start))
        # Determine relative timestamp inside the sliced clip
        if raw_ts >= clip_start:
            rel_start = raw_ts - clip_start
        else:
            rel_start = max(0.0, raw_ts)

        # Clamp relative start and end
        rel_start = max(0.0, min(rel_start, clip_duration - 1.0))
        rel_end = min(rel_start + 3.0, clip_duration)

        if rel_end <= rel_start + 0.5:
            continue

        clip_prefix = f"clip_{clip_id}_" if clip_id is not None else ""
        broll_filename = f"broll_{clip_prefix}{idx+1}_{re.sub(r'[^a-zA-Z0-9]', '_', keyword)}.mp4"
        broll_out_path = str(temp_path / broll_filename)

        downloaded_file = fetch_pixabay_broll_clip(
            keyword=keyword,
            output_path=broll_out_path,
            duration_sec=3.0,
        )

        if downloaded_file and os.path.isfile(downloaded_file):
            downloaded_brolls.append({
                "file_path": downloaded_file,
                "start_time": round(rel_start, 2),
                "end_time": round(rel_end, 2),
                "keyword": keyword,
            })

    return downloaded_brolls


# =====================================================================
# Background Music Selector
# =====================================================================

def select_random_background_music(music_dir: str = "music") -> Optional[str]:
    """
    Selects a random MP3 (or audio) file from the local music directory.

    Args:
        music_dir: Directory path containing background music tracks.

    Returns:
        Optional[str]: Absolute path to the selected audio file, or None if none found.
    """
    music_path = Path(music_dir).resolve()
    if not music_path.exists():
        return None

    music_files = (
        list(music_path.glob("*.mp3"))
        + list(music_path.glob("*.MP3"))
        + list(music_path.glob("*.wav"))
        + list(music_path.glob("*.m4a"))
    )

    if not music_files:
        print(f"[!] Info: No audio files found in '{music_dir}/'. Proceeding without background music.")
        return None

    selected = random.choice(music_files)
    print(f"[♫] Selected background music track: '{selected.name}'")
    return str(selected)


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


def generate_dynamic_ass_subtitles(
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
    Generates an Advanced SubStation Alpha (.ass) subtitle file formatted
    with a dynamic karaoke effect:
    - Active word highlighted in yellow (#00FFFF in ASS format, &H0000FFFF) and enlarged (115%).
    - Inactive words remain bold white (&H00FFFFFF).
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

            # Active word: Yellow (#00FFFF in ASS), enlarged 115%, bold
            # Inactive word: White (&H00FFFFFF), 100%, bold
            formatted_words = []
            for j, w in enumerate(chunk):
                if j == i:
                    formatted_words.append(r"{\c&H0000FFFF&\b1\fscx115\fscy115}" + w["word"] + r"{\r}")
                else:
                    formatted_words.append(r"{\c&H00FFFFFF&\b1\fscx100\fscy100}" + w["word"] + r"{\r}")

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


# Alias for compatibility
generate_karaoke_ass_subtitles = generate_dynamic_ass_subtitles


# =====================================================================
# Video Final Compositor (B-Roll Overlays + Subtitles + Background Music)
# =====================================================================

def composite_broll_subtitles_and_music(
    base_video_path: str,
    broll_clips: List[Dict[str, Any]],
    subtitles_ass_path: str,
    output_video_path: str,
    music_file: Optional[str] = None,
    music_volume: float = 0.10,
) -> str:
    """
    Combines:
    1. Dynamic face-tracked base video (1080x1920).
    2. Pixabay B-roll clip overlays at exact timestamps (setpts + overlay).
    3. Karaoke ASS subtitle burn.
    4. Looped, ducked background music (volume=0.10) mixed underneath spoken audio.
    """
    inputs = ["-i", base_video_path]
    video_filters = []
    audio_filters = []
    current_video_stream = "0:v"

    # 1. Overlay each B-roll clip with setpts timing shift and enable expression
    for idx, broll in enumerate(broll_clips, start=1):
        inputs.extend(["-i", broll["file_path"]])
        t_start = broll["start_time"]
        t_end = broll["end_time"]

        # Shift PTS of B-roll input so it starts playing at t_start
        delayed_broll = f"b{idx}_delayed"
        video_filters.append(f"[{idx}:v]setpts=PTS-STARTPTS+{t_start}/TB[{delayed_broll}]")

        # Overlay onto current stream
        next_stream = f"v_overlay_{idx}"
        video_filters.append(
            f"[{current_video_stream}][{delayed_broll}]overlay=x=0:y=0:enable='between(t,{t_start},{t_end})'[{next_stream}]"
        )
        current_video_stream = next_stream

    # 2. Burn ASS Subtitles on top of the overlaid video stream
    escaped_ass_path = subtitles_ass_path.replace("\\", "/").replace(":", "\\:")
    video_filters.append(f"[{current_video_stream}]ass='{escaped_ass_path}'[v_final]")

    # 3. Process Audio (Spoken voice + Ducked looped background music)
    final_audio_map = "0:a"
    if music_file and os.path.isfile(music_file):
        music_input_idx = len(inputs) // 2  # Each input adds two tokens ('-i', 'file')
        inputs.extend(["-i", music_file])

        # Loop background music infinitely, lower volume to 0.10, and mix with main audio
        audio_filters.append(
            f"[{music_input_idx}:a]aloop=loop=-1:size=2e+09,volume={music_volume:.2f}[bg_music]"
        )
        audio_filters.append(
            f"[0:a][bg_music]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[a_final]"
        )
        final_audio_map = "[a_final]"

    all_filters = video_filters + audio_filters
    full_filter_complex = ";".join(all_filters)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", full_filter_complex,
        "-map", "[v_final]",
        "-map", final_audio_map,
        "-c:v", "mpeg4",
        "-q:v", "2",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_video_path,
    ]

    print(f"[*] Rendering final Short (Dynamic Crop + Pixabay B-Roll Overlays + Karaoke Subtitles + Background Music)...")
    subprocess.run(cmd, check=True)

    return output_video_path


# Alias for backward compatibility
composite_broll_and_subtitles = composite_broll_subtitles_and_music


# =====================================================================
# Master Processing Pipeline
# =====================================================================

def process_shorts_video(
    video_path: str,
    start_time: float,
    end_time: float,
    words: List[Dict[str, Any]],
    b_roll_cues: Optional[List[Dict[str, Any]]] = None,
    output_dir: str = "ready",
    output_filename: Optional[str] = None,
    music_dir: str = "music",
    music_file: Optional[str] = None,
    temp_dir: str = "temp",
    clip_id: Optional[int] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    End-to-End Shorts Processor for a single clip:
    1. Dynamic 9:16 Face-Tracking Crop.
    2. Pixabay B-roll Fetching & Standardizing.
    3. Karaoke ASS Subtitle Generation.
    4. Random Background Music Selection & Looping / Ducking.
    5. Final FFmpeg Composite Render.
    6. Saves corresponding metadata JSON file (e.g. ready/clip_1_meta.json).
    """
    video_path = os.path.abspath(video_path)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")

    duration = max(1.0, end_time - start_time)
    base_name = Path(video_path).stem

    temp_path = Path(temp_dir).resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    ready_path = Path(output_dir).resolve()
    ready_path.mkdir(parents=True, exist_ok=True)

    if output_filename:
        final_name = output_filename
    elif clip_id is not None:
        final_name = f"clip_{clip_id}.mp4"
    else:
        final_name = f"{base_name}_short.mp4"

    output_video_path = str(ready_path / final_name)
    clip_tag = f"clip_{clip_id}" if clip_id is not None else Path(final_name).stem
    cropped_base_path = str(temp_path / f"{base_name}_{clip_tag}_cropped_facetrack.mp4")
    subtitles_ass_path = str(temp_path / f"{base_name}_{clip_tag}_subtitles.ass")

    print(f"\n" + "=" * 60)
    print(f"🎬 PROCESSING YOUTUBE SHORT" + (f" (Clip #{clip_id})" if clip_id is not None else ""))
    print(f"=" * 60)
    print(f"📹 Source Video : {video_path}")
    print(f"⏱  Clip Range   : {start_time:.2f}s -> {end_time:.2f}s (Duration: {duration:.2f}s)")
    print(f"📐 Target Format: 9:16 Vertical (1080x1920) with Face Tracking")
    if title:
        print(f"🔥 Title        : {title}")
    if description:
        print(f"📝 Description  : {description}")

    # 1. Dynamic Face-Tracking 9:16 Crop
    crop_video_with_face_tracking(
        video_path=video_path,
        start_time=start_time,
        end_time=end_time,
        output_cropped_path=cropped_base_path,
        target_width=1080,
        target_height=1920,
    )

    # 2. Fetch Pixabay B-Roll Clips
    broll_clips = []
    if b_roll_cues:
        print(f"\n[*] Fetching Pixabay B-roll for {len(b_roll_cues)} visual cues...")
        broll_clips = fetch_all_broll_cues(
            b_roll_cues=b_roll_cues,
            clip_start=start_time,
            clip_end=end_time,
            temp_dir=temp_dir,
            clip_id=clip_id,
        )
        print(f"[✓] Ready B-roll overlays: {len(broll_clips)}")

    # 3. Generate Karaoke ASS Subtitles
    print(f"\n[*] Generating karaoke ASS subtitles from {len(words)} word timestamps...")
    generate_dynamic_ass_subtitles(
        words=words,
        clip_start=start_time,
        clip_end=end_time,
        output_ass_path=subtitles_ass_path,
    )
    print(f"[✓] Subtitles generated at: {subtitles_ass_path}")

    # 4. Select Random Background Music from music/
    bg_music = music_file or select_random_background_music(music_dir=music_dir)

    # 5. Composite B-Roll Overlays + Subtitles + Ducked Looped Music
    print(f"\n[*] Slicing, overlaying Pixabay B-rolls, mixing ducked music, and burning subtitles via FFmpeg...")
    composite_broll_subtitles_and_music(
        base_video_path=cropped_base_path,
        broll_clips=broll_clips,
        subtitles_ass_path=subtitles_ass_path,
        output_video_path=output_video_path,
        music_file=bg_music,
        music_volume=0.10,
    )

    # Clean up intermediate cropped base file and subtitles file
    if os.path.exists(cropped_base_path):
        try:
            os.remove(cropped_base_path)
        except Exception:
            pass

    if os.path.exists(subtitles_ass_path):
        try:
            os.remove(subtitles_ass_path)
        except Exception:
            pass

    # 6. Save Metadata File (e.g., ready/clip_1_meta.json)
    meta_filename = f"{Path(output_video_path).stem}_meta.json"
    meta_path = str(ready_path / meta_filename)
    meta_data = {
        "title": title or f"Viral Short #{clip_id or 1} 🔥 #Shorts",
        "description": description or "Check out this clip! #Shorts",
    }
    if clip_id is not None:
        meta_data["clip_id"] = clip_id
    meta_data["start"] = start_time
    meta_data["end"] = end_time

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"🎉 YouTube Short Complete!" + (f" (Clip #{clip_id})" if clip_id is not None else ""))
    print(f"📁 Video Saved  : {output_video_path}")
    print(f"📄 Meta Saved   : {meta_path}")
    print(f"📊 File Size    : {os.path.getsize(output_video_path) / (1024 * 1024):.2f} MB")
    print("=" * 60)

    return output_video_path


def process_all_clips(
    video_path: str,
    clips: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    output_dir: str = "ready",
    music_dir: str = "music",
    music_file: Optional[str] = None,
    temp_dir: str = "temp",
    output_filename: Optional[str] = None,
) -> List[str]:
    """
    Processes all clips in the clips list:
    1. Iterates through each clip object.
    2. Executes dynamic face-tracking 9:16 crop, broll overlays, karaoke subtitles, and music.
    3. Saves rendered video to output_dir/clip_{clip_id}.mp4.
    4. Saves metadata JSON to output_dir/clip_{clip_id}_meta.json.

    Returns:
        List[str]: List of paths to rendered MP4 video files.
    """
    rendered_files: List[str] = []
    total_clips = len(clips)

    print("\n" + "=" * 60)
    print(f"🎬 BATCH PROCESSING {total_clips} SHORTS CLIP(S)")
    print("=" * 60)

    for idx, clip in enumerate(clips, start=1):
        clip_id = clip.get("clip_id", idx)
        start_time = float(clip.get("start", clip.get("start_time", 0.0)))
        end_time = float(clip.get("end", clip.get("end_time", start_time + 30.0)))
        title = clip.get("title", f"Viral Short #{clip_id} 🔥 #Shorts")
        description = clip.get("description", "Check out this clip! #Shorts")
        b_roll_cues = clip.get("b_roll_cues", [])

        # Determine output filename
        if total_clips == 1 and output_filename:
            target_out_name = output_filename
        else:
            target_out_name = f"clip_{clip_id}.mp4"

        print(f"\n>>> Processing Clip {idx}/{total_clips} (ID: {clip_id}, Range: {start_time:.2f}s - {end_time:.2f}s)...")

        out_path = process_shorts_video(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            words=words,
            b_roll_cues=b_roll_cues,
            output_dir=output_dir,
            output_filename=target_out_name,
            music_dir=music_dir,
            music_file=music_file,
            temp_dir=temp_dir,
            clip_id=clip_id,
            title=title,
            description=description,
        )
        rendered_files.append(out_path)

    print("\n" + "=" * 60)
    print(f"✨ ALL {len(rendered_files)} CLIPS PROCESSED SUCCESSFULLY!")
    for f in rendered_files:
        print(f"  • {f}")
    print("=" * 60)

    return rendered_files


# =====================================================================
# Checkpoint & Metadata Resolver
# =====================================================================

def load_clips_and_words(
    video_path: str,
    db_path: str = "checkpoints.db",
    analysis_json: Optional[str] = None,
    words_json: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Resolves clips list (from JSON array or SQLite) and word-level timestamps.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            (clips_list, words_list)
    """
    clips: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []

    # 1. Load clips from explicit analysis_json if provided or exists
    if analysis_json and os.path.isfile(analysis_json):
        try:
            with open(analysis_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    clips = data
                elif isinstance(data, dict):
                    if "clips" in data and isinstance(data["clips"], list):
                        clips = data["clips"]
                    else:
                        clips = [data]
                print(f"[✓] Loaded {len(clips)} clip(s) from '{analysis_json}'")
        except Exception as e:
            print(f"[!] Warning: Failed to load '{analysis_json}': {e}", file=sys.stderr)

    # 2. Load words from words_json if provided
    if words_json and os.path.isfile(words_json):
        try:
            with open(words_json, "r", encoding="utf-8") as f:
                words = json.load(f)
                print(f"[✓] Loaded {len(words)} words from '{words_json}'")
        except Exception as e:
            print(f"[!] Warning: Failed to load '{words_json}': {e}", file=sys.stderr)

    # 3. Check SQLite Checkpoints for transcription words or analysis
    if (not words or not clips) and os.path.isfile(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Load transcription words
            if not words:
                base_name = Path(video_path).stem
                cursor.execute(
                    "SELECT full_text, segments_json, words_json FROM transcription_checkpoints WHERE file_path LIKE ? ORDER BY created_at DESC LIMIT 1",
                    (f"%{base_name}%",),
                )
                row = cursor.fetchone()
                if not row:
                    cursor.execute(
                        "SELECT full_text, segments_json, words_json FROM transcription_checkpoints ORDER BY created_at DESC LIMIT 1"
                    )
                    row = cursor.fetchone()
                if row and row[2]:
                    words = json.loads(row[2])
                    print(f"[✓] Loaded {len(words)} words from SQLite transcription checkpoint.")

            # Load analysis clips if not already loaded from JSON
            if not clips:
                cursor.execute(
                    "SELECT result_json FROM analysis_checkpoints WHERE status = 'success' ORDER BY created_at DESC LIMIT 1"
                )
                analysis_row = cursor.fetchone()
                if not analysis_row:
                    cursor.execute(
                        "SELECT result_json FROM analysis_checkpoints ORDER BY created_at DESC LIMIT 1"
                    )
                    analysis_row = cursor.fetchone()

                if analysis_row and analysis_row[0]:
                    analysis_data = json.loads(analysis_row[0])
                    if isinstance(analysis_data, list):
                        clips = analysis_data
                    elif isinstance(analysis_data, dict):
                        if "clips" in analysis_data and isinstance(analysis_data["clips"], list):
                            clips = analysis_data["clips"]
                        else:
                            clips = [analysis_data]
                    print(f"[✓] Loaded {len(clips)} clip(s) from SQLite analysis checkpoint.")

            conn.close()
        except Exception as e:
            print(f"[!] Warning: Could not read from SQLite database '{db_path}': {e}", file=sys.stderr)

    # Also check if temp/words.json exists if words is still empty
    if not words and os.path.isfile("temp/words.json"):
        try:
            with open("temp/words.json", "r", encoding="utf-8") as f:
                words = json.load(f)
        except Exception:
            pass

    # Normalize clips list format
    normalized_clips: List[Dict[str, Any]] = []
    for idx, c in enumerate(clips, start=1):
        if not isinstance(c, dict):
            continue
        clip_id = c.get("clip_id", idx)
        start_t = float(c.get("start", c.get("start_time", 0.0)))
        end_t = float(c.get("end", c.get("end_time", start_t + 30.0)))
        title = c.get("title", f"Viral Short #{clip_id} 🔥 #Shorts")
        description = c.get("description", "Check out this clip! #Shorts")
        b_roll_cues = c.get("b_roll_cues", [])

        normalized_clips.append({
            "clip_id": clip_id,
            "start": start_t,
            "end": end_t,
            "title": title,
            "description": description,
            "b_roll_cues": b_roll_cues,
        })

    return normalized_clips, words


def load_from_checkpoints_or_json(
    video_path: str,
    db_path: str = "checkpoints.db",
    analysis_json: Optional[str] = None,
    words_json: Optional[str] = None,
) -> Tuple[float, float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Backward-compatible single-clip resolver."""
    clips, words = load_clips_and_words(
        video_path=video_path,
        db_path=db_path,
        analysis_json=analysis_json,
        words_json=words_json,
    )
    if clips:
        first = clips[0]
        return first["start"], first["end"], words, first.get("b_roll_cues", [])
    return 0.0, 30.0, words, []


# =====================================================================
# CLI Entrypoint
# =====================================================================

def find_latest_video(temp_dir: str = "temp") -> str:
    """Finds the most recently modified input MP4 in the temp folder."""
    temp_path = Path(temp_dir).resolve()
    if not temp_path.exists():
        raise FileNotFoundError(f"Directory '{temp_dir}' does not exist.")
    videos = [p for p in temp_path.glob("*.mp4") if not p.stem.endswith("_short") and not p.stem.startswith("broll_") and not p.stem.startswith("clip_")]
    if not videos:
        videos = list(temp_path.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 video files found in '{temp_dir}'. Run download.py first!")
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(videos[0])


def main():
    parser = argparse.ArgumentParser(
        description="Process YouTube Shorts from analysis.json clips array with dynamic face-tracking 9:16 crop, Pixabay B-roll overlays, karaoke subtitles, and background music."
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=None,
        help="Path to input MP4 video (defaults to latest video in temp/).",
    )
    parser.add_argument("--start", type=float, default=None, help="Start timestamp in seconds (overrides analysis clips).")
    parser.add_argument("--end", type=float, default=None, help="End timestamp in seconds (overrides analysis clips).")
    parser.add_argument("--analysis-json", type=str, default=None, help="Path to analysis JSON from analyze.py.")
    parser.add_argument("--words-json", type=str, default=None, help="Path to words JSON.")
    parser.add_argument("--db-path", type=str, default="checkpoints.db", help="SQLite checkpoints db path.")
    parser.add_argument("-o", "--ready-dir", type=str, default="ready", help="Output directory (default: ready).")
    parser.add_argument("--output-name", type=str, default=None, help="Custom output MP4 filename (for single clip).")
    parser.add_argument("--music-dir", type=str, default="music", help="Directory containing background MP3 tracks (default: music).")
    parser.add_argument("--music-file", type=str, default=None, help="Specific background music file to use.")

    args = parser.parse_args()

    try:
        video_path = args.video_path or find_latest_video()
        analysis_json = args.analysis_json
        if analysis_json is None and os.path.isfile("temp/analysis.json"):
            analysis_json = "temp/analysis.json"

        # Resolve clips list and words
        clips, words = load_clips_and_words(
            video_path=video_path,
            db_path=args.db_path,
            analysis_json=analysis_json,
            words_json=args.words_json,
        )

        # If --start or --end passed explicitly via CLI, override clips list with single clip
        if args.start is not None or args.end is not None:
            s = args.start if args.start is not None else (clips[0]["start"] if clips else 0.0)
            e = args.end if args.end is not None else (clips[0]["end"] if clips else s + 30.0)
            t = clips[0]["title"] if clips else "Viral Short 🔥 #Shorts"
            d = clips[0]["description"] if clips else "Check out this clip! #Shorts"
            b = clips[0].get("b_roll_cues", []) if clips else []
            clips = [{
                "clip_id": 1,
                "start": s,
                "end": e,
                "title": t,
                "description": d,
                "b_roll_cues": b,
            }]
        elif not clips:
            # Fallback if no clips found in json or db
            clips = [{
                "clip_id": 1,
                "start": 0.0,
                "end": 30.0,
                "title": "Viral Short 🔥 #Shorts",
                "description": "Check out this clip! #Shorts",
                "b_roll_cues": [],
            }]

        process_all_clips(
            video_path=video_path,
            clips=clips,
            words=words,
            output_dir=args.ready_dir,
            music_dir=args.music_dir,
            music_file=args.music_file,
            output_filename=args.output_name,
        )
    except Exception as e:
        print(f"\n[!] Error processing video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
