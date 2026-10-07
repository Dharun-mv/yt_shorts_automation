#!/usr/bin/env python3
"""
YouTube Video Downloader & Audio Extractor for YouTube Shorts Automation.
Downloads the highest quality MP4 video and extracts audio as a 16kHz mono WAV file.
"""

import argparse
import os
import sys
from pathlib import Path
import ffmpeg
import yt_dlp


def download_video_and_extract_audio(url: str, output_dir: str = "temp") -> tuple[str, str]:
    """
    Downloads the highest quality MP4 video from a YouTube URL and extracts audio as WAV.

    Args:
        url: The YouTube video or shorts URL.
        output_dir: Directory where files should be stored. Defaults to 'temp'.

    Returns:
        tuple[str, str]: Tuple containing (video_path, audio_path).
    """
    out_dir_path = Path(output_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Template for downloaded video
    output_template = str(out_dir_path / "%(id)s.%(ext)s")

    ydl_opts = {
        # Select best video + best audio and remux/merge to mp4
        "format": "bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "postprocessor_args": ["-ar", "16000", "-ac", "1"],
    }

    print(f"[*] Downloading highest quality MP4 from: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Determine the resulting MP4 path
        video_id = info.get("id")
        video_title = info.get("title", video_id)
        
        # When merged into mp4, the final filename will use .mp4 extension
        video_path = str(out_dir_path / f"{video_id}.mp4")
        if not os.path.exists(video_path):
            # Fallback to checking prepared filename if extension differs
            prepared = ydl.prepare_filename(info)
            base, _ = os.path.splitext(prepared)
            if os.path.exists(f"{base}.mp4"):
                video_path = f"{base}.mp4"
            elif os.path.exists(prepared):
                video_path = prepared

    print(f"[+] Downloaded video saved to: {video_path}")

    # Extract audio to WAV (16kHz mono PCM is optimal for Whisper transcription)
    wav_path = str(Path(video_path).with_suffix(".wav"))
    print(f"[*] Extracting audio to: {wav_path}")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(wav_path, acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=False)
        )
        print(f"[+] Extracted audio saved to: {wav_path}")
    except ffmpeg.Error as e:
        print(f"[-] FFmpeg error while extracting audio: {e}", file=sys.stderr)
        raise

    print("\n[✓] Processing Complete:")
    print(f"    - Title: {video_title}")
    print(f"    - Video (MP4): {video_path}")
    print(f"    - Audio (WAV): {wav_path}")

    return video_path, wav_path


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video/shorts as MP4 and extract audio as WAV."
    )
    parser.add_argument(
        "url",
        type=str,
        help="YouTube video or Shorts URL to process."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="temp",
        help="Directory to save downloaded video and extracted audio (default: temp)."
    )

    args = parser.parse_args()
    try:
        download_video_and_extract_audio(args.url, args.output_dir)
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
