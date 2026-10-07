import sys
"""
Master orchestration pipeline for YouTube Shorts Automation with interactive terminal menu.
Coordinates:
0. Interactive CLI prompt or RSS feed ingestion via fetch_sources.py.
1. Video download and audio extraction via download.py.
2. Word-level transcription, viral hook extraction, and B-roll cue generation via analyze.py.
3. Dynamic face tracking, Pixabay B-roll overlays, karaoke subtitles, and background music via process.py.
4. Scheduled YouTube OAuth2 upload via upload.py (or local save in /ready).
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


def cleanup_temp_dir(temp_dir: str = "temp") -> None:
    """
    Cleans up intermediate temporary working files from temp_dir upon error or completion.
    Preserves downloaded model assets (e.g. .tflite files) to avoid redundant downloads.
    """
    temp_path = Path(temp_dir).resolve()
    if not temp_path.exists():
        return

    print(f"[*] Cleaning up temporary files in '{temp_dir}/'...")
    for item in temp_path.iterdir():
        # Preserve model weight files
        if item.suffix in [".tflite", ".bin"]:
            continue
        try:
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception as e:
            print(f"[!] Warning: Could not remove '{item.name}': {e}", file=sys.stderr)


def get_target_url(
    source: Optional[str] = None,
    processed_file: str = "processed_videos.txt",
) -> Optional[str]:
    """
    Resolves the target YouTube video URL:
    - If source is provided and not 'AUTO', uses it.
    - If source is 'AUTO' or omitted, queries fetch_sources.py to find the latest unprocessed video.
    """
    if source and source.strip().upper() != "AUTO":
        return source.strip()

    print("=" * 60)
    print("📡 CHECKING PREDEFINED YOUTUBE RSS FEEDS")
    print("=" * 60)

    try:
        from fetch_sources import fetch_latest_unprocessed_video
        video_url = fetch_latest_unprocessed_video(
            processed_file=processed_file,
            mark_as_processed=True,
        )
        return video_url
    except Exception as e:
        print(f"[!] Error querying RSS feeds via fetch_sources: {e}", file=sys.stderr)
        return None


def find_completed_video(ready_dir: str = "ready", expected_stem: Optional[str] = None) -> Optional[str]:
    """Finds the completed short video in the ready directory."""
    ready_path = Path(ready_dir).resolve()
    if not ready_path.exists():
        return None

    if expected_stem:
        candidate = ready_path / f"{expected_stem}_short.mp4"
        if candidate.exists():
            return str(candidate)

    videos = list(ready_path.glob("*.mp4"))
    if not videos:
        return None

    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(videos[0])


def run_pipeline(
    source: Optional[str] = None,
    action: str = "UPLOAD",
    youtube_url: Optional[str] = None,
    skip_upload: Optional[bool] = None,
    model_size: str = "base",
    ready_dir: str = "ready",
    temp_dir: str = "temp",
    processed_file: str = "processed_videos.txt",
    keep_temp: bool = False,
) -> bool:
    """
    Executes the Shorts automation pipeline based on chosen source and action.

    Args:
        source: YouTube URL or 'AUTO' to fetch from RSS ingestor.
        action: 'UPLOAD' to publish to YouTube or 'SAVE' to keep locally.
        youtube_url: Backward compatibility alias for source.
        skip_upload: Optional boolean override for upload skipping (True = SAVE).
        model_size: Whisper model size for transcription.
        ready_dir: Target folder for the finished Short.
        temp_dir: Directory for temporary assets.
        processed_file: Path to processed videos tracking file.
        keep_temp: If True, skips temporary directory cleanup.

    Returns:
        bool: True if pipeline succeeded, False otherwise.
    """
    effective_source = source or youtube_url
    if not effective_source:
        print("[!] Error: No source URL or 'AUTO' mode specified.", file=sys.stderr)
        return False

    # Resolve target URL (fetch from RSS if AUTO)
    if effective_source.strip().upper() == "AUTO":
        target_url = get_target_url("AUTO", processed_file=processed_file)
        if not target_url:
            print("\n[-] No new unprocessed videos found across channels. Exiting cleanly.")
            return False
    else:
        target_url = effective_source.strip()

    # Determine action mode
    if skip_upload is not None:
        is_save = skip_upload
    else:
        is_save = action.strip().upper() == "SAVE"

    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    Path(ready_dir).mkdir(parents=True, exist_ok=True)

    try:
        print("\n" + "=" * 60)
        print("🚀 STARTING YOUTUBE SHORTS AUTOMATION PIPELINE")
        print("=" * 60)
        print(f"🔗 Target URL   : {target_url}")
        print(f"🧠 Whisper Model: {model_size}")
        print(f"📤 Action Mode  : {'SAVE (Local Only)' if is_save else 'UPLOAD (YouTube Scheduled)'}")
        print("=" * 60)

        # -------------------------------------------------------------
        # Phase 1: Download & Extract Audio
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("📥 Phase 1: Download Video & Extract Audio...")
        print("-" * 50)
        cmd_download = [
            sys.executable,
            "download.py",
            target_url,
            "-o",
            temp_dir,
        ]
        subprocess.run(cmd_download, check=True)

        mp4_files = sorted(Path(temp_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if mp4_files:
            video_path = str(mp4_files[0])
            wav_candidate = Path(video_path).with_suffix(".wav")
            audio_path = str(wav_candidate) if wav_candidate.exists() else str(next(Path(temp_dir).glob("*.wav"), ""))
        else:
            try:
                from download import download_video_and_extract_audio
                video_path, audio_path = download_video_and_extract_audio(target_url, output_dir=temp_dir)
            except Exception:
                video_path = str(Path(temp_dir) / "downloaded.mp4")
                audio_path = str(Path(temp_dir) / "downloaded.wav")
        print(f"[✓] Video saved: {video_path}")
        print(f"[✓] Audio saved: {audio_path}")

        # -------------------------------------------------------------
        # Phase 2: Analyze & Extract Hooks + B-Roll Cues
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("🔍 Phase 2: Transcribe & Extract Viral Clip...")
        print("-" * 50)
        analysis_json = str(Path(temp_dir) / "analysis.json")
        cmd_analyze = [
            sys.executable,
            "analyze.py",
            audio_path,
            "--model-size",
            model_size,
            "-o",
            analysis_json,
        ]
        subprocess.run(cmd_analyze, check=True)
        print(f"[✓] Analysis saved: {analysis_json}")

        # -------------------------------------------------------------
        # Phase 3: Dynamic Crop, B-Roll Overlays, Subtitles & Audio
        # -------------------------------------------------------------
        print("\n" + "-" * 50)
        print("✂️  Phase 3: Face Tracking, B-Roll Overlays, Subtitles & Audio...")
        print("-" * 50)
        cmd_process = [
            sys.executable,
            "process.py",
            video_path,
            "--analysis-json",
            analysis_json,
            "-o",
            ready_dir,
        ]
        subprocess.run(cmd_process, check=True)
        print(f"[✓] Processed video ready in '{ready_dir}/'")

        # -------------------------------------------------------------
        # Phase 4: YouTube API Upload or Local Save
        # -------------------------------------------------------------
        completed_video_path = find_completed_video(ready_dir, Path(video_path).stem)

        if is_save:
            print("\n" + "-" * 50)
            print("⏭️  Phase 4: YouTube Upload skipped ('SAVE' mode selected).")
            print("-" * 50)
            if completed_video_path:
                print(f"[✓] Path to completed video in '/{ready_dir}': {completed_video_path}")
            else:
                print(f"[✓] Completed video saved in directory: {Path(ready_dir).resolve()}")
        else:
            print("\n" + "-" * 50)
            print("🚀 Phase 4: YouTube OAuth2 Upload & Scheduling...")
            print("-" * 50)
            cmd_upload = [
                sys.executable,
                "upload.py",
                "--ready-dir",
                ready_dir,
                "--analysis-json",
                analysis_json,
            ]
            subprocess.run(cmd_upload, check=True)
            print("[✓] Video successfully uploaded to YouTube!")

        # Cleanup temp directory on successful run
        if not keep_temp:
            cleanup_temp_dir(temp_dir)

        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        if is_save and completed_video_path:
            print(f"📁 Local Video: {completed_video_path}")
        print("=" * 60)
        return True

    except subprocess.CalledProcessError as e:
        cmd_str = " ".join(e.cmd) if isinstance(e.cmd, list) else str(e.cmd)
        print(f"\n[!] Pipeline step failed during command execution: {cmd_str}", file=sys.stderr)
        cleanup_temp_dir(temp_dir)
        return False
    except KeyboardInterrupt:
        print("\n[!] Pipeline cancelled by user interrupt.", file=sys.stderr)
        cleanup_temp_dir(temp_dir)
        return False
    except Exception as e:
        print(f"\n[!] Pipeline encountered an unexpected error: {e}", file=sys.stderr)
        cleanup_temp_dir(temp_dir)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Master orchestration pipeline for YouTube Shorts Automation."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=None,
        help="Optional YouTube long-form URL or 'AUTO' (if omitted, prompts interactively).",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["UPLOAD", "SAVE", "upload", "save"],
        default=None,
        help="Action: 'UPLOAD' to publish to YouTube or 'SAVE' to keep locally.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip the final YouTube OAuth2 upload phase (equivalent to Action='SAVE').",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=os.getenv("WHISPER_MODEL_SIZE", "base"),
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--ready-dir",
        type=str,
        default="ready",
        help="Directory to save final processed video (default: ready).",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="temp",
        help="Temporary working directory (default: temp).",
    )
    parser.add_argument(
        "--processed-file",
        type=str,
        default="processed_videos.txt",
        help="Path to processed_videos.txt tracking file (default: processed_videos.txt).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not clean up temporary files after completion.",
    )

    args = parser.parse_args()

    source_choice = args.url
    action_choice = args.action
    if args.skip_upload:
        action_choice = "SAVE"

    # Interactive prompt for Question 1: Source
    if not source_choice:
        while True:
            try:
                user_input = input(
                    "Source: Enter a YouTube URL manually, or type 'AUTO' to fetch the latest podcast from the RSS ingestor: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[!] Operation cancelled.")
                sys.exit(0)
            if user_input:
                source_choice = user_input
                break
            print("[!] Please enter a valid YouTube URL or 'AUTO'.\n")

    # Interactive prompt for Question 2: Action
    if not action_choice:
        while True:
            try:
                user_input = input(
                    "Action: Type 'UPLOAD' to schedule the final video to YouTube, or 'SAVE' to just keep it locally: "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[!] Operation cancelled.")
                sys.exit(0)
            if user_input.upper() in ["UPLOAD", "SAVE"]:
                action_choice = user_input.upper()
                break
            print("[!] Invalid action. Please enter 'UPLOAD' or 'SAVE'.\n")

    success = run_pipeline(
        source=source_choice,
        action=action_choice,
        model_size=args.model_size,
        ready_dir=args.ready_dir,
        temp_dir=args.temp_dir,
        processed_file=args.processed_file,
        keep_temp=args.keep_temp,
    )

    if not success:
        print("\n[!] Execution halted gracefully.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
