#!/usr/bin/env python3
"""
YouTube RSS Feed Fetcher & Unprocessed Video Identifier.
Monitors predefined YouTube channel RSS feeds (with yt-dlp scraping fallback),
identifies the newest video that has not been processed yet, tracks it in a
local processed_videos.txt file, and returns the YouTube URL for processing.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
import requests


# Predefined YouTube Channel Targets
DEFAULT_CHANNELS = [
    {
        "name": "Lex Fridman",
        "channel_id": "UCSHZKyawb77ixDdsGog4iWA",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCSHZKyawb77ixDdsGog4iWA",
        "channel_url": "https://www.youtube.com/@lexfridman/videos",
        "user": "LexFridman",
    },
    {
        "name": "Huberman Lab",
        "channel_id": "UC2D2CMWXMOVWx7giW1n3LIg",
        "rss_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UC2D2CMWXMOVWx7giW1n3LIg",
        "channel_url": "https://www.youtube.com/@hubermanlab/videos",
        "user": "hubermanlab",
    },
]

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


def normalize_datetime(dt: Optional[datetime]) -> datetime:
    """Normalizes any datetime to a timezone-aware UTC datetime."""
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Extracts the 11-character YouTube video ID from a URL or raw ID string.

    Args:
        url_or_id: YouTube URL (watch, shorts, youtu.be) or raw ID.

    Returns:
        Optional[str]: Extracted 11-character video ID, or None if invalid.
    """
    if not url_or_id:
        return None

    cleaned = url_or_id.strip()

    # Match raw 11-character video ID
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", cleaned):
        return cleaned

    # Match standard watch URL: ?v=VIDEO_ID
    match_watch = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})", cleaned)
    if match_watch:
        return match_watch.group(1)

    # Match youtu.be, /shorts/, /embed/, /v/ URLs
    match_path = re.search(
        r"(?:youtu\.be/|/shorts/|/embed/|/v/)([a-zA-Z0-9_-]{11})", cleaned
    )
    if match_path:
        return match_path.group(1)

    return None


def load_processed_video_ids(filepath: str | Path = "processed_videos.txt") -> Set[str]:
    """
    Reads the local processed_videos.txt file and returns a set of processed video IDs.

    Args:
        filepath: Path to the processed videos file.

    Returns:
        Set[str]: Set of processed YouTube video IDs.
    """
    path = Path(filepath)
    if not path.exists():
        return set()

    processed_ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith("#"):
                continue

            # Strip inline comments
            line_content = line.split("#", 1)[0].strip()
            vid = extract_video_id(line_content)
            if vid:
                processed_ids.add(vid)
            else:
                # If extraction fails, store the whole trimmed token as a fallback
                processed_ids.add(line_content)

    return processed_ids


def append_processed_video(
    filepath: str | Path,
    video_id: str,
    video_url: str,
    title: str = "",
    channel_name: str = "",
) -> None:
    """
    Appends a processed video entry to the local processed_videos.txt file.

    Args:
        filepath: Path to the processed videos file.
        video_id: The YouTube video ID.
        video_url: The canonical YouTube watch URL.
        title: Optional title of the video.
        channel_name: Optional channel name.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{video_url}\n")


def fetch_videos_via_ytdlp(
    channel_url: str,
    channel_name: str = "",
    channel_id: str = "",
    max_items: int = 5,
) -> List[Dict]:
    """
    Fetches the latest videos directly from a YouTube channel URL using yt-dlp.
    Uses the Python yt_dlp module if available, and falls back to running
    the yt-dlp CLI via subprocess.

    Args:
        channel_url: YouTube channel videos URL (e.g., 'https://www.youtube.com/@lexfridman/videos').
        channel_name: Optional display name for the channel.
        channel_id: Optional channel ID.
        max_items: Number of recent video items to fetch (default: 5).

    Returns:
        List[Dict]: List of video dictionaries containing video_id, title, published, url, channel_name, channel_id.
    """
    entries: List[Dict] = []
    ch_title = channel_name
    ch_id = channel_id

    # 1. Try python yt_dlp library first
    try:
        import yt_dlp

        ydl_opts = {
            "extract_flat": True,
            "playlist_items": f"1:{max_items}",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            data = ydl.extract_info(channel_url, download=False)
            if data:
                ch_title = channel_name or data.get("channel") or data.get("uploader") or ""
                ch_id = channel_id or data.get("channel_id") or data.get("id") or ""
                raw_entries = data.get("entries")
                if raw_entries is not None:
                    entries = [e for e in raw_entries if e]
                elif "id" in data:
                    entries = [data]
    except Exception:
        entries = []

    # 2. Fallback to CLI subprocess
    if not entries:
        try:
            cmd = [
                "yt-dlp",
                "--flat-playlist",
                "--playlist-items",
                f"1:{max_items}",
                "-J",
                channel_url,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                ch_title = channel_name or data.get("channel") or data.get("uploader") or ""
                ch_id = channel_id or data.get("channel_id") or data.get("id") or ""
                raw_entries = data.get("entries")
                if raw_entries is not None:
                    entries = [e for e in raw_entries if e]
                elif "id" in data:
                    entries = [data]
        except Exception as e:
            print(f"[!] yt-dlp fallback error for {channel_name or channel_url}: {e}", file=sys.stderr)

    videos: List[Dict] = []
    for entry in entries:
        if not entry:
            continue
        v_id = entry.get("id")
        if not v_id:
            continue
        title = entry.get("title") or ""
        watch_url = f"https://www.youtube.com/watch?v={v_id}"

        pub_dt = datetime.min.replace(tzinfo=timezone.utc)
        pub_str = str(entry.get("upload_date") or entry.get("timestamp") or "")
        if entry.get("timestamp"):
            try:
                pub_dt = datetime.fromtimestamp(entry["timestamp"], tz=timezone.utc)
            except Exception:
                pass
        elif pub_str and len(pub_str) == 8 and pub_str.isdigit():
            try:
                pub_dt = datetime.strptime(pub_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        videos.append({
            "video_id": v_id,
            "title": title,
            "published": normalize_datetime(pub_dt),
            "published_str": pub_str,
            "url": watch_url,
            "channel_name": ch_title or channel_name,
            "channel_id": ch_id or channel_id,
        })

    return videos


def fetch_channel_videos(
    channel_id: Optional[str] = None,
    channel_name: str = "",
    timeout: int = 15,
    rss_url: Optional[str] = None,
    channel_url: Optional[str] = None,
    user: Optional[str] = None,
    max_items: int = 5,
) -> List[Dict]:
    """
    Fetches and parses the latest videos from a YouTube channel.
    Attempts RSS feeds first using channel_id and/or user parameters.
    If requests.get returns 404, 0 videos, or fails, falls back to yt-dlp scraping.

    Args:
        channel_id: YouTube Channel ID (e.g., 'UCSHZKyawb77ixDdsGog4iWA').
        channel_name: Optional display name for the channel.
        timeout: HTTP request timeout in seconds.
        rss_url: Optional explicit RSS feed URL.
        channel_url: Optional channel videos URL for yt-dlp fallback.
        user: Optional YouTube username / handle for RSS user param.
        max_items: Maximum items to retrieve via fallback.

    Returns:
        List[Dict]: List of video dictionaries containing id, title, published, url, channel_name, channel_id.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    # Build list of RSS feed candidate URLs
    rss_candidates: List[str] = []
    if rss_url:
        rss_candidates.append(rss_url)
    if channel_id:
        c_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        if c_url not in rss_candidates:
            rss_candidates.append(c_url)
    if user:
        u_url = f"https://www.youtube.com/feeds/videos.xml?user={user}"
        if u_url not in rss_candidates:
            rss_candidates.append(u_url)

    videos: List[Dict] = []

    for feed_url in rss_candidates:
        try:
            response = requests.get(feed_url, headers=headers, timeout=timeout)
            if response.status_code == 404:
                print(f"[!] RSS feed returned 404 for {channel_name or feed_url} ({feed_url})", file=sys.stderr)
                continue
            response.raise_for_status()

            root = ET.fromstring(response.content)
            current_ch_name = channel_name
            if not current_ch_name:
                title_el = root.find("atom:title", ATOM_NS)
                if title_el is not None and title_el.text:
                    current_ch_name = title_el.text.strip()

            parsed_videos: List[Dict] = []
            for entry in root.findall("atom:entry", ATOM_NS):
                v_id_el = entry.find("yt:videoId", ATOM_NS)
                title_el = entry.find("atom:title", ATOM_NS)
                pub_el = entry.find("atom:published", ATOM_NS)
                link_el = entry.find("atom:link", ATOM_NS)

                if v_id_el is None or v_id_el.text is None:
                    continue

                video_id = v_id_el.text.strip()
                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                pub_str = pub_el.text.strip() if pub_el is not None and pub_el.text else ""

                pub_dt = datetime.min
                if pub_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_str)
                    except ValueError:
                        pass

                watch_url = f"https://www.youtube.com/watch?v={video_id}"
                if link_el is not None and link_el.attrib.get("href"):
                    href = link_el.attrib.get("href")
                    if "watch?v=" in href:
                        watch_url = href

                parsed_videos.append({
                    "video_id": video_id,
                    "title": title,
                    "published": normalize_datetime(pub_dt),
                    "published_str": pub_str,
                    "url": watch_url,
                    "channel_name": current_ch_name,
                    "channel_id": channel_id or "",
                })

            if parsed_videos:
                videos = parsed_videos
                break  # Successfully parsed RSS feed
        except (requests.RequestException, ET.ParseError) as e:
            print(f"[!] Warning: RSS fetch error for {channel_name or feed_url}: {e}", file=sys.stderr)
            continue

    # Fallback mechanism if RSS returned 404, error, or 0 videos
    if not videos:
        # Determine fallback channel URL
        target_channel_url = channel_url
        if not target_channel_url:
            if user:
                target_channel_url = f"https://www.youtube.com/@{user.lstrip('@')}/videos"
            elif channel_id:
                target_channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
            elif channel_name:
                target_channel_url = f"https://www.youtube.com/@{channel_name.lower().replace(' ', '')}/videos"

        if target_channel_url:
            print(f"[*] RSS feed unavailable or empty for '{channel_name}'. Falling back to yt-dlp on '{target_channel_url}'...")
            videos = fetch_videos_via_ytdlp(
                channel_url=target_channel_url,
                channel_name=channel_name,
                channel_id=channel_id or "",
                max_items=max_items,
            )

    return videos


def fetch_latest_unprocessed_video(
    processed_file: str | Path = "processed_videos.txt",
    channels: Optional[List[Dict[str, str]]] = None,
    mark_as_processed: bool = True,
) -> Optional[str]:
    """
    Parses RSS feeds (with yt-dlp scraping fallback) for the predefined channels,
    finds the newest video not yet present in processed_videos.txt, appends it to
    the file, and returns its YouTube URL.

    Args:
        processed_file: Path to the local text file tracking processed video IDs/URLs.
        channels: Optional list of channel dicts.
        mark_as_processed: If True, writes the new video URL to processed_videos.txt.

    Returns:
        Optional[str]: YouTube URL of the latest unprocessed video, or None if no new video found.
    """
    if channels is None:
        channels = DEFAULT_CHANNELS

    processed_ids = load_processed_video_ids(processed_file)
    print(f"[*] Loaded {len(processed_ids)} previously processed video ID(s) from '{processed_file}'")

    all_unprocessed: List[Dict] = []

    for ch in channels:
        ch_name = ch.get("name", ch.get("channel_id", "Unknown"))
        ch_id = ch.get("channel_id", "")
        rss_url = ch.get("rss_url") or ch.get("url")
        ch_url = ch.get("channel_url")
        user = ch.get("user")

        target_display = ch_id or ch_url or rss_url or user or ch_name
        print(f"[*] Checking videos for '{ch_name}' ({target_display})...")
        videos = fetch_channel_videos(
            channel_id=ch_id,
            channel_name=ch_name,
            rss_url=rss_url,
            channel_url=ch_url,
            user=user,
        )
        print(f"    -> Found {len(videos)} video(s).")

        for v in videos:
            if v["video_id"] not in processed_ids:
                all_unprocessed.append(v)

    if not all_unprocessed:
        print("[-] No new unprocessed videos found across all channels.")
        return None

    # Sort by publication date descending (latest first)
    all_unprocessed.sort(key=lambda x: x["published"], reverse=True)

    latest_video = all_unprocessed[0]
    vid_id = latest_video["video_id"]
    vid_url = latest_video["url"]
    vid_title = latest_video["title"]
    vid_channel = latest_video["channel_name"]
    vid_pub = latest_video["published_str"]

    print("\n" + "=" * 60)
    print("✨ LATEST UNPROCESSED VIDEO FOUND")
    print("=" * 60)
    print(f"📺 Channel   : {vid_channel}")
    print(f"🎬 Title     : {vid_title}")
    print(f"📅 Published : {vid_pub}")
    print(f"🔗 URL       : {vid_url}")
    print("=" * 60)

    if mark_as_processed:
        append_processed_video(
            filepath=processed_file,
            video_id=vid_id,
            video_url=vid_url,
            title=vid_title,
            channel_name=vid_channel,
        )
        print(f"[✓] Appended '{vid_url}' to '{processed_file}'")

    return vid_url


def main():
    parser = argparse.ArgumentParser(
        description="Fetch latest unprocessed video from predefined YouTube RSS feeds (with yt-dlp fallback)."
    )
    parser.add_argument(
        "-f", "--processed-file",
        type=str,
        default="processed_videos.txt",
        help="Path to processed_videos.txt file (default: processed_videos.txt).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find the latest video without appending it to processed_videos.txt.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode: only output the YouTube URL to stdout.",
    )

    args = parser.parse_args()

    # In quiet mode, suppress stdout prints from fetch_latest_unprocessed_video
    if args.quiet:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            video_url = fetch_latest_unprocessed_video(
                processed_file=args.processed_file,
                mark_as_processed=not args.dry_run,
            )
        if video_url:
            print(video_url)
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        video_url = fetch_latest_unprocessed_video(
            processed_file=args.processed_file,
            mark_as_processed=not args.dry_run,
        )
        if video_url:
            print(f"\n[✓] Returned URL: {video_url}")
            sys.exit(0)
        else:
            print("\n[-] No video to process.")
            sys.exit(0)


if __name__ == "__main__":
    main()
