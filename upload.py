#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# YouTube Upload Scope
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Logging Configuration
LOG_FILE = "uploads.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("yt_uploader")


def get_authenticated_service(
    client_secrets_file: str = "client_secrets.json",
    token_file: str = "token.json",
):
    """
    Authenticate with YouTube Data API v3 using OAuth2.
    Loads credentials from token.json if available and valid.
    Refreshes expired credentials if refresh_token is available.
    Otherwise authenticates via client secrets file and saves token.json.
    """
    creds = None

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            logger.info(f"Loaded credentials from {token_file}")
        except Exception as e:
            logger.warning(f"Could not load token file '{token_file}': {e}. Re-authenticating...")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Credentials expired. Refreshing token...")
            try:
                creds.refresh(Request())
                logger.info("Token refreshed successfully.")
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}. Starting fresh auth flow...")
                creds = None

        if not creds:
            secret_path = client_secrets_file
            if not os.path.exists(secret_path):
                if os.path.exists("client_secret.json"):
                    secret_path = "client_secret.json"
                elif os.path.exists("client_secrets.json"):
                    secret_path = "client_secrets.json"

            if not os.path.exists(secret_path):
                raise FileNotFoundError(
                    f"OAuth2 client secrets file not found! Looked for '{client_secrets_file}', "
                    f"'client_secret.json', and 'client_secrets.json'."
                )

            logger.info(f"Initiating OAuth 2.0 authorization flow using '{secret_path}'...")
            flow = InstalledAppFlow.from_client_secrets_file(secret_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w", encoding="utf-8") as token_out:
            token_out.write(creds.to_json())
        logger.info(f"Saved credentials to '{token_file}'")

    return build("youtube", "v3", credentials=creds)


def resolve_video_metadata(
    video_path: str,
    analysis_json: str = "temp/analysis.json",
    clip_index: int = 0,
):
    """
    Extracts title and description for a given video from matching _meta.json or analysis.json.
    Sanitizes title to <= 100 characters and ensures #Shorts is present in title or description.
    """
    title = None
    description = None

    # 1. Look for matching _meta.json in same directory
    base_name = os.path.splitext(video_path)[0]
    meta_json_path = f"{base_name}_meta.json"
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                if isinstance(meta_data, dict):
                    title = meta_data.get("title")
                    description = meta_data.get("description")
                    logger.info(f"Loaded metadata from '{meta_json_path}'")
        except Exception as e:
            logger.warning(f"Failed reading metadata file '{meta_json_path}': {e}")

    # 2. Fallback to analysis.json if title or description is missing
    if (not title or not description) and os.path.exists(analysis_json):
        try:
            with open(analysis_json, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)

            clip_data = None
            numbers = re.findall(r"\d+", os.path.basename(video_path))
            clip_id = int(numbers[0]) if numbers else clip_index + 1

            if isinstance(analysis_data, list):
                for item in analysis_data:
                    if isinstance(item, dict) and item.get("clip_id") == clip_id:
                        clip_data = item
                        break
                if not clip_data and analysis_data:
                    idx = clip_index if clip_index < len(analysis_data) else 0
                    clip_data = analysis_data[idx] if isinstance(analysis_data[idx], dict) else None
            elif isinstance(analysis_data, dict):
                clip_data = analysis_data

            if clip_data:
                title = title or clip_data.get("title")
                description = description or clip_data.get("description")
                logger.info(f"Loaded metadata from '{analysis_json}'")
        except Exception as e:
            logger.warning(f"Failed reading analysis JSON '{analysis_json}': {e}")

    # 3. Default fallback if still missing
    if not title:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        title = f"{stem.replace('_', ' ').title()} #Shorts"

    if not description:
        description = "Check out this incredible short! #Shorts"

    title = title.strip()
    description = description.strip()

    # Ensure #Shorts is present in title or description
    has_shorts_tag = "#shorts" in (title + " " + description).lower()
    if not has_shorts_tag:
        if len(title) + 8 <= 100:
            title = f"{title} #Shorts"
        else:
            description = f"{description}\n\n#Shorts"

    # Sanitize title to under 100 characters
    if len(title) > 100:
        title = title[:100]

    return title, description


def upload_videos(
    ready_dir: str = "ready",
    analysis_json: str = "temp/analysis.json",
    client_secrets_file: str = "client_secrets.json",
    token_file: str = "token.json",
):
    """
    Finds all .mp4 files in ready_dir, uploads each with staggered scheduling,
    and removes uploaded .mp4 and _meta.json files upon success.
    """
    if not os.path.exists(ready_dir):
        logger.info("No videos to upload")
        return

    pattern = os.path.join(ready_dir, "*.mp4")
    video_files = glob.glob(pattern)

    if not video_files:
        logger.info("No videos to upload")
        return

    # Natural sort files
    def natural_sort_key(file_path: str):
        numbers = re.findall(r"\d+", os.path.basename(file_path))
        return (0, int(numbers[0])) if numbers else (1, file_path)

    video_files.sort(key=natural_sort_key)

    logger.info(f"Found {len(video_files)} video(s) to upload in '{ready_dir}'")

    youtube = get_authenticated_service(
        client_secrets_file=client_secrets_file,
        token_file=token_file,
    )

    base_time = datetime.now(timezone.utc)

    for i, video_path in enumerate(video_files):
        # Calculate publish time: 1st video +1 day, 2nd video +2 days, etc.
        scheduled_time = base_time + timedelta(days=i + 1)
        publish_at_str = scheduled_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        title, description = resolve_video_metadata(
            video_path=video_path,
            analysis_json=analysis_json,
            clip_index=i,
        )

        logger.info(f"Uploading video [{i + 1}/{len(video_files)}]: {video_path}")
        logger.info(f"Title: {title}")
        logger.info(f"Scheduled publish date: {publish_at_str}")

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": "private",
                "publishAt": publish_at_str,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                logger.info(f"Upload progress: {progress}%")

        video_id = response.get("id") if isinstance(response, dict) else None
        logger.info(f"Upload success! Video ID: {video_id}, Scheduled publish date: {publish_at_str}")

        # Delete the .mp4 file and matching _meta.json file from ready/
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.info(f"Deleted uploaded video file: '{video_path}'")
            except Exception as e:
                logger.warning(f"Could not delete video file '{video_path}': {e}")

        meta_json_path = f"{os.path.splitext(video_path)[0]}_meta.json"
        if os.path.exists(meta_json_path):
            try:
                os.remove(meta_json_path)
                logger.info(f"Deleted matching metadata file: '{meta_json_path}'")
            except Exception as e:
                logger.warning(f"Could not delete metadata file '{meta_json_path}': {e}")

    logger.info("All ready videos uploaded successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Shorts Uploader with staggered scheduling and automatic cleanup."
    )
    parser.add_argument(
        "--ready-dir",
        type=str,
        default="ready",
        help="Target directory containing ready .mp4 video files (default: 'ready').",
    )
    parser.add_argument(
        "--analysis-json",
        type=str,
        default="temp/analysis.json",
        help="Path to analysis JSON fallback (default: 'temp/analysis.json').",
    )
    parser.add_argument(
        "--client-secrets",
        type=str,
        default="client_secrets.json",
        help="Path to OAuth2 client secrets JSON file (default: 'client_secrets.json').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="token.json",
        help="Path to stored OAuth2 token JSON file (default: 'token.json').",
    )

    args = parser.parse_args()

    try:
        upload_videos(
            ready_dir=args.ready_dir,
            analysis_json=args.analysis_json,
            client_secrets_file=args.client_secrets,
            token_file=args.token,
        )
    except Exception as e:
        logger.error(f"Upload process encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
