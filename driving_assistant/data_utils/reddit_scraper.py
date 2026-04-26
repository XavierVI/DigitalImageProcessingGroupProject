import time

import requests
import yt_dlp
import os
import re

import argparse


def _safe_filename(name, max_length=120):
    # Keep filenames portable by removing characters invalid on common filesystems.
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', name).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return (cleaned[:max_length]).rstrip(' .') or 'video'


def scrape_without_api(subreddit, limit=10):
    # 1. Hit the JSON endpoint directly
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

    # You MUST set a User-Agent or Reddit will block you
    headers = {'User-Agent': 'Mozilla/5.0 (DIP Project Scraper)'}

    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        print(f"Failed to fetch data: {response.status_code}")
        return

    data = response.json()
    posts = data['data']['children']

    os.makedirs("downloads", exist_ok=True)
    output_dir = "data/reddit_dashcam_videos"
    os.makedirs(output_dir, exist_ok=True)
    index_offset = len(os.listdir(output_dir))

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        # %(title).50s clips the title to 50 chars to avoid "Filename too long" errors
        # %(id)s ensures uniqueness
        'outtmpl': 'downloads/%(id)s.%(ext)s',
        'merge_output_format': 'mp4',
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, post in enumerate(posts):
            post_data = post['data']
            # Filter for video posts
            if post_data.get('is_video') and post_data.get('media', {}).get('reddit_video'):
                video_url = post_data['media']['reddit_video']['fallback_url']
                title = post_data.get('title', 'video')
                post_id = post_data.get('id', 'unknown')
                safe_title = _safe_filename(title)
                print(f"Downloading: {title}")
                try:
                    info = ydl.extract_info(video_url, download=True)

                    download_path = None
                    requested = info.get('requested_downloads') if isinstance(
                        info, dict) else None
                    if requested and requested[0].get('filepath'):
                        download_path = requested[0]['filepath']
                    elif isinstance(info, dict) and info.get('_filename'):
                        download_path = info['_filename']
                    elif isinstance(info, dict):
                        download_path = ydl.prepare_filename(info)

                    if not download_path or not os.path.exists(download_path):
                        raise FileNotFoundError(
                            f"Could not resolve downloaded file path for post {post_id}"
                        )

                    ext = os.path.splitext(download_path)[1] or '.mp4'
                    # name the file "video_i"
                    destination = os.path.join(output_dir, f"video_{i + index_offset}{ext}")
                    os.replace(download_path, destination)
                except Exception as e:
                    print(f"Skipping {video_url}: {e}")

            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape Reddit videos without API")
    parser.add_argument("--subreddit", type=str,
                        default="dashcams", help="Subreddit to scrape")
    parser.add_argument("--limit", type=int, default=25,
                        help="Number of posts to scrape")
    args = parser.parse_args()

    scrape_without_api(args.subreddit, limit=args.limit)
