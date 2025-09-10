"""
Download all SoccerNet 224p videos for train/valid/test splits, with parallelism.

Requirements:
- pip install SoccerNet
- Set NDA password via env var SOCCERNET_PASSWORD (recommended)

Usage:
  SOCCERNET_PASSWORD=your_password \
  python scripts/download_224p_videos.py --out ./data/SoccerNet --workers 8

Notes:
- This script only requests 224p videos: "1_224p.mkv" and "2_224p.mkv".
- If your dataset has additional 224p files, add them to FILES list.
"""

import os
import time
import argparse
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    # Library name is "SoccerNet" on PyPI
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "SoccerNet package is required. Install with: pip install SoccerNet\n"
        f"Import error: {e}"
    )


FILES_224P: List[str] = [
    "1_224p.mkv",
    "2_224p.mkv",
]


def main():
    parser = argparse.ArgumentParser(description="Download SoccerNet 224p videos")
    parser.add_argument(
        "--out",
        default=os.path.join(".", "data", "SoccerNet"),
        help="Local download directory (default: ./data/SoccerNet)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help='Splits to download (default: "train valid test")',
    )
    parser.add_argument(
        "--task",
        default="spotting",
        help='Task for game list (default: "spotting")',
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per game for missing files (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume by skipping games whose files already exist (default: on)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("SOCCERNET_PASSWORD", ""),
        help="NDA password (or set SOCCERNET_PASSWORD env var)",
    )
    args = parser.parse_args()

    if not args.password:
        raise SystemExit(
            "Missing NDA password. Set --password or SOCCERNET_PASSWORD env var."
        )

    os.makedirs(args.out, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=args.out)
    # Set NDA password as in SoccerNet examples
    downloader.password = args.password

    # Helpers for resume/need
    def missing_files_for(game_dir: str) -> List[str]:
        missing = []
        for f in FILES_224P:
            if not os.path.exists(os.path.join(game_dir, f)):
                missing.append(f)
        return missing

    # Build (split, game) list (with resume: only those with missing files)
    jobs: List[Tuple[str, str]] = []
    for spl in args.splits:
        games = getListGames(split=spl, task=args.task, dataset="SoccerNet")
        for g in games:
            game_dir = os.path.join(args.out, g)
            missing = missing_files_for(game_dir)
            if not args.resume or len(missing) > 0:
                jobs.append((spl, g))

    print(f"Found {len(jobs)} games to fetch across splits {args.splits}. Starting {args.workers} workers...")

    # Download per-game concurrently
    errors: List[Tuple[str, str, Exception]] = []
    completed = 0

    def _download_one(spl: str, game: str) -> str:
        game_dir = os.path.join(args.out, game)
        attempt = 0
        backoff = 2
        while attempt <= args.retries:
            attempt += 1
            missing = missing_files_for(game_dir)
            if not missing:
                break
            try:
                downloader.downloadGame(game=game, files=missing, spl=spl, verbose=True)
            except Exception as e:
                if attempt > args.retries:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
        # Final check
        missing = missing_files_for(game_dir)
        if missing:
            raise RuntimeError(f"missing after retries: {missing}")
        return game

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_job = {ex.submit(_download_one, spl, game): (spl, game) for spl, game in jobs}
        for fut in as_completed(future_to_job):
            spl, game = future_to_job[fut]
            try:
                _ = fut.result()
            except Exception as e:  # collect but continue
                errors.append((spl, game, e))
                print(f"[ERROR] {spl}/{game}: {e}")
            finally:
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    print(f"Progress: {completed}/{len(jobs)} games")

    if errors:
        print(f"Done with {len(errors)} errors. Output dir: {os.path.abspath(args.out)}")
    else:
        print("All downloads completed successfully. Output dir:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
