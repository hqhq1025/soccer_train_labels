#!/usr/bin/env python3
"""
Download SoccerNet videos using the official Python API.

Two common usages:

1) Download specific game(s):
   python scripts/download_videos.py \
     --games "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" \
     --files 1_720p.mkv 2_720p.mkv \
     --out-dir /path/to/save

2) Download every game found under your local labels folder (raw_jsons):
   python scripts/download_videos.py \
     --labels-root raw_jsons \
     --files 1_720p.mkv 2_720p.mkv \
     --out-dir /path/to/save

Notes:
- Default password is "s0cc3rn3t". You can also set SOCCERNET_PASSWORD env var.
- If 720p is not available for a game, try: --files 1.mkv 2.mkv or 1_224p.mkv 2_224p.mkv
- This script requires: pip install SoccerNet --upgrade
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def find_game_ids_from_labels(labels_root: Path) -> List[str]:
    """Discover game IDs from a local SoccerNet labels mirror (raw_jsons).

    Expected structure:
      raw_jsons/<competition>/<season>/<game>/Labels-v*.json

    Returns a list of game IDs formatted as
      "<competition>/<season>/<game>"
    which matches the IDs expected by the SoccerNet downloader.
    """
    labels_root = labels_root.resolve()
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root not found: {labels_root}")

    game_ids: List[str] = []
    # Match both Labels-v1.json and Labels-v2.json if present
    for labels_path in labels_root.rglob("Labels-*.json"):
        # Handle both patterns:
        #   raw_jsons/<competition>/<season>/<game>/Labels-*.json
        #   raw_jsons/<split>_labels/<competition>/<season>/<game>/Labels-*.json
        try:
            rel = labels_path.relative_to(labels_root)
        except Exception:
            continue

        parts = rel.parts
        # Need at least: <a>/<b>/<c>/Labels-*.json
        if len(parts) < 4:
            continue

        competition = parts[-4]
        season = parts[-3]
        game = parts[-2]
        game_id = f"{competition}/{season}/{game}"
        game_ids.append(game_id)

    # De-duplicate and sort for stable order
    return sorted(set(game_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SoccerNet videos via API")
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--labels-root",
        type=Path,
        default=Path("raw_jsons"),
        help="Root folder of local labels (to auto-discover games)",
    )
    src.add_argument(
        "--games",
        nargs="*",
        help="Explicit game ID(s), e.g. 'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'",
    )
    # Split-based remote download mode (official API)
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=["train", "valid", "test", "challenge"],
        help="Download by split(s) using SoccerNet's index (bypasses local labels)",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Shortcut for --splits train valid test challenge",
    )

    parser.add_argument(
        "--files",
        nargs="+",
        default=["1_720p.mkv", "2_720p.mkv"],
        help="Video filenames to fetch per game (order matters)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/SoccerNet"),
        help="Local directory where videos will be stored",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("SOCCERNET_PASSWORD", "s0cc3rn3t"),
        help="Password for SoccerNet videos (or set SOCCERNET_PASSWORD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of games downloaded",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded, do not download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure env var is set for broad compatibility across SoccerNet versions
    if args.password:
        os.environ["SOCCERNET_PASSWORD"] = args.password

    # Lazy import to keep CLI responsive for --dry-run
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except Exception as e:
        raise SystemExit(
            "Could not import SoccerNet. Did you run 'pip install SoccerNet --upgrade'?\n"
            f"Original error: {e}"
        )

    # Some versions don't accept a password kwarg; they read env instead
    try:
        downloader = SoccerNetDownloader(LocalDirectory=str(args.out_dir))
    except TypeError:
        # Fallback: older API name casing variations
        try:
            downloader = SoccerNetDownloader(local_directory=str(args.out_dir))
        except Exception as e:
            raise SystemExit(f"Could not initialize SoccerNetDownloader: {e}")

    # Best-effort: set password attribute if supported
    try:
        if getattr(downloader, "password", None) != args.password and args.password:
            setattr(downloader, "password", args.password)
    except Exception:
        pass

    # If splits specified (or all-splits), use official split-based download
    if args.all_splits or args.splits:
        splits = ["train", "valid", "test", "challenge"] if args.all_splits else args.splits
        print(f"Split-based download. Splits: {splits}")
        print(f"Saving into: {args.out_dir}")
        print(f"Files: {', '.join(args.files)}")

        if args.dry_run:
            print(f"DRY-RUN: would call downloader.downloadGames(files={args.files}, split={splits})")
            return

        try:
            downloader.downloadGames(files=args.files, split=splits)
            print("\nSplit-based download finished.")
        except Exception as e:
            raise SystemExit(f"downloadGames failed: {e}")
        return

    # Otherwise, resolve games from explicit IDs or local labels and download per-game
    if args.games:
        game_ids = list(dict.fromkeys(args.games))  # preserve order, de-dup
    else:
        game_ids = find_game_ids_from_labels(Path(args.labels_root))

    if args.limit is not None:
        game_ids = game_ids[: args.limit]

    if not game_ids:
        print("No games found. Provide --games or check --labels-root.")
        return

    print(f"Games to download: {len(game_ids)}")
    print(f"Saving into: {args.out_dir}")
    print(f"Files to fetch per game: {', '.join(args.files)}")

    if args.dry_run:
        for gid in game_ids:
            print(f"DRY-RUN: would download {args.files} for {gid}")
        return

    failures: List[str] = []
    for i, gid in enumerate(game_ids, 1):
        print(f"[{i}/{len(game_ids)}] Downloading: {gid}")
        try:
            downloader.downloadGame(gid, files=args.files)
        except Exception as e:
            print(f"  Failed: {gid} -> {e}")
            failures.append(gid)

    if failures:
        print("\nSome games failed to download (see above):")
        for gid in failures:
            print(f" - {gid}")
        print("Tip: try different --files, e.g. '1.mkv 2.mkv' or lower resolutions.")
    else:
        print("\nAll requested games downloaded successfully.")


if __name__ == "__main__":
    main()
