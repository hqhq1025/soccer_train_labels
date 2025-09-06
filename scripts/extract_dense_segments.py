#!/usr/bin/env python3
"""
Extract dense event segments from SoccerNet-style Labels-v2.json files.

Core idea
- Bin events into fixed windows (default: 10s per bin).
- Find contiguous time ranges between min_len and max_len whose event density
  (events per bin) meets or exceeds a threshold.

Supports two modes:
- avg: segment's average density >= threshold (default)
- strict: every bin in the segment has count >= threshold

Segments are aligned to bin boundaries. By default, the tool returns
non-overlapping segments via a greedy selection from left to right
(choose the longest qualifying window at each start). Optionally, return
all qualifying windows (may be large) with --allow-overlap.

Usage examples
  Single file:
    python scripts/extract_dense_segments.py \
      --path "england_epl/2014-2015/.../Labels-v2.json" \
      --threshold 1 --bin-size 10 --min-seconds 20 --max-seconds 120

  Directory (recursively finds Labels-v2.json):
    python scripts/extract_dense_segments.py --path england_epl --threshold 1

  Only visible events, strict per-bin density >=2:
    python scripts/extract_dense_segments.py --path .../Labels-v2.json \
      --only-visible --mode strict --threshold 2

Return format: prints JSON to stdout with the segments per file, and
also human-readable table. Use --out to save JSON to a file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------- Data structures ----------------------------


@dataclass
class Event:
    position_ms: int
    label: str
    visibility: str
    team: str
    game_time: str


@dataclass
class Segment:
    start_s: float
    end_s: float
    bins: int
    events: int
    density_per_bin: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_seconds": round(self.start_s, 3),
            "end_seconds": round(self.end_s, 3),
            "length_seconds": round(self.end_s - self.start_s, 3),
            "bins": self.bins,
            "events": self.events,
            "density_per_bin": round(self.density_per_bin, 6),
            "start_time_mmss": seconds_to_mmss(self.start_s),
            "end_time_mmss": seconds_to_mmss(self.end_s),
        }


# ---------------------------- Helpers ----------------------------


def seconds_to_mmss(x: float) -> str:
    x = max(0.0, x)
    m = int(x // 60)
    s = int(round(x - 60 * m))
    return f"{m:02d}:{s:02d}"


def load_events(path: Path) -> List[Event]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = []
    for ann in data.get("annotations", []):
        pos_str = ann.get("position")
        try:
            pos_ms = int(pos_str)
        except Exception:
            # If position is missing or malformed, skip.
            continue
        events.append(
            Event(
                position_ms=pos_ms,
                label=str(ann.get("label", "")),
                visibility=str(ann.get("visibility", "")),
                team=str(ann.get("team", "")),
                game_time=str(ann.get("gameTime", "")),
            )
        )
    # Sort by time for deterministic behavior
    events.sort(key=lambda e: e.position_ms)
    return events


def filter_events(
    events: Sequence[Event],
    include_labels: Optional[Sequence[str]] = None,
    exclude_labels: Optional[Sequence[str]] = None,
    only_visible: bool = False,
    include_visibility: Optional[Sequence[str]] = None,
) -> List[Event]:
    include_set = set(l.lower() for l in include_labels) if include_labels else None
    exclude_set = set(l.lower() for l in exclude_labels) if exclude_labels else set()
    include_vis_set = (
        set(v.lower() for v in include_visibility) if include_visibility else None
    )

    out: List[Event] = []
    for e in events:
        label_l = e.label.lower()
        vis_l = e.visibility.lower()

        if include_set is not None and label_l not in include_set:
            continue
        if label_l in exclude_set:
            continue
        if only_visible and vis_l != "visible":
            continue
        if include_vis_set is not None and vis_l not in include_vis_set:
            continue
        out.append(e)
    return out


def build_bins(events: Sequence[Event], bin_size_s: int) -> Tuple[List[int], float]:
    if not events:
        return [], 0.0
    max_pos_ms = max(e.position_ms for e in events)
    duration_s = max_pos_ms / 1000.0
    bin_size_s = float(bin_size_s)
    n_bins = int(math.ceil((duration_s + 1e-9) / bin_size_s))
    counts = [0] * n_bins
    for e in events:
        idx = int(e.position_ms // int(bin_size_s * 1000))
        if 0 <= idx < n_bins:
            counts[idx] += 1
    return counts, duration_s


def prefix_sums(arr: Sequence[int]) -> List[int]:
    ps = [0]
    total = 0
    for x in arr:
        total += x
        ps.append(total)
    return ps


def window_sum(ps: Sequence[int], i: int, length: int) -> int:
    return ps[i + length] - ps[i]


def find_segments_avg(
    counts: Sequence[int],
    bin_size_s: int,
    threshold: float,
    min_bins: int,
    max_bins: int,
    allow_overlap: bool,
) -> List[Segment]:
    ps = prefix_sums(counts)
    n = len(counts)
    segments: List[Segment] = []
    i = 0
    while i < n:
        best: Optional[Tuple[int, int, float]] = None  # (length, sum, density)
        for L in range(max_bins, min_bins - 1, -1):
            if i + L > n:
                continue
            s = window_sum(ps, i, L)
            density = s / float(L)
            if density + 1e-12 >= threshold:
                best = (L, s, density)
                break  # prefer longest
        if best is None:
            i += 1
            continue
        L, s, density = best
        start_s = i * bin_size_s
        end_s = (i + L) * bin_size_s
        segments.append(
            Segment(
                start_s=start_s,
                end_s=end_s,
                bins=L,
                events=s,
                density_per_bin=density,
            )
        )
        if allow_overlap:
            i += 1
        else:
            i += L
    return segments


def find_segments_strict(
    counts: Sequence[int],
    bin_size_s: int,
    threshold: float,
    min_bins: int,
    max_bins: int,
    allow_overlap: bool,
) -> List[Segment]:
    # threshold is per-bin minimal count
    n = len(counts)
    segments: List[Segment] = []

    i = 0
    while i < n:
        if counts[i] < threshold:
            i += 1
            continue
        # Find run of bins all >= threshold
        j = i
        while j < n and counts[j] >= threshold:
            j += 1
        run_len = j - i
        # Carve non-overlapping windows within this run
        k = i
        while k < j:
            remaining = j - k
            if remaining < min_bins:
                break
            L = min(max_bins, remaining)
            start_s = k * bin_size_s
            end_s = (k + L) * bin_size_s
            s = sum(counts[k : k + L])
            density = s / float(L)
            segments.append(
                Segment(
                    start_s=start_s,
                    end_s=end_s,
                    bins=L,
                    events=s,
                    density_per_bin=density,
                )
            )
            if allow_overlap:
                k += 1
            else:
                k += L
        i = j
    return segments


def discover_label_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    out: List[Path] = []
    for p in path.rglob("Labels-v2.json"):
        out.append(p)
    return sorted(out)


def process_file(
    file_path: Path,
    threshold: float,
    bin_size_s: int,
    min_seconds: int,
    max_seconds: int,
    only_visible: bool,
    include_visibility: Optional[Sequence[str]],
    include_labels: Optional[Sequence[str]],
    exclude_labels: Optional[Sequence[str]],
    mode: str,
    allow_overlap: bool,
) -> Dict[str, object]:
    events = load_events(file_path)
    events = filter_events(
        events,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        only_visible=only_visible,
        include_visibility=include_visibility,
    )
    counts, duration_s = build_bins(events, bin_size_s=bin_size_s)
    min_bins = max(1, min_seconds // bin_size_s)
    max_bins = max(min_bins, max_seconds // bin_size_s)

    if mode == "avg":
        segments = find_segments_avg(
            counts, bin_size_s, threshold, min_bins, max_bins, allow_overlap
        )
    elif mode == "strict":
        segments = find_segments_strict(
            counts, bin_size_s, threshold, min_bins, max_bins, allow_overlap
        )
    else:
        raise ValueError("mode must be 'avg' or 'strict'")

    return {
        "file": str(file_path),
        "duration_seconds": round(duration_s, 3),
        "bin_size_seconds": bin_size_s,
        "threshold_per_bin": threshold,
        "mode": mode,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "total_events_considered": len(events),
        "segments": [s.to_dict() for s in segments],
    }


def print_human_readable(summary: Dict[str, object]) -> None:
    print(f"\nFile: {summary['file']}")
    print(
        "Settings: bin={}s, mode={}, threshold/bin={}, min={}s, max={}s".format(
            summary["bin_size_seconds"],
            summary["mode"],
            summary["threshold_per_bin"],
            summary["min_seconds"],
            summary["max_seconds"],
        )
    )
    segs = summary.get("segments", [])
    if not segs:
        print("  No segments found.")
        return
    print("  Segments (start~end [len], events, density/bin):")
    for seg in segs:
        start = seg["start_time_mmss"]
        end = seg["end_time_mmss"]
        ln = seg["length_seconds"]
        ev = seg["events"]
        den = seg["density_per_bin"]
        print(f"    {start} ~ {end}  [{ln:.1f}s], events={ev}, dens/bin={den:.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract dense event segments from SoccerNet Labels-v2.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--path",
        required=True,
        help="Path to Labels-v2.json or a directory to scan recursively",
    )
    p.add_argument("--threshold", type=float, default=1.0, help="Events per bin")
    p.add_argument("--bin-size", type=int, default=10, help="Bin size in seconds")
    p.add_argument("--min-seconds", type=int, default=20, help="Min segment length")
    p.add_argument("--max-seconds", type=int, default=120, help="Max segment length")
    p.add_argument(
        "--mode",
        choices=["avg", "strict"],
        default="avg",
        help="Density rule: avg over segment or strict per-bin",
    )
    p.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow overlapping segments (returns many windows)",
    )
    p.add_argument(
        "--only-visible",
        action="store_true",
        help="Keep only annotations with visibility == 'visible'",
    )
    p.add_argument(
        "--include-visibility",
        nargs="*",
        default=None,
        help="Whitelist of visibility values to include (case-insensitive). Overrides --only-visible if set.",
    )
    p.add_argument(
        "--include-label",
        dest="include_labels",
        nargs="*",
        default=None,
        help="Whitelist of labels to include (e.g., 'corner' 'goal')",
    )
    p.add_argument(
        "--exclude-label",
        dest="exclude_labels",
        nargs="*",
        default=None,
        help="Labels to exclude (e.g., 'ball out of play')",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON file to write aggregated results",
    )

    args = p.parse_args(argv)

    root = Path(args.path)
    files = discover_label_files(root)
    if not files:
        print(f"No Labels-v2.json found under: {root}", file=sys.stderr)
        return 2

    all_summaries: List[Dict[str, object]] = []
    for f in files:
        summary = process_file(
            f,
            threshold=args.threshold,
            bin_size_s=args.bin_size,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            only_visible=args.only_visible,
            include_visibility=args.include_visibility,
            include_labels=args.include_labels,
            exclude_labels=args.exclude_labels,
            mode=args.mode,
            allow_overlap=args.allow_overlap,
        )
        all_summaries.append(summary)
        print_human_readable(summary)

    # Also print JSON to stdout
    as_json = {
        "path": str(root),
        "file_count": len(all_summaries),
        "results": all_summaries,
    }
    text = json.dumps(as_json, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\nSaved JSON to: {args.out}")
    else:
        print("\nJSON summary:")
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

