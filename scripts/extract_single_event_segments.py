#!/usr/bin/env python3
"""
Extract single-event dense segments across a dataset of SoccerNet-style Labels-v2.json.

For each distinct label (event type), the script:
  1) Filters annotations to that label only (ignoring "not shown" by default).
  2) Bins events into fixed windows (default 10s per bin).
  3) Finds contiguous time ranges between --min-seconds and --max-seconds whose
     density (events per bin) meets or exceeds --threshold, using either
     average or strict per-bin mode.
  4) Exports per-segment JSONs containing ONLY that label's annotations, with
     output directory mirroring the input tree under: out_dir/<label>/...

Flexible and fast by reusing core logic from scripts/extract_dense_segments.py
via importlib (no code copy, no repo refactor required).

Example:
  python scripts/extract_single_event_segments.py \
    --path england_epl \
    --threshold 1 --bin-size 10 --min-seconds 20 --max-seconds 120 \
    --out-dir dense_single_event_segments \
    --out dense_single_event_segments/summary.json

Notes
  - By default, events with visibility == 'not shown' are ignored.
    Use --include-not-shown to include them.
  - You can restrict to visible only via --only-visible.
  - Use --labels to limit to specific labels; otherwise all labels are traversed.
  - To avoid filename collisions, segments for different labels are written
    under separate label subdirectories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from importlib.machinery import SourceFileLoader


# Load the existing dense segment helper module dynamically
_EDS = SourceFileLoader(
    "_eds",
    str(Path(__file__).parent / "extract_dense_segments.py"),
).load_module()


def _normalize_label(x: Optional[str]) -> str:
    return (x or "").strip()


def _label_key(x: Optional[str]) -> str:
    return (x or "").strip().lower()


def _safe_label_dir(label: str) -> str:
    # Lowercase, replace spaces and unsafe chars
    s = _label_key(label)
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {" ", "/", "\\", ":", "|", "*", "?", "\"", "<", ">"}:
            out.append("-")
        else:
            out.append("_")
    # Collapse repeats
    import re

    cleaned = re.sub(r"[-_]{2,}", r"-", "".join(out)).strip("-_")
    return cleaned or "_unknown"


def discover_label_files(path: Path) -> List[Path]:
    # Reuse _EDS implementation if available
    return _EDS.discover_label_files(path)


def collect_distinct_labels(files: Sequence[Path], include_not_shown: bool) -> List[str]:
    labels: Set[str] = set()
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for a in data.get("annotations", []):
            vis = (a.get("visibility") or "").lower()
            if not include_not_shown and vis == "not shown":
                continue
            lab = a.get("label")
            if lab is None:
                continue
            labels.add(_normalize_label(lab))
    # Sort case-insensitively but keep original spelling where possible
    return sorted(labels, key=lambda x: x.lower())


@dataclass
class LabelRunSummary:
    label: str
    file_count: int
    segment_files_written: int
    total_segments: int


def run_for_label(
    label: str,
    files: Sequence[Path],
    threshold: float,
    bin_size_s: int,
    min_seconds: int,
    max_seconds: int,
    only_visible: bool,
    include_visibility: Optional[Sequence[str]],
    mode: str,
    allow_overlap: bool,
    lead_seconds: float,
    include_not_shown: bool,
    out_dir: Path,
    root: Path,
) -> Tuple[LabelRunSummary, List[Dict[str, object]]]:
    include_labels = [label]
    exclude_labels = None
    label_dir = out_dir / _safe_label_dir(label)
    label_dir.mkdir(parents=True, exist_ok=True)

    per_file_summaries: List[Dict[str, object]] = []
    seg_written_total = 0
    seg_count_total = 0

    # Determine base for mirroring directory layout
    root_base: Optional[Path] = root if root.is_dir() else root.parent

    for f in files:
        summary = _EDS.process_file(
            file_path=f,
            threshold=threshold,
            bin_size_s=bin_size_s,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            only_visible=only_visible,
            include_visibility=include_visibility,
            include_labels=include_labels,
            exclude_labels=exclude_labels,
            mode=mode,
            allow_overlap=allow_overlap,
            lead_seconds=lead_seconds,
            include_not_shown=include_not_shown,
            export=True,
            out_dir=label_dir,
            root_base=root_base,
        )
        # Count segments and files written: export happens inside process_file
        segs = summary.get("segments", [])
        seg_count = len(segs) if isinstance(segs, list) else 0
        seg_count_total += seg_count
        if seg_count:
            seg_written_total += seg_count
        # Attach label to summary entry
        summary["single_label"] = label
        per_file_summaries.append(summary)

    return (
        LabelRunSummary(
            label=label,
            file_count=len(files),
            segment_files_written=seg_written_total,
            total_segments=seg_count_total,
        ),
        per_file_summaries,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract dense segments for single events (one label at a time)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--path", required=True, help="Path to file or directory to scan for Labels-v2.json")
    p.add_argument("--threshold", type=float, default=1.0, help="Events per bin (for the single label)")
    p.add_argument("--bin-size", type=int, default=10, help="Bin size in seconds")
    p.add_argument("--min-seconds", type=int, default=20, help="Min segment length")
    p.add_argument("--max-seconds", type=int, default=120, help="Max segment length")
    p.add_argument("--lead-seconds", type=float, default=5.0, help="First event occurs at least this many seconds after segment start")
    p.add_argument("--mode", choices=["avg", "strict"], default="avg", help="Density rule: avg over segment or strict per-bin")
    p.add_argument("--allow-overlap", action="store_true", help="Allow overlapping segments (may be many windows)")
    p.add_argument("--only-visible", action="store_true", help="Keep only annotations with visibility == 'visible'")
    p.add_argument("--include-visibility", nargs="*", default=None, help="Whitelist of visibility values to include; overrides --only-visible")
    p.add_argument("--include-not-shown", action="store_true", help="Include 'not shown' events as well")
    p.add_argument("--labels", nargs="*", default=None, help="Optional set of labels to restrict (case-insensitive)")
    p.add_argument("--exclude-labels", nargs="*", default=None, help="Labels to skip (case-insensitive)")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON file to write aggregated results")
    p.add_argument("--out-dir", type=str, default="dense_single_event_segments", help="Directory to write per-segment JSON files; segments are placed under <out-dir>/<label>/...")

    args = p.parse_args(argv)

    root = Path(args.path)
    files = discover_label_files(root)
    if not files:
        print(f"No Labels-v2.json found under: {root}", file=sys.stderr)
        return 2

    # Decide which labels to traverse
    if args.labels:
        label_whitelist = {_label_key(l) for l in args.labels}
    else:
        label_whitelist = None
    label_blacklist = {_label_key(l) for l in (args.exclude_labels or [])}

    all_labels = collect_distinct_labels(files, include_not_shown=args.include_not_shown)
    run_labels = []
    for lab in all_labels:
        lk = _label_key(lab)
        if label_whitelist is not None and lk not in label_whitelist:
            continue
        if lk in label_blacklist:
            continue
        run_labels.append(lab)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_summaries: List[Dict[str, object]] = []
    grand_segments = 0
    for lab in run_labels:
        print(f"\n[Label] {lab}")
        run_sum, per_file = run_for_label(
            label=lab,
            files=files,
            threshold=args.threshold,
            bin_size_s=args.bin_size,
            min_seconds=args.min_seconds,
            max_seconds=args.max_seconds,
            only_visible=args.only_visible,
            include_visibility=args.include_visibility,
            mode=args.mode,
            allow_overlap=args.allow_overlap,
            lead_seconds=args.lead_seconds,
            include_not_shown=args.include_not_shown,
            out_dir=out_dir,
            root=root,
        )
        label_summaries.append(
            {
                "label": lab,
                "file_count": run_sum.file_count,
                "segments_total": run_sum.total_segments,
                "segment_files_written": run_sum.segment_files_written,
            }
        )
        grand_segments += run_sum.total_segments

    as_json = {
        "path": str(root),
        "label_count": len(run_labels),
        "labels": run_labels,
        "total_segments": grand_segments,
        "settings": {
            "threshold_per_bin": args.threshold,
            "bin_size_seconds": args.bin_size,
            "min_seconds": args.min_seconds,
            "max_seconds": args.max_seconds,
            "lead_seconds": args.lead_seconds,
            "mode": args.mode,
            "allow_overlap": bool(args.allow_overlap),
            "only_visible": bool(args.only_visible),
            "include_visibility": args.include_visibility,
            "include_not_shown": bool(args.include_not_shown),
        },
        "per_label_summary": label_summaries,
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

