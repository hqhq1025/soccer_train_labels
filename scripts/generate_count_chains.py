#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TEAMFUL = {"left", "right"}
HALF_OFFSET_SECONDS = 45 * 60


def parse_game_time_seconds(game_time: str, fallback_ms: Optional[int] = None) -> Optional[float]:
    try:
        half_str, mmss = [t.strip() for t in str(game_time).split("-")]
        half = int(half_str)
        mm, ss = [int(x) for x in mmss.split(":")]
        sec = mm * 60 + ss
        if half >= 2:
            sec = (half - 1) * HALF_OFFSET_SECONDS + sec
        return float(sec)
    except Exception:
        if fallback_ms is not None:
            return float(fallback_ms) / 1000.0
        return None


def iter_segment_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("segment_*.json"):
        if p.is_file():
            yield p


@dataclass
class Ev:
    pos_s: float
    label: str


COUNT_TEMPLATES_EN = [
    "Track {EVENT}. Each time it occurs, output the cumulative count since the question started as a number only (e.g., 3).",
    "Monitor {EVENT}. Whenever it happens, respond with the running total since start using digits only.",
    "Watch for {EVENT}; on each occurrence, return the cumulative number since question time as an Arabic numeral.",
    "Please track {EVENT}; each time it occurs, output the running total as a number only.",
    "Observe {EVENT} and report the cumulative occurrence count since start using digits only on every occurrence.",
]


def format_event_set(ev: str) -> str:
    return ev


def build_count_for_segment(
    seg_path: Path,
    rng: random.Random,
    include_not_shown: bool,
    min_gap_seconds: float,
    min_occurrences: int,
) -> Optional[Dict[str, object]]:
    data = json.loads(seg_path.read_text(encoding="utf-8"))
    seg_meta = data.get("segment", {})
    s_start = float(seg_meta.get("start_seconds", 0.0))
    s_end = float(seg_meta.get("end_seconds", s_start))
    src_file = str(data.get("source_file", ""))

    evs: List[Ev] = []
    for a in data.get("annotations", []) or []:
        vis = str(a.get("visibility", "")).lower()
        if not include_not_shown and vis == "not shown":
            continue
        raw_pos_ms = None
        try:
            raw_pos_ms = int(a.get("position"))
        except Exception:
            raw_pos_ms = None
        gt = str(a.get("gameTime", "")).strip()
        pos_s = parse_game_time_seconds(gt, fallback_ms=raw_pos_ms)
        if pos_s is None:
            continue
        if not (s_start <= pos_s < s_end):
            continue
        label = str(a.get("label", "")).strip()
        evs.append(Ev(pos_s=pos_s, label=label))

    if not evs:
        return None
    evs.sort(key=lambda e: (e.pos_s, e.label.lower()))

    # Choose a label with the most occurrences (after min_gap), prefer higher-count labels
    by_label: Dict[str, List[Ev]] = {}
    for e in evs:
        by_label.setdefault(e.label, []).append(e)
    best_label = None
    best_count = -1
    for lab, lst in by_label.items():
        c = sum(1 for x in lst if x.pos_s >= s_start + min_gap_seconds)
        if c > best_count:
            best_count = c
            best_label = lab
    if best_label is None or best_count < max(1, int(min_occurrences)):
        return None
    target = best_label

    # Build question
    q_text = rng.choice(COUNT_TEMPLATES_EN).replace("{EVENT}", f"{{{format_event_set(target)}}}")
    q_list = [{"time": 0.0, "count": 0, "text": q_text}]

    # Build answers (segment-relative) with running count
    cnt = 0
    ans: List[Dict[str, object]] = []
    for e in evs:
        if e.label != target:
            continue
        rel = e.pos_s - s_start
        if rel < min_gap_seconds:
            continue
        cnt += 1
        # Answer text: digits only
        ans.append({"start": round(rel, 6), "end": round(rel, 6), "count": 0, "text": f"{cnt}"})

    if not ans:
        return None

    from uuid import uuid5, NAMESPACE_URL
    seg_id = str(uuid5(NAMESPACE_URL, f"{src_file}#{seg_path.name}"))

    pub = {
        "source": "soccer",
        "id": 0,
        "video_id": seg_id,
        "data_type": "online",
        "train_stage": 2,
        "length": round(max(0.0, s_end - s_start), 6),
        "question_category": "count",
        "question": q_list,
        "answer": ans,
        "_raw_source_file": src_file or None,
        "_target_label": target,
    }
    return pub


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate count chain QA from dense segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base-dir", type=str, required=True)
    ap.add_argument("--per-file-root", type=str, default="out/count")
    ap.add_argument("--per-file-subdir", type=str, default="count")
    ap.add_argument("--include-not-shown", action="store_true")
    ap.add_argument("--random-seed", type=int, default=7001)
    ap.add_argument("--per-segment-max", type=int, default=0, help="<=0 for unlimited")
    ap.add_argument("--max-files", type=int, default=0, help="<=0 for unlimited")
    ap.add_argument("--min-gap-seconds", type=float, default=0.5)
    ap.add_argument("--min-occurrences", type=int, default=1, help="Require at least this many occurrences after question time")

    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Base dir not found: {base}")
        return 2
    rng = random.Random(args.random_seed)
    out_root = Path(args.per_file_root)
    written = 0
    for seg in iter_segment_files(base):
        per_seg = 0
        seen = set()
        attempts = 0
        while True:
            attempts += 1
            item = build_count_for_segment(
                seg,
                rng,
                include_not_shown=bool(args.include_not_shown),
                min_gap_seconds=float(args.min_gap_seconds),
                min_occurrences=int(args.min_occurrences),
            )
            if not item:
                if attempts >= 6:
                    break
                else:
                    continue
            raw_src = item.get("_raw_source_file")
            if not raw_src:
                break
            label = str(item.get("_target_label", ""))
            if label in seen:
                # In single-event segments the label set is fixed; avoid spinning
                break
            seen.add(label)

            parts = Path(str(raw_src)).parts
            try:
                idx = parts.index("raw_jsons")
                rel_inside = Path(*parts[idx + 1 : -1])
            except ValueError:
                rel_inside = Path(str(raw_src)).parent
            target_dir = out_root / rel_inside
            subdir = (args.per_file_subdir or "").strip()
            if subdir:
                target_dir = target_dir / subdir
            target_dir.mkdir(parents=True, exist_ok=True)

            # filename includes segment start and label
            try:
                seg_obj = json.loads(Path(seg).read_text(encoding="utf-8"))
                s0 = float(seg_obj.get("segment", {}).get("start_seconds", 0.0))
            except Exception:
                s0 = 0.0
            safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)[:30]
            fname = f"count_{int(round(s0*1000)):07d}_{safe_label}.json"
            item["id"] = written + 1
            item.pop("_raw_source_file", None)
            item.pop("_target_label", None)
            (target_dir / fname).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
            per_seg += 1
            if args.max_files > 0 and written >= args.max_files:
                break
            if args.per_segment_max > 0 and per_seg >= args.per_segment_max:
                break
        if args.max_files > 0 and written >= args.max_files:
            break
    print(f"Wrote {written} count files under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
