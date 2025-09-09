#!/usr/bin/env python3
"""
Generate switch-trace chain QA from dense segment JSONs.

Category: switch_trace

Behavior per segment (unit=segment):
- Q1 (count=0) at segment start: track either a single label (Q_single) or a set of two labels (Q_multiple).
- Q2 (count=1) inserted later in the same segment: explicitly instructs to stop Q1 tracking and switch to Q2.
- Q1 and Q2 must be different kinds (single vs multiple), and track disjoint label sets.
- Answers for Q1 are only from [Q1_time + min_gap, Q2_time); answers for Q2 are from [Q2_time + min_gap, seg_end].

Schema (one JSON per QA):
- source: soccer
- id: 0..N
- video_id: stable UUID based on raw source + segment filename
- data_type: online
- train_stage: 2
- length: segment length (seconds)
- question_category: switch_trace
- question: array
  [
    {time: 0.0, count: 0, text: "(Q1) ..."},
    {time: t2,  count: 1, text: "(Q2) ..."}
  ]
- answer: interleaved array with {start, end, count, text}, times are segment-relative

Usage example
  python scripts/generate_switch_trace_chains.py \
    --base-dir train/dense_segments \
    --per-file-root out/switch_trace \
    --per-file-subdir switch_trace \
    --min-gap-seconds 3.0 \
    --per-segment-max 0 \
    --max-files 0
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TEAMFUL = {"left", "right"}


def slug_to_uuid(s: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def iter_segment_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("segment_*.json"):
        if p.is_file():
            yield p


@dataclass
class Ann:
    pos_s: float
    label: str
    team: str
    game_time: str


TEMPLATES_SINGLE = [
    "Track {EVENT_SET} and alert me whenever it occurs. Answer exactly: 'Alert - <Event>'.",
    "Please monitor {EVENT_SET} and notify me each time it happens. Use 'Alert - <Event>'.",
    "Watch for {EVENT_SET} and tell me when it occurs. Reply 'Alert - <Event>'.",
    "Keep an eye on {EVENT_SET}; alert me on occurrence. Answer format: 'Alert - <Event>'.",
    "Observe {EVENT_SET} and report every time it happens. Respond 'Alert - <Event>'.",
]

TEMPLATES_MULTIPLE = [
    "Track {EVENT_SET} and alert me whenever they occur. Reply 'Alert - <Event>'.",
    "Please monitor {EVENT_SET} and notify me each time they happen. Use 'Alert - <Event>'.",
    "Watch for {EVENT_SET} and tell me when they occur. Answer: 'Alert - <Event>'.",
    "Keep an eye on {EVENT_SET}; alert me on occurrences. Respond 'Alert - <Event>'.",
    "Observe {EVENT_SET} and report every time they happen. Exact text 'Alert - <Event>'.",
]

TEMPLATES_SWITCH = [
    "Stop the previous tracking and switch to continuously monitor {EVENT_SET}. Respond as 'Alert - <Event>'.",
    "Cease the prior task and start continuously tracking {EVENT_SET} now. Reply 'Alert - <Event>'.",
    "Stop tracking before and begin to continuously monitor {EVENT_SET}. Use 'Alert - <Event>'.",
    "End the previous tracking; from now on continuously track {EVENT_SET}. Answer 'Alert - <Event>'.",
    "Stop the last one and continuously follow {EVENT_SET} from now on. Respond 'Alert - <Event>'.",
]


def _format_event_set(labels: Sequence[str]) -> str:
    labs = [l for l in labels if l]
    if not labs:
        return ""
    if len(labs) == 1:
        return labs[0]
    return ", ".join(labs[:-1]) + " and " + labs[-1]


def _uniq_labels_in_order(evs: Sequence[Ann]) -> List[str]:
    seen = set()
    out: List[str] = []
    for e in evs:
        if e.label not in seen:
            seen.add(e.label)
            out.append(e.label)
    return out


def choose_q2_time(rng: random.Random, seg_start: float, seg_end: float, event_times: List[float]) -> float:
    if not event_times:
        return min(seg_end - 1e-3, seg_start + 5.0)
    # pick between 30%~80% percentile of event times
    n = len(event_times)
    lo = max(0, int(n * 0.3))
    hi = max(lo + 1, int(n * 0.8))
    t = event_times[rng.randrange(lo, hi)]
    return float(max(seg_start, min(t, seg_end - 1e-3)))


def build_switch_for_segment(
    seg_path: Path,
    rng: random.Random,
    include_not_shown: bool,
    min_gap_seconds: float,
) -> Optional[Dict[str, object]]:
    data = json.loads(seg_path.read_text(encoding="utf-8"))
    seg_meta = data.get("segment", {})
    seg_start = float(seg_meta.get("start_seconds", 0.0))
    seg_end = float(seg_meta.get("end_seconds", seg_start))
    src_file = str(data.get("source_file", ""))

    anns_raw = data.get("annotations", []) or []
    evs: List[Ann] = []
    for a in anns_raw:
        vis = str(a.get("visibility", "")).lower()
        if not include_not_shown and vis == "not shown":
            continue
        try:
            pos_s = int(a.get("position")) / 1000.0
        except Exception:
            continue
        lab = str(a.get("label", ""))
        team = str(a.get("team", "")).lower() or "not applicable"
        gt = str(a.get("gameTime", ""))
        if not (seg_start <= pos_s < seg_end):
            continue
        evs.append(Ann(pos_s=pos_s, label=lab, team=team, game_time=gt))

    if not evs:
        return None
    evs.sort(key=lambda e: (e.pos_s, e.label.lower()))
    labels_any = _uniq_labels_in_order(evs)
    if len(labels_any) < 2:
        return None

    # Decide Q1 kind and Q2 kind (different)
    kinds = ["single", "multiple"]
    q1_kind = rng.choice(kinds)
    q2_kind = "multiple" if q1_kind == "single" else "single"

    # Build label pools
    def pick_labels(kind: str, pool: Sequence[str]) -> List[str]:
        if kind == "single" or len(pool) == 1:
            return [rng.choice(list(pool))]
        # multiple: pick 2 distinct labels
        if len(pool) == 2:
            return list(pool)
        # choose 2 without replacement
        return rng.sample(list(pool), k=2)

    # Q1 labels
    q1_labels = pick_labels(q1_kind, labels_any)

    # Q2 labels: disjoint and must have occurrences after q2_time
    event_times = [e.pos_s for e in evs]
    q2_time = choose_q2_time(rng, seg_start, seg_end, event_times)

    # Build candidate pool for Q2
    after = [e for e in evs if e.pos_s >= (q2_time + min_gap_seconds)]
    labels_after = _uniq_labels_in_order(after)
    labels_after = [l for l in labels_after if l not in set(q1_labels)]
    if not labels_after:
        return None
    q2_labels = pick_labels(q2_kind, labels_after)

    # Compose questions (segment-relative times)
    q1_text = (rng.choice(TEMPLATES_SINGLE) if q1_kind == "single" else rng.choice(TEMPLATES_MULTIPLE))
    q1_text = q1_text.replace("{EVENT_SET}", f"{{{_format_event_set(q1_labels)}}}")
    q2_text = rng.choice(TEMPLATES_SWITCH).replace("{EVENT_SET}", f"{{{_format_event_set(q2_labels)}}}")

    q_list = [
        {"time": 0.0, "count": 0, "text": q1_text},
        {"time": round(max(0.0, q2_time - seg_start), 6), "count": 1, "text": q2_text},
    ]

    # Answers: Q1 until q2_time, Q2 afterwards (both with min_gap)
    answers: List[Dict[str, object]] = []
    for e in evs:
        rel = e.pos_s - seg_start
        if rel >= (q2_time - seg_start + min_gap_seconds):
            if e.label in set(q2_labels):
                answers.append({"start": round(rel, 6), "end": round(rel, 6), "count": 1, "text": f"Alert - {e.label}"})
        elif rel >= (0.0 + min_gap_seconds):
            if e.label in set(q1_labels):
                answers.append({"start": round(rel, 6), "end": round(rel, 6), "count": 0, "text": f"Alert - {e.label}"})

    if not answers:
        return None

    # Public payload
    seg_id_str = f"{src_file}#{seg_path.name}"
    pub = {
        "source": "soccer",
        "id": 0,
        "video_id": slug_to_uuid(seg_id_str),
        "data_type": "online",
        "train_stage": 2,
        "length": round(max(0.0, seg_end - seg_start), 6),
        "question_category": "switch_trace",
        "question": q_list,
        "answer": answers,
    }
    pub["_raw_source_file"] = src_file or None
    return pub


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate switch-trace chain QA from dense segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base-dir", type=str, required=True, help="Root of dense segment JSONs")
    ap.add_argument("--per-file-root", type=str, default="out/switch_trace", help="Mirror root for per-file outputs")
    ap.add_argument("--per-file-subdir", type=str, default="switch_trace", help="Subdirectory name inside each match dir")
    ap.add_argument("--include-not-shown", action="store_true", help="Include visibility == 'not shown' events")
    ap.add_argument("--random-seed", type=int, default=1234, help="Random seed for template and label choices")
    ap.add_argument("--min-gap-seconds", type=float, default=3.0, help="Minimal gap between question and first answer")
    ap.add_argument("--per-segment-max", type=int, default=0, help="Max QA items per segment (<=0 for unlimited)")
    ap.add_argument("--max-files", type=int, default=0, help="Cap on total outputs (<=0 for unlimited)")

    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Base dir not found: {base}")
        return 2

    rng = random.Random(args.random_seed)
    out_root = Path(args.per_file_root)
    written = 0
    for seg_path in iter_segment_files(base):
        per_seg = 0
        seen_sigs = set()
        attempts = 0
        while True:
            attempts += 1
            item = build_switch_for_segment(seg_path, rng, include_not_shown=bool(args.include_not_shown), min_gap_seconds=float(args.min_gap_seconds))
            if not item:
                if attempts >= 6:
                    break
                else:
                    continue
            raw_src = item.get("_raw_source_file")
            if not raw_src:
                break
            # de-dup signature: (q1 text, q2 text, q2 time)
            try:
                q_arr = item.get("question", [])
                q1t = q_arr[0].get("text", "") if len(q_arr) > 0 else ""
                q2t = q_arr[1].get("text", "") if len(q_arr) > 1 else ""
                q2time = float(q_arr[1].get("time", 0.0)) if len(q_arr) > 1 else 0.0
                sig = (q1t, q2t, round(q2time, 3))
            except Exception:
                sig = None
            if sig is not None and sig in seen_sigs:
                if attempts >= 12:
                    break
                else:
                    continue
            if sig is not None:
                seen_sigs.add(sig)

            # Compute mirror path
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

            # filename
            seg_meta = {} if not item else {}
            s_sec = 0.0
            try:
                # find original segment from base dir
                with seg_path.open("r", encoding="utf-8") as f:
                    seg_src = json.load(f)
                segm = seg_src.get("segment", {})
                s_sec = float(segm.get("start_seconds", 0.0))
            except Exception:
                pass
            q2_ins = float(q2time)
            fname = f"switch_{int(round(s_sec*1000)):07d}_{int(round(q2_ins*1000)):07d}_{per_seg:02d}.json"

            # assign id and drop internal fields
            item["id"] = written + 1
            item.pop("_raw_source_file", None)
            (target_dir / fname).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
            per_seg += 1
            if args.max_files > 0 and written >= args.max_files:
                break
            if args.per_segment_max > 0 and per_seg >= args.per_segment_max:
                break
        if args.max_files > 0 and written >= args.max_files:
            break

    print(f"Wrote {written} switch_trace files under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
