#!/usr/bin/env python3
"""
Generate parallel-question chain QA from dense segment JSONs.

Category: parallel_question

Behavior per segment (unit=segment):
- Insert Q1 at the segment start time.
- Insert Q2 at a random later time inside the same segment.
- Q types (random per Q):
  1) Alert-only: "When {EVENT_SET} occurs, alert me."
  2) Which-team: "When {EVENT_SET} occurs, tell me which team."
- Q1 and Q2 track different event labels. Answers (A1/A2) are emitted whenever
  their respective events occur after each question's insertion time; answers are
  interleaved according to event times.

Outputs (one JSON per segment under a mirrored directory tree):
- source: "soccernet-dense-parallel"
- question_category: "parallel_question"
- question: a multi-line string combining Q1 and Q2
- questions: array of per-question metadata (qid, type, labels, insert_time)
- answer: interleaved responses with start/end (global seconds) and text noting Qid
- raw_source_file, segment_path, segment_meta, match_path, video_id

Usage example
  python scripts/generate_parallel_question_chains.py \
    --base-dir train/dense_segments \
    --per-file-root out/parallel_question \
    --per-file-subdir parallel_question \
    --max-files 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


ALERT_TEMPLATES_EN = [
    "Please track {EVENT_SET} and alert me whenever it occurs. Reply exactly: 'Alert - <Event>'.",
    "Monitor {EVENT_SET} and notify me each time it happens. Use: 'Alert - <Event>'.",
    "Watch for {EVENT_SET} and alert me upon occurrence. Answer format: 'Alert - <Event>'.",
    "Keep an eye on {EVENT_SET}; ping me every time it happens. Respond 'Alert - <Event>'.",
    "Observe {EVENT_SET} and let me know when it occurs. Respond exactly 'Alert - <Event>'.",
    "Track {EVENT_SET} and alert me whenever it happens. Answer: 'Alert - <Event>'.",
    "Continuously monitor {EVENT_SET} and notify me on occurrence. Reply with 'Alert - <Event>'.",
    "Please watch for {EVENT_SET} and alert me whenever it appears. Use 'Alert - <Event>'.",
    "Keep watching for {EVENT_SET} and let me know each time. Strictly 'Alert - <Event>'.",
    "Observe the stream and alert me for every {EVENT_SET}. Exact text: 'Alert - <Event>'.",
]


TEAM_TEMPLATES_EN = [
    "Track {EVENT_SET}; when it occurs, tell me which team (left/right only). Reply: 'Team left/right - <Event>'.",
    "Monitor {EVENT_SET} and report the team whenever it happens (left/right only). Answer: 'Team left/right - <Event>'.",
    "Watch for {EVENT_SET}; identify the team (left/right only) each time. Respond 'Team left/right - <Event>'.",
    "Follow {EVENT_SET} and respond with the team (left/right only) on occurrence: 'Team left/right - <Event>'.",
    "Observe {EVENT_SET} and specify the team when it happens (left/right only). Use 'Team left/right - <Event>'.",
    "Keep monitoring {EVENT_SET}; when it appears, tell me the team (left/right only). Reply 'Team left/right - <Event>'.",
    "Track {EVENT_SET} and report which team triggers it (left/right only). Answer 'Team left/right - <Event>'.",
    "Watch {EVENT_SET} and say which team (left/right only) when it happens. Use 'Team left/right - <Event>'.",
    "Observe {EVENT_SET}; on occurrence, identify the team (left/right only). Respond 'Team left/right - <Event>'.",
    "Monitor {EVENT_SET}; each time it occurs, tell me the team (left/right only). Exact: 'Team left/right - <Event>'.",
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


def build_questions(
    rng: random.Random,
    q1_labels: Sequence[str],
    q2_labels: Sequence[str],
    q1_type: str,
    q2_type: str,
) -> Tuple[str, List[Dict[str, object]]]:
    def fill(tmpls: Sequence[str], labels: Sequence[str]) -> str:
        evset = _format_event_set(labels)
        tmpl = rng.choice(list(tmpls))
        return tmpl.replace("{EVENT_SET}", f"{{{evset}}}")

    if q1_type == "alert":
        q1_text = fill(ALERT_TEMPLATES_EN, q1_labels)
    else:
        q1_text = fill(TEAM_TEMPLATES_EN, q1_labels)
    if q2_type == "alert":
        q2_text = fill(ALERT_TEMPLATES_EN, q2_labels)
    else:
        q2_text = fill(TEAM_TEMPLATES_EN, q2_labels)

    question_combined = f"Q1: {q1_text}\nQ2: {q2_text}"
    questions_meta = [
        {"qid": "Q1", "type": q1_type, "labels": list(q1_labels)},
        {"qid": "Q2", "type": q2_type, "labels": list(q2_labels)},
    ]
    return question_combined, questions_meta


def choose_insert_time(rng: random.Random, seg_start: float, seg_end: float, event_times: List[float]) -> float:
    # Prefer an insertion near an actual event to ensure quick answers
    if not event_times:
        return max(seg_start, seg_start + 0.1)
    # pick from the middle 60% of events to avoid too early/late
    n = len(event_times)
    lo = n // 5
    hi = max(lo + 1, int(n * 0.8))
    idx = rng.randrange(lo, hi)
    t = event_times[idx]
    # clamp into segment
    return float(max(seg_start, min(t, seg_end - 1e-3)))


def build_parallel_for_segment(
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
    match_id = str(data.get("UrlLocal", ""))

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
        # keep only events that truly reside in the segment window (inclusive start, exclusive end)
        if not (seg_start <= pos_s < seg_end):
            continue
        evs.append(Ann(pos_s=pos_s, label=lab, team=team, game_time=gt))

    if not evs:
        return None
    evs.sort(key=lambda e: (e.pos_s, e.label.lower()))

    # Build label pools
    labels_any = _uniq_labels_in_order(evs)
    labels_teamful = _uniq_labels_in_order([e for e in evs if e.team in TEAMFUL])
    if not labels_any:
        return None

    # Q1 at segment start
    q1_time = seg_start
    # Q2 randomly inserted later
    event_times = [e.pos_s for e in evs]
    q2_time = choose_insert_time(rng, seg_start, seg_end, event_times)

    # Q types
    q1_type = rng.choice(["alert", "team"]) if labels_teamful else "alert"
    q2_type = rng.choice(["alert", "team"]) if labels_teamful else "alert"

    # Pick label sets
    def pick_labels(pool: Sequence[str]) -> List[str]:
        k = 1 if len(pool) == 1 else rng.choice([1, 2])
        return rng.sample(list(pool), k=k)

    q1_pool = labels_teamful if q1_type == "team" else labels_any
    q1_labels = pick_labels(q1_pool)

    # Q2 labels must differ and also have occurrences AFTER q2_time
    def labels_after(t: float, team_required: bool) -> List[str]:
        filt = [e for e in evs if e.pos_s >= t and (e.team in TEAMFUL if team_required else True)]
        return _uniq_labels_in_order(filt)

    q2_pool_dyn = labels_after(q2_time, q2_type == "team")
    # ensure disjoint from q1 labels to match requirement
    q2_pool_dyn = [l for l in q2_pool_dyn if l not in set(q1_labels)]
    if not q2_pool_dyn:
        return None
    q2_labels = pick_labels(q2_pool_dyn)

    # Build question strings and meta
    question_text, questions_meta = build_questions(rng, q1_labels, q2_labels, q1_type, q2_type)
    # Attach insert times and convert to segment-relative times
    q1_time_rel = max(0.0, q1_time - seg_start)
    q2_time_rel = max(0.0, q2_time - seg_start)
    questions_meta[0]["insert_time"] = round(q1_time_rel, 6)
    questions_meta[1]["insert_time"] = round(q2_time_rel, 6)

    # Build interleaved answers
    answers: List[Dict[str, object]] = []

    def add_answer(qid: str, e: Ann, kind: str) -> None:
        if kind == "alert":
            text = f"Alert - {e.label}"
        else:
            # which-team
            if e.team in TEAMFUL:
                text = f"Team {e.team} - {e.label}"
            else:
                # skip if team not available
                return
        answers.append({
            "start": round(max(0.0, e.pos_s - seg_start), 6),
            "end": round(max(0.0, e.pos_s - seg_start), 6),
            "text": text,
            "_qid": qid,
        })

    # Iterate events in order and emit for each Q that has started
    for e in evs:
        if e.pos_s >= (q2_time + min_gap_seconds) and e.label in q2_labels:
            add_answer("Q2", e, q2_type)
        if e.pos_s >= (q1_time + min_gap_seconds) and e.label in q1_labels:
            add_answer("Q1", e, q1_type)

    if not answers:
        return None

    # Compose public payload following required schema
    seg_id_str = f"{src_file}#{seg_path.name}"
    q_arr = [
        {"time": questions_meta[0]["insert_time"], "count": 0, "text": question_text.split("\n")[0][4:]},
        {"time": questions_meta[1]["insert_time"], "count": 1, "text": question_text.split("\n")[1][4:]},
    ]
    # Add counts to answers based on qid
    a_arr = []
    for a in sorted(answers, key=lambda x: (x["start"], x["text"])):
        cnt = 0 if a.get("_qid") == "Q1" else 1
        a_arr.append({"start": a["start"], "end": a["end"], "count": cnt, "text": a["text"]})

    pub = {
        "source": "soccer",
        "id": 0,
        "video_id": slug_to_uuid(seg_id_str),
        "data_type": "online",
        "train_stage": 2,
        "length": round(max(0.0, seg_end - seg_start), 6),
        "question_category": "parallel_question",
        "question": q_arr,
        "answer": a_arr,
    }
    # Attach raw for pathing
    pub["_raw_source_file"] = src_file or None
    return pub


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate parallel-question chain QA from dense segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base-dir", type=str, required=True, help="Root of dense segment JSONs")
    ap.add_argument("--per-file-root", type=str, default="out/parallel_question", help="Mirror root for per-file outputs")
    ap.add_argument("--per-file-subdir", type=str, default="parallel_question", help="Subdirectory name inside each match dir")
    ap.add_argument("--include-not-shown", action="store_true", help="Include visibility == 'not shown' events")
    ap.add_argument("--random-seed", type=int, default=123, help="Random seed for template and placement")
    ap.add_argument("--min-gap-seconds", type=float, default=3.0, help="Minimal time gap between a question insertion and its first answer")
    ap.add_argument("--max-files", type=int, default=0, help="Cap number of outputs (<=0 for unlimited)")
    ap.add_argument("--per-segment-max", type=int, default=0, help="Max number of QA items per segment (<=0 for unlimited)")

    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Base dir not found: {base}", file=sys.stderr)
        return 2

    rng = random.Random(args.random_seed)

    out_root = Path(args.per_file_root)
    written = 0
    for seg_path in iter_segment_files(base):
        per_seg_written = 0
        seen_sigs = set()
        attempts = 0
        while True:
            attempts += 1
            item = build_parallel_for_segment(
                seg_path,
                rng=rng,
                include_not_shown=bool(args.include_not_shown),
                min_gap_seconds=float(args.min_gap_seconds),
            )
            if not item:
                # Give up after a few failed attempts to build for this segment
                if attempts >= 5:
                    break
                else:
                    continue
            raw_src = item.get("_raw_source_file")
            if not raw_src:
                break
            # Dedup signature: (Q1 text, Q2 text, Q2 time)
            try:
                q_arr = item.get("question", [])
                q1_text = q_arr[0].get("text", "") if len(q_arr) > 0 else ""
                q2_text = q_arr[1].get("text", "") if len(q_arr) > 1 else ""
                q2_time = float(q_arr[1].get("time", 0.0)) if len(q_arr) > 1 else 0.0
                sig = (q1_text, q2_text, round(q2_time, 3))
            except Exception:
                sig = None
            if sig is not None and sig in seen_sigs:
                if attempts >= 12:
                    break
                else:
                    continue
            if sig is not None:
                seen_sigs.add(sig)

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

            # Build filename from segment start and Q2 time, plus index to avoid clashes
            try:
                seg_meta = item.get("segment_meta", {})
                s_sec = float(seg_meta.get("start_seconds", 0.0))
            except Exception:
                s_sec = 0.0
            q2_ins = float(q2_time) if 'q2_time' in locals() else 0.0
            fname = f"parallel_{int(round(s_sec*1000)):07d}_{int(round(q2_ins*1000)):07d}_{per_seg_written:02d}.json"
            # Assign a simple sequential id
            item["id"] = written + 1
            # Remove internal field before writing
            item.pop("_raw_source_file", None)
            (target_dir / fname).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
            per_seg_written += 1
            if args.max_files > 0 and written >= args.max_files:
                break
            if args.per_segment_max > 0 and per_seg_written >= args.per_segment_max:
                break
        if args.max_files > 0 and written >= args.max_files:
            break

    print(f"Wrote {written} parallel_question files under: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
