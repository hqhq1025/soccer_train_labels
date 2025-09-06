#!/usr/bin/env python3
"""
Build QA samples from pre-cut soccer event segments into the requested schema.

Output schema (array of objects):
[
  {
    "source": "soccernet-dense",
    "id": 1,
    "video_id": "<stable-uuid-from-match>",
    "data_type": "online",
    "train_stage": 2,
    "length": 59.47,
    "question_category": "Status Confirmation & Instruction Following",
    "question": "Alert me every time {TARGET_EVENT} occurs.",
    "answer": [
      {"start": 276.05, "end": 276.05, "text": "Alert: {TARGET_EVENT} occurred."}
    ]
  }
]

Supports two tasks:
- multi: monitor a set of labels and report whenever one occurs
- single: monitor a single label and report its occurrences

Time reference:
- By default we use "global" timestamps (seconds from match start) from the
  original annotations' "position" field. Optionally choose "segment" to
  express time as seconds from the start of the exported segment.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def seconds_to_mmss(x: float) -> str:
    if x < 0:
        x = 0.0
    m = int(x // 60)
    s = int(round(x - 60 * m))
    return f"{m:02d}:{s:02d}"


def slug_to_uuid(s: str) -> str:
    # Stable UUID5 from path-like string
    ns = uuid.NAMESPACE_URL
    return str(uuid.uuid5(ns, s))


def iter_segment_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("segment_*.json"):
        if p.is_file():
            yield p


def normalize_label(x: Optional[str]) -> str:
    return (x or "").strip()


def label_key(x: Optional[str]) -> str:
    return (x or "").strip().lower()


@dataclass
class BuildConfig:
    task: str  # 'single' | 'multi'
    label: Optional[str]
    event_set: Optional[List[str]]
    segments_dir: Path
    output: Path
    data_type: str
    train_stage: int
    question_category: str
    time_mode: str  # 'global' | 'segment'
    max_samples: Optional[int]
    none_token: str
    source_tag: str
    use_all_labels_for_multi: bool
    q_templates_single: List[str]
    q_templates_multi: List[str]
    rng: random.Random


def _fill_single_template(tmpl: str, label: str) -> str:
    # Replace placeholder with curly-braced label
    return tmpl.replace("{TARGET_LABEL}", f"{{{label}}}")


def _fill_multi_template(tmpl: str, labels: Sequence[str]) -> str:
    listed = ", ".join(labels)
    return tmpl.replace("{EVENT_SET}", f"{{{listed}}}")


def build_question(cfg: BuildConfig) -> str:
    if cfg.task == "single":
        assert cfg.label, "--label is required for single task"
        tmpl = cfg.rng.choice(cfg.q_templates_single)
        return _fill_single_template(tmpl, cfg.label)
    else:
        assert cfg.event_set, "--event-set is required for multi task"
        tmpl = cfg.rng.choice(cfg.q_templates_multi)
        return _fill_multi_template(tmpl, cfg.event_set)


def make_answer_text(label: str, team: str, single: bool) -> str:
    # Answers should directly use the label per request; keep templates minimal.
    if single:
        return f"Alert: {label}."
    else:
        return f"{label}"


def event_passes_filters(event: Dict[str, object], allowed: Optional[set]) -> bool:
    lab = label_key(str(event.get("label", "")))
    if allowed is None:
        return True
    return lab in allowed


def is_not_shown(event: Dict[str, object]) -> bool:
    vis = (str(event.get("visibility", "")).lower())
    return vis == "not shown"


def build_item_from_segment(cfg: BuildConfig, seg_path: Path) -> Optional[Dict[str, object]]:
    with seg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    seg_meta = obj.get("segment", {})
    anns = obj.get("annotations", [])

    # Basic metadata
    source_file = obj.get("source_file", "")
    match_id_str = source_file or str(seg_path)
    video_id = slug_to_uuid(match_id_str)

    seg_start = float(seg_meta.get("start_seconds", 0.0))
    seg_end = float(seg_meta.get("end_seconds", 0.0))
    seg_len = float(seg_meta.get("length_seconds", max(0.0, seg_end - seg_start)))

    # Filters
    allow: Optional[set] = None
    single = cfg.task == "single"
    if single:
        assert cfg.label is not None
        allow = {label_key(cfg.label)}
    elif cfg.event_set:
        allow = {label_key(x) for x in cfg.event_set}

    # Build answer list
    answers: List[Dict[str, object]] = []
    for a in anns:
        if is_not_shown(a):
            continue
        if not event_passes_filters(a, allow):
            continue
        try:
            pos_s = int(a.get("position")) / 1000.0
        except Exception:
            continue
        if cfg.time_mode == "segment":
            t = max(0.0, pos_s - seg_start)
        else:
            t = pos_s
        team = str(a.get("team", "not applicable")) or "not applicable"
        label = normalize_label(str(a.get("label", "")))
        text = make_answer_text(label if not single else cfg.label or label, team, single)
        answers.append({
            "start": round(t, 6),
            "end": round(t, 6),
            "text": text,
        })

    # Ensure chronological order
    answers.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))

    # Compose sample
    # Category mapping per request: single -> Instruction Following, multi -> Status Confirmation
    question_category = "Instruction Following" if cfg.task == "single" else "Status Confirmation"

    sample: Dict[str, object] = {
        "source": cfg.source_tag,
        "id": 0,  # filled later
        "video_id": video_id,
        "data_type": cfg.data_type,
        "train_stage": cfg.train_stage,
        "length": round(seg_len, 6),
        "question_category": question_category,
        "question": build_question(cfg),
        "answer": answers,
        # add original segment json path for traceability
        "segment_path": str(seg_path),
    }

    return sample


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Compose QA samples from dense segment JSONs into the requested schema",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--task", choices=["single", "multi"], required=True, help="Type of QA task to build")
    p.add_argument("--label", type=str, default=None, help="Single-event label (for task=single)")
    p.add_argument("--event-set", nargs="*", default=None, help="Event labels to monitor (for task=multi)")
    p.add_argument("--segments-dir", type=str, required=True, help="Root directory of segment_*.json files")
    p.add_argument("--output", type=str, required=True, help="Output JSON file path (array)")
    p.add_argument("--data-type", type=str, default="online", help="online/offline")
    p.add_argument("--train-stage", type=int, default=2)
    p.add_argument("--question-category", type=str, default="Status Confirmation & Instruction Following")
    p.add_argument("--time-mode", choices=["global", "segment"], default="global", help="Timestamp reference for answers")
    p.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of samples")
    p.add_argument("--none-token", type=str, default="none", help="Text for no-event case if needed")
    p.add_argument("--source-tag", type=str, default="soccernet-dense", help="Value for 'source' field")
    p.add_argument("--multi-use-all-labels", action="store_true", help="For task=multi, auto-fill EVENT_SET with all labels found under --segments-dir if --event-set is not provided")
    p.add_argument("--question-templates", type=str, default=None, help="Optional JSON file with keys 'single' and 'multi' providing question templates")
    p.add_argument("--random-seed", type=int, default=42, help="Seed for question template selection")

    args = p.parse_args(argv)

    # Load question templates
    default_single = [
        "Please continuously watch the match video. When {TARGET_LABEL} occurs, alert me.",
        "Keep watching the live broadcast and notify me when {TARGET_LABEL} happens.",
        "Monitor the game stream and alert me upon any {TARGET_LABEL}.",
        "Observe the match footage and tell me whenever {TARGET_LABEL} occurs.",
        "Watch the video and report when {TARGET_LABEL} happens."
    ]
    default_multi = [
        "Please continuously watch the match video and monitor {EVENT_SET}. Whenever a key event occurs, tell me which key event happened?",
        "Keep watching the live broadcast and monitor {EVENT_SET}. Each time one occurs, state which event happened.",
        "Observe the game stream and track {EVENT_SET}. Report which event occurs whenever it happens.",
        "Monitor {EVENT_SET} in the match footage. Whenever an event occurs, tell me which event it is.",
        "Watch the video and monitor {EVENT_SET}. Every time one occurs, say which event happened."
    ]

    if args.question_templates:
        try:
            tmpl_obj = json.loads(Path(args.question_templates).read_text(encoding="utf-8"))
            single_list = tmpl_obj.get("single") or []
            multi_list = tmpl_obj.get("multi") or []
            if not isinstance(single_list, list) or not single_list:
                single_list = default_single
            if not isinstance(multi_list, list) or not multi_list:
                multi_list = default_multi
        except Exception:
            single_list = default_single
            multi_list = default_multi
    else:
        single_list = default_single
        multi_list = default_multi

    rng = random.Random(args.random_seed)

    cfg = BuildConfig(
        task=args.task,
        label=args.label,
        event_set=args.event_set,
        segments_dir=Path(args.segments_dir),
        output=Path(args.output),
        data_type=args.data_type,
        train_stage=args.train_stage,
        question_category=args.question_category,
        time_mode=args.time_mode,
        max_samples=args.max_samples,
        none_token=args.none_token,
        source_tag=args.source_tag,
        use_all_labels_for_multi=bool(args.multi_use_all_labels),
        q_templates_single=single_list,
        q_templates_multi=multi_list,
        rng=rng,
    )

    if cfg.task == "single" and not cfg.label:
        print("--label is required for task=single", file=sys.stderr)
        return 2
    if cfg.task == "multi" and (not cfg.event_set or len(cfg.event_set) == 0):
        if cfg.use_all_labels_for_multi:
            # Discover all labels under segments_dir by scanning annotations
            all_labels_set = set()
            for seg_path in iter_segment_files(cfg.segments_dir):
                try:
                    data = json.loads(Path(seg_path).read_text(encoding="utf-8"))
                except Exception:
                    continue
                for a in data.get("annotations", []):
                    vis = str(a.get("visibility", "")).lower()
                    if vis == "not shown":
                        continue
                    lab = a.get("label")
                    if lab:
                        all_labels_set.add(normalize_label(str(lab)))
            cfg.event_set = sorted(all_labels_set, key=lambda x: x.lower())
            if not cfg.event_set:
                print("No labels found under --segments-dir for multi task", file=sys.stderr)
                return 2
        else:
            print("--event-set is required for task=multi (or use --multi-use-all-labels)", file=sys.stderr)
            return 2

    out_list: List[Dict[str, object]] = []
    count = 0
    for seg_path in iter_segment_files(cfg.segments_dir):
        item = build_item_from_segment(cfg, seg_path)
        if item is None:
            continue
        count += 1
        item["id"] = count
        out_list.append(item)
        if cfg.max_samples is not None and len(out_list) >= cfg.max_samples:
            break

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(out_list, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_list)} samples to: {cfg.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
