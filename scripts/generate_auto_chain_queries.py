#!/usr/bin/env python3
"""
从 dense_segments 自动构造“链式 online 问题”样本（无需手工指定事件）。

链式模板（中文）：
  A) 触发+跟踪(任意事件)：
     - 问题：请持续观看视频。当下一个{TRIGGER}发生时提醒我，并告诉我是 left 还是 right；
             之后继续跟踪这支球队，依次提醒该队后续的事件（最多K个）。
     - 答案：给出触发事件与后续事件发生的时间点与文字描述（left/right + label）。

  B) 触发+指定后续标签：
     - 问题：请持续观看视频。当下一个{TRIGGER}发生时提醒我，并告诉我是 left 还是 right；
             然后如果这支球队发生{FOLLOW}事件时提醒我（最多1次）。
     - 答案：给出触发事件时间与首个符合 FOLLOW 的事件时间。

注意
- 仅使用可判定球队的事件：team ∈ {left, right}
- 默认忽略 visibility == 'not shown'（可用 --include-not-shown 开启）。
- 事件时间以“global seconds”（来自 annotation.position 毫秒）为准。
- 为避免重复，按 (position_ms, label, team) 去重后排序。

输出
- 默认写出一个 JSON 数组，每项为一个链式样本：
  {
    "source": "soccernet-dense-chain",
    "id": 123,
    "video_id": "<uuid-from-match-path>",
    "data_type": "online",
    "question": "...含 {TRIGGER}/{FOLLOW} ...",
    "answer": [ {"start": 123.45, "end": 123.45, "text": "..."}, ... ],
    "match_path": "england_epl/.../",
    "chain_type": "A|B",
  }
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


def slug_to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


@dataclass
class Ann:
    pos_ms: int
    pos_s: float
    label: str
    team: str
    game_time: str
    half: int
    sec_in_half: int


def iter_segment_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("segment_*.json"):
        if p.is_file():
            yield p


def collect_match_events(
    base_dir: Path,
    include_not_shown: bool,
    only_match_substr: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """聚合所有比赛的（去重后的）事件序列。
    返回 { match_id(UrlLocal): { 'video_id', 'events': List[Ann], 'match_path': str } }
    """
    matches: Dict[str, Dict[str, object]] = {}
    for fp in iter_segment_files(base_dir):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        match_id = str(obj.get("UrlLocal") or "").strip()
        if not match_id:
            # fallback: 相对路径
            match_id = str(fp.parent.relative_to(base_dir)) + "/"
        if only_match_substr and only_match_substr not in match_id:
            continue
        m = matches.setdefault(
            match_id,
            {
                "video_id": slug_to_uuid(match_id),
                "events": [],
                "seen": set(),  # for dedup
                "match_path": match_id,
                "raw_source_file": None,
            },
        )

        anns = obj.get("annotations", []) or []
        # Source raw Labels file path (relative to repo root)
        src_file = obj.get("source_file")
        if src_file and not m["raw_source_file"]:
            m["raw_source_file"] = str(src_file)
        for a in anns:
            team = str(a.get("team", "")).lower().strip()
            if team not in TEAMFUL:
                continue
            vis = str(a.get("visibility", "")).lower().strip()
            if (not include_not_shown) and vis == "not shown":
                continue
            label = str(a.get("label", "")).strip()
            try:
                pos_ms = int(a.get("position"))
            except Exception:
                continue
            key = (pos_ms, label.lower(), team)
            if key in m["seen"]:
                continue
            m["seen"].add(key)
            gt = str(a.get("gameTime", ""))
            # Parse half and mm:ss
            try:
                half_str, mmss = [t.strip() for t in gt.split("-")]
                half = int(half_str)
                mm, ss = [int(x) for x in mmss.split(":")]
                sec_in_half = mm * 60 + ss
            except Exception:
                half = 9
                sec_in_half = 10**9
            m["events"].append(
                Ann(
                    pos_ms=pos_ms,
                    pos_s=pos_ms / 1000.0,
                    label=label,
                    team=team,
                    game_time=gt,
                    half=half,
                    sec_in_half=sec_in_half,
                )
            )

    # 排序每个比赛的事件
    for match_id, rec in matches.items():
        rec["events"].sort(key=lambda e: (e.half, e.sec_in_half, e.pos_ms, e.label.lower()))
        # 清理 seen
        rec.pop("seen", None)
    return matches


def collect_segment_records(
    base_dir: Path,
    include_not_shown: bool,
    only_match_substr: Optional[str] = None,
) -> List[Dict[str, object]]:
    """逐段（per-segment）收集事件，不跨段聚合。
    返回列表，每项：{
        'match_path', 'raw_source_file', 'segment_path', 'video_id', 'segment', 'events': List[Ann]
    }
    """
    records: List[Dict[str, object]] = []
    for fp in iter_segment_files(base_dir):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        match_id = str(obj.get("UrlLocal") or "").strip()
        if not match_id:
            match_id = str(fp.parent.relative_to(base_dir)) + "/"
        if only_match_substr and only_match_substr not in match_id:
            continue
        src_file = str(obj.get("source_file") or "")
        seg_meta = obj.get("segment", {})

        anns_raw = obj.get("annotations", []) or []
        events: List[Ann] = []
        for a in anns_raw:
            team = str(a.get("team", "")).lower().strip()
            if team not in TEAMFUL:
                continue
            vis = str(a.get("visibility", "")).lower().strip()
            if (not include_not_shown) and vis == "not shown":
                continue
            label = str(a.get("label", "")).strip()
            # Prefer half-aware global seconds; fallback to position
            raw_pos_ms = None
            try:
                raw_pos_ms = int(a.get("position"))
            except Exception:
                raw_pos_ms = None
            gt = str(a.get("gameTime", "")).strip()
            try:
                half_str, mmss = [t.strip() for t in gt.split("-")]
                half = int(half_str)
                mm, ss = [int(x) for x in mmss.split(":")]
                sec_in_half = mm * 60 + ss
            except Exception:
                half = 9
                sec_in_half = 10**9
            pos_s_global = parse_game_time_seconds(gt, fallback_ms=raw_pos_ms)
            if pos_s_global is None:
                continue
            pos_ms = int(round(pos_s_global * 1000.0))
            events.append(
                Ann(
                    pos_ms=pos_ms,
                    pos_s=pos_s_global,
                    label=label,
                    team=team,
                    game_time=gt,
                    half=half,
                    sec_in_half=sec_in_half,
                )
            )
        if not events:
            continue
        events.sort(key=lambda e: (e.half, e.sec_in_half, e.pos_ms, e.label.lower()))

        # Ensure stable per-segment video_id
        seg_id_str = f"{src_file}#{Path(fp).name}"
        video_id = slug_to_uuid(seg_id_str)
        records.append(
            {
                "match_path": match_id,
                "raw_source_file": src_file or None,
                "segment_path": str(fp),
                "segment": seg_meta,
                "video_id": video_id,
                "events": events,
            }
        )
    return records


def _uniq_labels_in_order(events: Sequence[Ann]) -> List[str]:
    seen = set()
    out: List[str] = []
    for e in events:
        key = e.label.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _format_event_set(labels: Sequence[str]) -> str:
    # Join with commas and 'and' for readability
    labs = [l for l in labels if l]
    if not labs:
        return ""
    if len(labs) == 1:
        return labs[0]
    return ", ".join(labs[:-1]) + " and " + labs[-1]


FOLLOW_TEMPLATES_EN = [
    "After the next {TRIGGER}, follow that team. Answer format: at trigger reply 'Trigger: {TRIGGER} by left/right'; for each {EVENT_SET}, reply 'Then: <Event> by left/right'. Use left/right only.",
    "When the next {TRIGGER} happens, lock onto that team. Respond 'Trigger: {TRIGGER} by left/right' and thereafter 'Then: <Event> by left/right' for {EVENT_SET}. Left/right only.",
    "At the next {TRIGGER}, start tracking that team. Use responses 'Trigger: {TRIGGER} by left/right' and 'Then: <Event> by left/right' for {EVENT_SET}.", 
    "Once the next {TRIGGER} occurs, keep following the same team and reply exactly 'Trigger: {TRIGGER} by left/right'; then for each {EVENT_SET} reply 'Then: <Event> by left/right'.", 
    "When the next {TRIGGER} occurs, follow the initiating team; answer strictly as 'Trigger: {TRIGGER} by left/right' and 'Then: <Event> by left/right' for {EVENT_SET}.", 
    "On the next {TRIGGER}, stick with that team. For {EVENT_SET}, reply 'Then: <Event> by left/right' (left/right only).", 
    "After the next {TRIGGER}, continue tracking that team. Answers must be 'Trigger: {TRIGGER} by left/right' and 'Then: <Event> by left/right' for {EVENT_SET}.", 
    "Watch for the next {TRIGGER}; then follow that team. Respond 'Trigger: {TRIGGER} by left/right', and for {EVENT_SET} reply 'Then: <Event> by left/right'.", 
    "Upon the next {TRIGGER}, follow the same team and list all subsequent {EVENT_SET}. Use 'Then: <Event> by left/right' only (left/right only).", 
    "At the next {TRIGGER}, keep following that team. Use left/right only; 'Trigger: {TRIGGER} by left/right', then 'Then: <Event> by left/right' for {EVENT_SET}.",
]


def _build_question_follow_team(rng: random.Random, trigger_label: str, event_set_labels: Sequence[str]) -> str:
    tmpl = rng.choice(FOLLOW_TEMPLATES_EN)
    evset = _format_event_set(event_set_labels)
    return tmpl.replace("{TRIGGER}", f"{{{trigger_label}}}").replace("{EVENT_SET}", f"{{{evset}}}")


def make_chain_A(
    trigger: Ann,
    follows: List[Ann],
    video_id: str,
    match_path: str,
    raw_source_file: Optional[str],
    k: int,
    segment_path: Optional[str] = None,
    segment_meta: Optional[Dict[str, object]] = None,
    question_category: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> Dict[str, object]:
    # Build follow label set from subsequent same-team events
    follow_labels = _uniq_labels_in_order(follows)
    if rng is None:
        rng = random.Random()
    q = _build_question_follow_team(rng, trigger.label, follow_labels)

    ans = [
        {
            "start": round(trigger.pos_s, 6),
            "end": round(trigger.pos_s, 6),
            "text": f"Trigger: {trigger.label} by {trigger.team}",
        }
    ]
    count = 0
    for e in follows:
        ans.append(
            {
                "start": round(e.pos_s, 6),
                "end": round(e.pos_s, 6),
                "text": f"Then: {e.label} by {e.team}",
            }
        )
        count += 1
        # For follow_team category, do not cap the number of events
        if (question_category or "follow_team") != "follow_team" and k > 0 and count >= k:
            break

    item = {
        "source": "soccernet-dense-chain",
        "id": 0,
        "video_id": video_id,
        "data_type": "online",
        "question_category": question_category or "follow_team",
        "question": q,
        "answer": ans,
        "match_path": match_path,
        "raw_source_file": raw_source_file,
        "chain_type": "A",
        "trigger_ms": trigger.pos_ms,
        "team": trigger.team,
    }
    if segment_path:
        item["segment_path"] = segment_path
    if segment_meta is not None:
        item["segment_meta"] = segment_meta
    return item


def make_chain_B(
    trigger: Ann,
    follow_label: str,
    follow_event: Ann,
    video_id: str,
    match_path: str,
    raw_source_file: Optional[str],
    segment_path: Optional[str] = None,
    segment_meta: Optional[Dict[str, object]] = None,
    question_category: Optional[str] = None,
    rng: Optional[random.Random] = None,
    follows: Optional[List[Ann]] = None,
) -> Dict[str, object]:
    # For B, list unique labels from subsequent events (same team).
    labels = _uniq_labels_in_order(follows or [])
    if not labels:
        labels = [follow_label]
    if rng is None:
        rng = random.Random()
    q = _build_question_follow_team(rng, trigger.label, labels)
    ans = [
        {
            "start": round(trigger.pos_s, 6),
            "end": round(trigger.pos_s, 6),
            "text": f"Trigger: {trigger.label} by {trigger.team}",
        },
        {
            "start": round(follow_event.pos_s, 6),
            "end": round(follow_event.pos_s, 6),
            "text": f"Then: {follow_event.label} by {follow_event.team}",
        },
    ]
    item = {
        "source": "soccernet-dense-chain",
        "id": 0,
        "video_id": video_id,
        "data_type": "online",
        "question_category": question_category or "follow_team",
        "question": q,
        "answer": ans,
        "match_path": match_path,
        "raw_source_file": raw_source_file,
        "chain_type": "B",
        "trigger_ms": trigger.pos_ms,
        "team": trigger.team,
    }
    if segment_path:
        item["segment_path"] = segment_path
    if segment_meta is not None:
        item["segment_meta"] = segment_meta
    return item


def build_chains(
    matches: Dict[str, Dict[str, object]],
    max_chains: Optional[int],
    chain_types: Sequence[str],
    follow_k: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    # 顺序遍历，直到达上限
    for match_id, rec in matches.items():
        vid = str(rec.get("video_id"))
        events: List[Ann] = rec.get("events", [])
        if not events:
            continue
        n = len(events)
        for i, trig in enumerate(events):
            # 后续同队事件
            follows = [
                e
                for e in events[i + 1 :]
                if e.team == trig.team
                and (
                    (e.half > trig.half)
                    or (e.half == trig.half and e.sec_in_half > trig.sec_in_half)
                )
            ]
            if not follows:
                continue

            if "A" in chain_types:
                itemA = make_chain_A(trig, follows, vid, match_id, rec.get("raw_source_file"), k=follow_k, question_category=question_category, rng=rng)
                out.append(itemA)
                if max_chains is not None and len(out) >= max_chains:
                    return out

            if "B" in chain_types:
                # 选择一个后续事件的 label 作为 FOLLOW
                f0 = follows[0]
                itemB = make_chain_B(trig, f0.label, f0, vid, match_id, rec.get("raw_source_file"), question_category=question_category, rng=rng, follows=follows)
                out.append(itemB)
                if max_chains is not None and len(out) >= max_chains:
                    return out
    return out


def build_chains_per_segment(
    segments: List[Dict[str, object]],
    max_chains: Optional[int],
    chain_types: Sequence[str],
    follow_k: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for rec in segments:
        events: List[Ann] = rec.get("events", [])
        if not events:
            continue
        match_path = str(rec.get("match_path", ""))
        raw_src = rec.get("raw_source_file")
        seg_path = rec.get("segment_path")
        seg_meta = rec.get("segment")
        video_id = str(rec.get("video_id"))

        for i, trig in enumerate(events):
            follows = [
                e
                for e in events[i + 1 :]
                if e.team == trig.team
                and (
                    (e.half > trig.half)
                    or (e.half == trig.half and e.sec_in_half > trig.sec_in_half)
                )
            ]
            if not follows:
                continue
            if "A" in chain_types:
                out.append(
                    make_chain_A(
                        trig,
                        follows,
                        video_id,
                        match_path,
                        raw_src if isinstance(raw_src, str) else None,
                        k=follow_k,
                        segment_path=str(seg_path) if seg_path else None,
                        segment_meta=seg_meta if isinstance(seg_meta, dict) else None,
                        question_category=question_category,
                        rng=rng,
                    )
                )
                if max_chains is not None and len(out) >= max_chains:
                    return out
            if "B" in chain_types:
                f0 = follows[0]
                out.append(
                    make_chain_B(
                        trig,
                        f0.label,
                        f0,
                        video_id,
                        match_path,
                        raw_src if isinstance(raw_src, str) else None,
                        segment_path=str(seg_path) if seg_path else None,
                        segment_meta=seg_meta if isinstance(seg_meta, dict) else None,
                        question_category=question_category,
                        rng=rng,
                    )
                )
                if max_chains is not None and len(out) >= max_chains:
                    return out
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="从 dense_segments 自动生成链式 online 样本（无需手工指定事件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--base-dir", type=str, default="train/dense_segments", help="dense_segments 根目录")
    ap.add_argument(
        "--unit",
        choices=["segment", "match"],
        default="segment",
        help="链条构造粒度：segment=仅在每个分段内构造；match=跨段聚合（原逻辑）",
    )
    ap.add_argument("--output", type=str, required=False, help="输出 JSON 文件路径（数组）。若未提供且启用 --per-file，则只写单文件样本")
    ap.add_argument("--category", type=str, default="follow_team", help="question_category 字段（用于区分链式任务类别）")
    ap.add_argument("--random-seed", type=int, default=42, help="Random seed for question template selection")
    ap.add_argument(
        "--max-chains",
        type=int,
        default=500,
        help="最多生成多少条链式样本（A/B 总和）。<=0 表示不设上限",
    )
    ap.add_argument("--chain-types", nargs="*", choices=["A", "B"], default=["A", "B"], help="生成的链式类型：A(任意后续) / B(指定后续标签)")
    ap.add_argument("--follow-k", type=int, default=3, help="A 型链式中，最多提醒的后续事件数量")
    ap.add_argument("--include-not-shown", action="store_true", help="包含 visibility == 'not shown' 的事件")
    ap.add_argument("--only-match-substr", type=str, default=None, help="仅处理路径包含该子串的比赛（调试用）")
    ap.add_argument(
        "--per-file",
        action="store_true",
        help="将每条链式样本写为单独 JSON 文件，按 raw_jsons 的目录结构进行镜像组织",
    )
    ap.add_argument(
        "--per-file-root",
        type=str,
        default="out/raw_like_chains",
        help="镜像输出的根目录（新建），其下结构与 raw_jsons 内部一致（从 split 开始）",
    )
    ap.add_argument(
        "--per-file-subdir",
        type=str,
        default="qa_chains",
        help="在每个比赛目录下创建的子目录名；留空则直接写到比赛目录内",
    )

    args = ap.parse_args(argv)

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Base dir not found: {base}", file=sys.stderr)
        return 2

    # Expose category to downstream helpers
    global question_category
    question_category = args.category

    max_chains = None if args.max_chains <= 0 else args.max_chains
    rng = random.Random(args.random_seed)
    if args.unit == "segment":
        segs = collect_segment_records(
            base,
            include_not_shown=bool(args.include_not_shown),
            only_match_substr=args.only_match_substr,
        )
        if not segs:
            print("No segment events found under base-dir.", file=sys.stderr)
            return 1
        items = build_chains_per_segment(
            segs,
            max_chains=max_chains,
            chain_types=args.chain_types,
            follow_k=max(1, args.follow_k),
            rng=rng,
        )
    else:
        matches = collect_match_events(
            base,
            include_not_shown=bool(args.include_not_shown),
            only_match_substr=args.only_match_substr,
        )
        if not matches:
            print("No match events found under base-dir.", file=sys.stderr)
            return 1
        items = build_chains(
            matches,
            max_chains=max_chains,
            chain_types=args.chain_types,
            follow_k=max(1, args.follow_k),
            rng=rng,
        )
    # 赋 id
    for i, it in enumerate(items, start=1):
        it["id"] = i

    # 可选：聚合输出
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payloads: List[Dict[str, object]] = []
        for it in items:
            seg_meta = it.get("segment_meta") or it.get("_segment_meta")
            s0 = 0.0
            slen = 0.0
            try:
                if isinstance(seg_meta, dict):
                    s0 = float(seg_meta.get("start_seconds", 0.0))
                    e0 = float(seg_meta.get("end_seconds", 0.0))
                    slen = max(0.0, e0 - s0)
            except Exception:
                pass
            # Build question array (single question for follow_team)
            q_arr = [
                {"time": 0.0, "count": 0, "text": str(it.get("question", ""))},
            ]
            # Build answers with relative time and count mapping
            a_arr = []
            for a in it.get("answer", []) or []:
                try:
                    st = float(a.get("start", 0.0))
                    en = float(a.get("end", st))
                except Exception:
                    st = 0.0
                    en = st
                a_arr.append(
                    {
                        "start": round(max(0.0, st - s0), 6),
                        "end": round(max(0.0, en - s0), 6),
                        "count": 0,
                        "text": str(a.get("text", "")),
                    }
                )
            pub = {
                "source": "soccer",
                "id": int(it.get("id", 0)),
                "video_id": str(it.get("video_id", "")),
                "data_type": "online",
                "train_stage": 2,
                "length": round(slen, 6),
                "question_category": str(it.get("question_category", "follow_team")),
                "question": q_arr,
                "answer": a_arr,
            }
            payloads.append(pub)
        out_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {len(items)} chain samples to: {out_path}")

    # 单文件输出：按 raw_jsons 的比赛目录结构组织
    if args.per_file:
        written = 0
        mirror_root = Path(args.per_file_root)
        mirror_root.mkdir(parents=True, exist_ok=True)
        subdir_name_raw = args.per_file_subdir
        subdir_name = None if subdir_name_raw is None else subdir_name_raw.strip()
        for it in items:
            raw_src = it.get("raw_source_file") or it.get("_raw_source_file")
            if not raw_src:
                continue
            raw_src_path = Path(str(raw_src))
            # 期望 raw_src_path 形如 raw_jsons/<split>_labels/.../<match>/Labels-v2.json
            parts = raw_src_path.parts
            try:
                idx = parts.index("raw_jsons")
                rel_inside_raw = Path(*parts[idx + 1 : -1])  # 去掉 raw_jsons 和文件名
            except ValueError:
                # 回退：使用去掉文件名的整段相对路径
                rel_inside_raw = raw_src_path.parent

            target_dir = mirror_root / rel_inside_raw
            if subdir_name:
                target_dir = target_dir / subdir_name
            target_dir.mkdir(parents=True, exist_ok=True)

            team = str(it.get("team", "team")).lower()
            trig_ms = int(it.get("trigger_ms", 0))
            chain_type = str(it.get("chain_type", "A"))
            fname = f"chain_{chain_type}_{trig_ms:07d}_{team}.json"
            fp = target_dir / fname
            # Build schema-compliant payload as above
            seg_meta = it.get("segment_meta") or it.get("_segment_meta")
            s0 = 0.0
            slen = 0.0
            try:
                if isinstance(seg_meta, dict):
                    s0 = float(seg_meta.get("start_seconds", 0.0))
                    e0 = float(seg_meta.get("end_seconds", 0.0))
                    slen = max(0.0, e0 - s0)
            except Exception:
                pass
            q_arr = [
                {"time": 0.0, "count": 0, "text": str(it.get("question", ""))},
            ]
            a_arr = []
            for a in it.get("answer", []) or []:
                try:
                    st = float(a.get("start", 0.0))
                    en = float(a.get("end", st))
                except Exception:
                    st = 0.0
                    en = st
                a_arr.append(
                    {
                        "start": round(max(0.0, st - s0), 6),
                        "end": round(max(0.0, en - s0), 6),
                        "count": 0,
                        "text": str(a.get("text", "")),
                    }
                )
            pub = {
                "source": "soccer",
                "id": int(it.get("id", 0)),
                "video_id": str(it.get("video_id", "")),
                "data_type": "online",
                "train_stage": 2,
                "length": round(slen, 6),
                "question_category": str(it.get("question_category", "follow_team")),
                "question": q_arr,
                "answer": a_arr,
            }
            fp.write_text(json.dumps(pub, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
        print(
            f"Wrote {written} per-file chain QA JSONs under mirror root: {mirror_root}"
            + (f" (subdir='{subdir_name}')" if subdir_name else "")
            + "."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
