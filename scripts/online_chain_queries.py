#!/usr/bin/env python3
"""
在线链式事件查询 (dense_segments)

功能
- 以“在线”顺序遍历 dense_segments 下的分段 JSON，按比赛时间流式输出事件。
- 支持链式查询：
  1) 等待下一个满足条件的触发事件（如：Throw-in/手抛球）；
  2) 锁定该事件所属球队（left/right）；
  3) 继续跟踪并报告该球队后续的事件（可限定关注的事件类型、数量、时间窗）。

使用示例
- 等待下一个 Throw-in，并输出该球队后续 5 个可判队事件：
  python scripts/online_chain_queries.py \
    --base-dir train/dense_segments \
    --trigger 手抛球 \
    --follow-count 5

- 等待下一个 Throw-in，随后只关注该队的 Foul 或 Corner（提醒 3 次）：
  python scripts/online_chain_queries.py \
    --base-dir train/dense_segments \
    --trigger Throw-in \
    --follow-label 犯规 角球 \
    --follow-count 3

说明
- 仅考虑能判定球队的事件（team ∈ {left, right}），忽略 team == 'not applicable'。
- 默认忽略 visibility == 'not shown' 的标注（可用 --include-not-shown 开启）。
- 事件按“比赛路径 + 上半/下半 + 半场内秒数”进行排序，同一比赛内时间是有序的；
  跨比赛仅按路径字典序保证确定性（并不代表真实时间先后）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Tuple


# ---------------------- 中文 -> 数据集英文 标签映射 ----------------------

CH2EN: Dict[str, List[str]] = {
    # Throw-in
    "手抛球": ["Throw-in"],
    "界外球": ["Throw-in"],
    "边线球": ["Throw-in"],

    # Corner
    "角球": ["Corner"],

    # Foul
    "犯规": ["Foul"],

    # Offside
    "越位": ["Offside"],

    # Shots
    "射正": ["Shots on target"],
    "射偏": ["Shots off target"],

    # Goal
    "进球": ["Goal"],

    # Clearance
    "解围": ["Clearance"],

    # Free-kick
    "任意球": ["Direct free-kick", "Indirect free-kick"],
    "直接任意球": ["Direct free-kick"],
    "间接任意球": ["Indirect free-kick"],

    # Kick-off
    "开球": ["Kick-off"],

    # Substitution
    "换人": ["Substitution"],
}


TEAMFUL_VALUES = {"left", "right"}


def normalize_label_inputs(labels: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not labels:
        return None
    out: List[str] = []
    for x in labels:
        if not x:
            continue
        # 如果是中文，映射到英文标签列表；否则直接按原样
        engs = CH2EN.get(x.strip(), None)
        if engs:
            out.extend(engs)
        else:
            out.append(x.strip())
    # 去重保持顺序
    seen = set()
    uniq: List[str] = []
    for l in out:
        ll = l.lower()
        if ll not in seen:
            seen.add(ll)
            uniq.append(l)
    return uniq or None


def parse_game_time(game_time: str) -> Tuple[int, int]:
    """将 "1 - mm:ss" / "2 - mm:ss" 解析为 (half, seconds_in_half).
    若解析失败，返回 (9, 10**9) 以便排序在末尾。
    """
    try:
        part = game_time.strip().split("-")
        if len(part) != 2:
            return (9, 10**9)
        half = int(part[0].strip())
        mm, ss = part[1].strip().split(":")
        sec = int(mm) * 60 + int(ss)
        return (half, sec)
    except Exception:
        return (9, 10**9)


@dataclass
class DenseEvent:
    match_id: str           # 来自 UrlLocal (如: england_epl/.../)
    file_path: Path         # segment_*.json 路径
    half: int               # 1/2
    sec_in_half: int        # 半场内秒数
    game_time: str          # 原始 "h - mm:ss"
    label: str              # 事件标签
    team: str               # left/right
    visibility: str         # visible/not shown/...

    def abs_sort_key(self) -> Tuple[str, int, int, str, str]:
        # 跨比赛：按 match_id 字典序；比赛内：按 half, sec 排序
        return (self.match_id, self.half, self.sec_in_half, self.label.lower(), self.game_time)


def iter_dense_events(
    base_dir: Path,
    include_labels: Optional[Sequence[str]] = None,
    only_teamful: bool = True,
    ignore_not_shown: bool = True,
) -> Generator[DenseEvent, None, None]:
    """遍历 base_dir (dense_segments) 下所有 segment_*.json，产出事件流。
    - include_labels: 若提供，仅保留这些标签（不区分大小写）。
    - only_teamful: 仅保留 team in {left,right}。
    - ignore_not_shown: 过滤 visibility == 'not shown'。
    """
    include_set = {l.lower() for l in include_labels} if include_labels else None

    # 先收集所有文件，按路径排序，保证确定性
    files = sorted(base_dir.rglob("segment_*.json"))
    if not files:
        return

    # 为避免跨 segment 的重复事件，按比赛（UrlLocal）维度去重
    cur_match: Optional[str] = None
    seen_keys: set = set()

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

        match_id = str(data.get("UrlLocal") or "").strip()
        if not match_id:
            # 尝试从路径推断：dense_segments/.../<match_dir>/segment_*.json
            match_id = str(fp.parent.relative_to(base_dir)) + "/"

        # 当切换比赛时，重置去重 set
        if cur_match != match_id:
            cur_match = match_id
            seen_keys = set()

        anns = data.get("annotations", []) or []
        # 排序：按 half, sec_in_half
        parsed: List[DenseEvent] = []
        for a in anns:
            team = str(a.get("team", "")).lower().strip()
            if only_teamful and team not in TEAMFUL_VALUES:
                continue
            vis = str(a.get("visibility", "")).lower().strip()
            if ignore_not_shown and vis == "not shown":
                continue
            label = str(a.get("label", "")).strip()
            if include_set is not None and label.lower() not in include_set:
                continue
            gt = str(a.get("gameTime", "")).strip()
            half, sec = parse_game_time(gt)
            parsed.append(
                DenseEvent(
                    match_id=match_id,
                    file_path=fp,
                    half=half,
                    sec_in_half=sec,
                    game_time=gt,
                    label=label,
                    team=team,
                    visibility=vis,
                )
            )

        parsed.sort(key=lambda e: (e.half, e.sec_in_half, e.label.lower(), e.game_time))

        for ev in parsed:
            # 去重 key: (half, sec, label, team)
            key = (ev.half, ev.sec_in_half, ev.label.lower(), ev.team)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            yield ev


def run_chain(
    base_dir: Path,
    trigger_labels: Sequence[str],
    follow_labels: Optional[Sequence[str]],
    follow_count: int,
    follow_window_seconds: Optional[int],
    include_not_shown: bool,
    debug_match_prefix: Optional[str] = None,
) -> int:
    """执行链式：触发 -> 锁队 -> 跟踪

    - trigger_labels: 等待的触发事件集合（或其中之一命中即可）。
    - follow_labels: 只关注这些后续事件；若为空则输出该队所有可判队事件。
    - follow_count: 最多提醒多少次后续事件。
    - follow_window_seconds: 仅在触发事件后的该时间窗内（半场内秒数）进行提醒；为空则不限。
    - include_not_shown: 是否包含 visibility == 'not shown'。
    - debug_match_prefix: 若提供，仅处理路径包含该前缀的比赛，便于调试。
    """
    inc_trigger = [l.strip() for l in trigger_labels if l.strip()]
    inc_follow = [l.strip() for l in (follow_labels or []) if l.strip()]
    inc_all = sorted(set(inc_trigger + inc_follow)) if inc_follow else inc_trigger

    # 1) 遍历找到触发事件
    trigger_ev: Optional[DenseEvent] = None
    for ev in iter_dense_events(
        base_dir,
        include_labels=inc_all,
        only_teamful=True,
        ignore_not_shown=not include_not_shown,
    ):
        if debug_match_prefix and debug_match_prefix not in ev.match_id:
            continue
        if ev.label.lower() in {l.lower() for l in inc_trigger}:
            trigger_ev = ev
            break

    if trigger_ev is None:
        print("[未找到触发事件] 请检查 --trigger 标签或数据目录。", file=sys.stderr)
        return 1

    print(
        "触发: {lab} @ {gt} | 球队: {team} | 比赛: {mid}".format(
            lab=trigger_ev.label,
            gt=trigger_ev.game_time,
            team=trigger_ev.team,
            mid=trigger_ev.match_id,
        )
    )

    # 2) 锁定球队，继续遍历并提醒
    locked_team = trigger_ev.team
    locked_match = trigger_ev.match_id
    start_half = trigger_ev.half
    start_sec = trigger_ev.sec_in_half

    n_reported = 0
    follow_set = {l.lower() for l in inc_follow} if inc_follow else None

    for ev in iter_dense_events(
        base_dir,
        include_labels=inc_all if inc_follow else None,  # 若无 follow 限制，则不过滤
        only_teamful=True,
        ignore_not_shown=not include_not_shown,
    ):
        # 只在同一比赛中继续
        if ev.match_id != locked_match:
            continue
        # 只考虑触发事件之后（半场编号 + 半场秒数均用于比较）
        if (ev.half, ev.sec_in_half) < (start_half, start_sec):
            continue
        # 跳过本身这条触发事件
        if ev is trigger_ev or (
            ev.half == start_half and ev.sec_in_half == start_sec and ev.label == trigger_ev.label
        ):
            continue
        # 仅跟踪同一球队
        if ev.team != locked_team:
            continue
        # 若设定了时间窗
        if follow_window_seconds is not None:
            dt = (ev.half - start_half) * 10**6 + (ev.sec_in_half - start_sec)
            # 简化处理：若跨半场则视为超窗（因为上式 half 差权重大）
            if ev.half != start_half or dt < 0 or dt > follow_window_seconds:
                continue

        if (follow_set is None) or (ev.label.lower() in follow_set):
            n_reported += 1
            print(
                "随后: {lab} @ {gt} | 球队: {team}".format(
                    lab=ev.label, gt=ev.game_time, team=ev.team
                )
            )
            if n_reported >= follow_count:
                break

    if n_reported == 0:
        print("(无符合条件的后续事件)")
    else:
        print(f"共提醒 {n_reported} 条后续事件。")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Dense segments 在线链式事件查询 (team-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="train/dense_segments",
        help="dense_segments 根目录",
    )
    p.add_argument(
        "--trigger",
        nargs=1,
        required=True,
        help="触发事件（中文或英文标签均可，如 '手抛球'/'Throw-in'）",
    )
    p.add_argument(
        "--follow-label",
        dest="follow_labels",
        nargs="*",
        default=None,
        help="后续关注的事件标签（中文或英文，留空则关注该队所有可判队事件）",
    )
    p.add_argument(
        "--follow-count",
        type=int,
        default=5,
        help="最多提醒多少条后续事件",
    )
    p.add_argument(
        "--follow-window-seconds",
        type=int,
        default=None,
        help="仅在触发后该时间窗内提醒（半场内秒数；跨半场不计）",
    )
    p.add_argument(
        "--include-not-shown",
        action="store_true",
        help="包含 visibility == 'not shown' 的事件",
    )
    p.add_argument(
        "--debug-match-prefix",
        type=str,
        default=None,
        help="仅处理比赛路径包含该前缀的样本（便于调试）",
    )

    args = p.parse_args(argv)

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Base dir 不存在: {base_dir}", file=sys.stderr)
        return 2

    trigger_labels = normalize_label_inputs(args.trigger)
    follow_labels = normalize_label_inputs(args.follow_labels)

    if not trigger_labels:
        print("请提供有效的 --trigger 标签", file=sys.stderr)
        return 2

    return run_chain(
        base_dir=base_dir,
        trigger_labels=trigger_labels,
        follow_labels=follow_labels,
        follow_count=max(1, int(args.follow_count)),
        follow_window_seconds=args.follow_window_seconds,
        include_not_shown=bool(args.include_not_shown),
        debug_match_prefix=args.debug_match_prefix,
    )


if __name__ == "__main__":
    raise SystemExit(main())

