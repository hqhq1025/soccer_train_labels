# Soccer Dense Segment Extractors / 足球稠密片段提取工具

本仓库提供两个脚本，用于从 SoccerNet 风格的 `Labels-v2.json` 注释中提取“事件密度较高”的时间片段：
- 多事件密度片段：`scripts/extract_dense_segments.py`
- 单一事件密度片段：`scripts/extract_single_event_segments.py`

Two Python scripts extract “dense” event segments from SoccerNet‑style `Labels-v2.json` files:
- Multi‑event density segments: `scripts/extract_dense_segments.py`
- Single‑event density segments: `scripts/extract_single_event_segments.py`

---

## Quick Start / 快速开始

- 单文件（多事件）：
  ```bash
  python3 scripts/extract_dense_segments.py \
    --path "england_epl/.../Labels-v2.json" \
    --threshold 1 --bin-size 10 --min-seconds 20 --max-seconds 120 \
    --lead-seconds 5 \
    --out-dir dense_segments \
    --out dense_segments/summary.json
  ```

- 单文件（单一事件，阈值更稀疏示例 0.5）：
  ```bash
  python3 scripts/extract_single_event_segments.py \
    --path "england_epl/.../Labels-v2.json" \
    --threshold 0.5 --bin-size 10 --min-seconds 20 --max-seconds 120 \
    --lead-seconds 5 \
    --out-dir dense_single_event_segments \
    --out dense_single_event_segments/summary.json
  ```

- 目录递归运行（`--path` 指向数据集根目录，如 `.` 或 `raw_jsons`）：
  ```bash
  # 多事件
  python3 scripts/extract_dense_segments.py --path . --out-dir dense_segments --out dense_segments/summary.json

  # 单一事件（会按标签创建子目录）
  python3 scripts/extract_single_event_segments.py --path . --threshold 0.5 --out-dir dense_single_event_segments --out dense_single_event_segments/summary.json
  ```

---

## Input Format / 输入格式

脚本假设 SoccerNet 风格的 JSON 结构（关键字段）：
- `annotations`: 列表，每个元素包含：
  - `gameTime`: 形如 `"1 - 05:19"`（半场编号 - 分:秒）
  - `label`: 事件标签（如 `Corner`, `Throw-in`, `Goal` 等）
  - `position`: 相对整场的毫秒时间戳（字符串数字）
  - `team`: `left`/`right`/`not applicable`
  - `visibility`: 可见性，如 `visible`、`not shown`
- 其他可选字段：`UrlLocal`, `UrlYoutube` 等（若存在将透传到导出的片段）

The scripts expect SoccerNet‑style JSON with key fields listed above; only these are required by the extractors.

---

## Core Idea / 核心思路

- 时间轴按固定窗口大小分箱（默认 `--bin-size 10` 秒）。
- 在给定时长区间 `--min-seconds ... --max-seconds` 内寻找“连续片段”，使得事件“密度”满足阈值 `--threshold`：
  - `avg` 模式：片段内平均每 bin 事件数 ≥ 阈值（默认）
  - `strict` 模式：片段内每个 bin 的事件数都 ≥ 阈值
- 片段对齐至 bin 边界。默认不重叠（从左到右贪心选择）；若指定 `--allow-overlap`，则输出所有满足条件的窗口。
- 片段“前置空白”约束：`--lead-seconds`（默认 5s）。片段内首个事件相对于片段起点至少 `lead-seconds` 秒之后出现，否则该窗口不会被选中。
- 过滤：默认忽略 `visibility == "not shown"` 的事件；可通过 `--include-not-shown` 包含，或用 `--only-visible` 仅保留 `visible`。

Time is discretized into fixed bins; windows with sufficient density are selected, aligned to bin boundaries, with an optional lead‑time requirement for the first event in the segment.

---

## Script 1: Multi‑Event Segments / 多事件密度片段

Path: `scripts/extract_dense_segments.py`

- 作用：对（过滤后的）所有事件计数，寻找密度高的时间片段。
- 常用参数：
  - `--path`: `Labels-v2.json` 文件或目录（递归查找）
  - `--threshold`: 每个 bin 的事件阈值（默认 1.0）
  - `--bin-size`: 分箱大小（秒），默认 10
  - `--min-seconds` / `--max-seconds`: 片段时长范围（默认 20/120）
  - `--lead-seconds`: 片段开始到首事件的最小间隔（默认 5.0）
  - `--mode`: `avg` 或 `strict`
  - `--allow-overlap`: 允许重叠窗口
  - 过滤相关：`--only-visible`、`--include-visibility`、`--include-label`、`--exclude-label`、`--include-not-shown`
  - 输出：`--out-dir`（默认 `dense_segments`，镜像原目录结构）、`--out`（汇总 JSON）

- 输出结构：
  - 片段文件：`<out-dir>/<联赛>/<赛季>/<比赛>/segment_{start}_{end}.json`
  - 汇总文件：`<out>`（若提供），包含每个源文件的设置与片段列表

- 单段 JSON 示例（关键字段）：
  ```json
  {
    "UrlLocal": "england_epl/.../",
    "UrlYoutube": "",
    "source_file": ".../Labels-v2.json",
    "segment": {
      "start_seconds": 300,
      "end_seconds": 360,
      "length_seconds": 60,
      "start_time_mmss": "05:00",
      "end_time_mmss": "06:00",
      "bin_size_seconds": 10,
      "threshold_per_bin": 1.0,
      "mode": "avg"
    },
    "annotations": [ { ... 仅包含片段时间窗内且通过过滤的注释 ... } ]
  }
  ```

---

## Script 2: Single‑Event Segments / 单一事件密度片段

Path: `scripts/extract_single_event_segments.py`

- 作用：遍历数据中出现的每种标签，按“单一标签”独立计数与筛选高密度片段；每个标签输出到独立子目录，避免冲突。
- 常用参数（除与脚本 1 相同的参数外）：
  - `--labels`: 仅处理指定标签（大小写不敏感）
  - `--exclude-labels`: 排除指定标签
  - `--out-dir`: 默认 `dense_single_event_segments`；会在其下创建 `<label>/...` 子目录

- 输出结构：
  - 片段文件：`<out-dir>/<label>/<联赛>/<赛季>/<比赛>/segment_{start}_{end}.json`
    - `<label>` 为归一化目录名（小写，空格/特殊字符替换为 `-`/`_`）
  - 汇总文件：`<out>`（若提供），包含：
    ```json
    {
      "path": "<root>",
      "label_count": 17,
      "labels": ["Corner", "Throw-in", ...],
      "total_segments": 42356,
      "settings": { ... 所有运行参数 ... },
      "per_label_summary": [
        {"label": "ball-out-of-play", "segments_total": 10637, ...},
        ...
      ]
    }
    ```

---

## Examples / 示例

- 更稀疏阈值（单一事件，0.5）：
  ```bash
  python3 scripts/extract_single_event_segments.py \
    --path . \
    --threshold 0.5 \
    --bin-size 10 --min-seconds 20 --max-seconds 120 \
    --lead-seconds 5 \
    --out-dir dense_single_event_segments \
    --out dense_single_event_segments/summary.json
  ```

- 更严格分布（每个 bin 都满足）：
  ```bash
  python3 scripts/extract_dense_segments.py --path . --mode strict --threshold 1
  ```

- 仅保留可见事件：
  ```bash
  python3 scripts/extract_dense_segments.py --path . --only-visible
  ```

- 指定/排除标签（单一事件脚本）：
  ```bash
  python3 scripts/extract_single_event_segments.py --path . --labels Corner Throw-in
  python3 scripts/extract_single_event_segments.py --path . --exclude-labels "Ball out of play"
  ```

---

## Output Conventions / 输出约定

- 文件命名：`segment_{start}_{end}.json`，`start`/`end` 为秒数取整并 6 位补零（例如 `000300` 表示 300s）。
- 片段边界：与 `--bin-size` 对齐（通常是 10 的倍数）。
- 目录镜像：输出目录在联赛/赛季/比赛层级上镜像输入路径。
- 片段注释：仅包含片段时间窗内（`[start, end)`）并满足过滤条件的原始注释，字段保持不变。

---

## Lead Time Rule / 片段前置时间约束

- 默认 `--lead-seconds 5`：确保片段开始后的前 5 秒内没有首个事件，便于模型/标注拥有上下文。
- 可按需调整：`--lead-seconds 2.5` 或 `--lead-seconds 0`（不建议）。

---

## Notes & Tips / 说明与提示

- 可见性：默认忽略 `visibility == "not shown"`。如需包含，添加 `--include-not-shown`；如需仅 `visible`，使用 `--only-visible`。
- 规模与磁盘：在整库运行时，单一事件脚本会为每个标签生成大量片段（示例结果 4 万+ 文件，仅供参考，实际依数据集而定）。
- 复现实验：建议先对单场/少量比赛跑 demo 验证，再全量运行。

---

## Development / 开发

- 依赖：Python 3（仅标准库）。
- 代码风格：两个脚本均自包含、无第三方包；`extract_single_event_segments.py` 动态加载复用 `extract_dense_segments.py` 的核心逻辑，避免重复。
- 贡献：欢迎反馈问题与改进建议（参数设计、导出格式、自适应阈值、Top‑K 片段等）。

---

## FAQ

- Q: 如何统计导出片段的数量？
  - A: 
    ```bash
    find dense_segments -type f -name "segment_*.json" | wc -l
    find dense_single_event_segments -type f -name "segment_*.json" | wc -l
    ```

- Q: 能否保证片段内只包含某一种事件？
  - A: 使用单一事件脚本（Script 2），它在计数与导出时都仅保留该事件的注释。

- Q: 能否输出重叠窗口？
  - A: 可以，添加 `--allow-overlap`，注意可能产生大量片段。

---

如需定制更复杂的筛选策略（例如按标签自适应阈值、最小事件数限制、Top‑K 最密片段等），请提出需求。

If you need customized strategies (per‑label adaptive thresholds, minimum event counts per segment, top‑K densest segments, etc.), feel free to request.

---

## Build QA Datasets / 构建 QA 数据集

Path: `scripts/build_qa_from_segments.py`

- 作用：将已切好的片段 JSON（多事件或单事件）转换为目标 QA 格式（数组或逐片段单文件），用于“状态确认 / 指令遵循”类任务。
- 任务类型：
  - `--task multi`：监控一组标签，发生时回答“是哪种事件”。`question_category` 自动置为 `Status Confirmation`。
  - `--task single`：监控单一标签，发生时提醒。`question_category` 自动置为 `Instruction Following`。
- 时间基准：
  - `--time-mode global`（默认演示）：`start/end` 为整场相对秒（来自 `position`）。
  - `--time-mode segment`：相对片段起点的秒。
- 问句模板（英文）：
  - 内置多样化模板，自动随机选取（可通过 `--random-seed` 复现；可用 `--question-templates` 自定义）。
  - 多事件问句会在花括号中列出全部监控标签；单事件在花括号中仅放该标签。
- 输出字段：`source`、`id`、`video_id`（由源路径稳定派生 UUID5）、`data_type`、`train_stage`、`length`、`question_category`、`question`、`answer[]`、`segment_path`。

### Aggregated JSON / 汇总文件

- 多事件（列出全部标签）：
  ```bash
  python3 scripts/build_qa_from_segments.py \
    --task multi \
    --segments-dir dense_segments \
    --output qa/multi/qa_multi_all_labels.json \
    --time-mode global \
    --multi-use-all-labels \
    --random-seed 123
  ```

- 单事件（以角球为例）：
  ```bash
  python3 scripts/build_qa_from_segments.py \
    --task single --label "Corner" \
    --segments-dir dense_single_event_segments/corner \
    --output qa/single/corner.json \
    --time-mode global \
    --random-seed 123
  ```

### Per‑Segment Files / 每片段单文件

- 多事件（全量逐片段）：
  ```bash
  python3 scripts/build_qa_from_segments.py \
    --task multi \
    --segments-dir dense_segments \
    --output qa/multi/placeholder.json \
    --per-file --output-dir qa/multi_files \
    --time-mode global \
    --multi-use-all-labels \
    --random-seed 123
  ```

- 单事件（按标签目录逐片段）：
  ```bash
  python3 scripts/build_qa_from_segments.py \
    --task single --label "Corner" \
    --segments-dir dense_single_event_segments/corner \
    --output qa/single/placeholder.json \
    --per-file --output-dir qa/single_files/corner \
    --time-mode global \
    --random-seed 123
  ```

### Output Layout / 输出目录组织

- 多事件每片段：`qa/multi_files/<联赛>/<赛季>/<比赛>/segment_XXXXXX_YYYYYY.qa.json`
- 单事件每片段：`qa/single_files/<label>/<联赛>/<赛季>/<比赛>/segment_XXXXXX_YYYYYY.qa.json`

每个 QA JSON 均包含 `segment_path` 指向其来源的片段 JSON，以便追溯。

### Count / 数量统计

```bash
find qa/multi_files  -type f -name "*.qa.json" | wc -l   # 多事件条目数
find qa/single_files -type f -name "*.qa.json" | wc -l   # 单事件条目数
```

### Template Customization / 自定义问句模板

- 提供一个 JSON 文件：
  ```json
  { "single": ["Watch the video and alert me when {TARGET_LABEL} occurs."],
    "multi":  ["Monitor {EVENT_SET} and report which event occurs each time."] }
  ```
- 通过 `--question-templates my_templates.json` 指定。
