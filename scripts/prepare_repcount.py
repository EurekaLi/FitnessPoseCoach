"""将 RepCount-A 官方 CSV 转为三类动作的视频清单，不改动原始数据。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("squat", "push_up", "jump_jack")
# 当前下载包实际存在拼写错误和两种命名；保留 action_raw 以追溯原标注。
ALIASES = {"squat": "squat", "squant": "squat", "push_up": "push_up",
           "pushups": "push_up", "jump_jack": "jump_jack", "jumpjacks": "jump_jack"}


def parse_count(text: str) -> int | None:
    """原数据的空计数表示未标注，不能转换成零次或根据周期数补造标签。"""
    if not text.strip():
        return None
    value = float(text)
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        raise ValueError(f"非法 count：{text}")
    return int(value)


def cycle_annotations(row: dict) -> tuple[list[list[int]], bool]:
    """保留 L1/L2、L3/L4 等原视频帧边界；不把边界自动变为逐帧阶段标签。"""
    keys = sorted((key for key in row if key and key.startswith("L") and key[1:].isdigit()),
                  key=lambda key: int(key[1:]))
    values = []
    for key in keys:
        text = (row[key] or "").strip()
        if text:
            value = float(text)
            if not math.isfinite(value) or value < 0 or not value.is_integer():
                raise ValueError(f"{row['name']} 存在非法帧边界：{key}={text}")
            values.append(int(value))
    count_value = parse_count(row["count"])
    pairs = [values[i:i + 2] for i in range(0, len(values) - 1, 2)]
    complete = (count_value is not None and len(values) == 2 * count_value
                and all(start <= end for start, end in pairs)
                and values == sorted(values))
    return pairs, complete


def make_records(root: Path) -> list[dict]:
    records, seen = [], set()
    for split in ("train", "valid", "test"):
        annotation_path = root / "annotation" / f"{split}.csv"
        with annotation_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"type", "name", "count"}.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV 缺少必要字段：{annotation_path}")
            for row in reader:
                raw_action = row["type"].strip().lower()
                if raw_action not in ALIASES:
                    continue
                name = row["name"].strip()
                if Path(name).name != name or not name.lower().endswith(".mp4"):
                    raise ValueError(f"非预期视频文件名：{name}")
                if name.casefold() in seen:
                    raise ValueError(f"目标视频在官方清单中重复：{name}；停止，避免跨集合重复")
                seen.add(name.casefold())
                video = root / "video" / split / name
                if not video.is_file():
                    raise FileNotFoundError(f"视频未解压或缺失：{video}")
                cycles, complete = cycle_annotations(row)
                action = ALIASES[raw_action]
                try:
                    video_path = video.resolve().relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    video_path = str(video.resolve())
                records.append({
                    "sample_id": f"{split}/{Path(name).stem}",
                    "official_split": split,
                    "action": action,
                    "action_raw": raw_action,
                    "class_id": CLASSES.index(action),
                    "video_path": video_path,
                    "count": parse_count(row["count"]),
                    "cycle_frame_pairs": cycles,
                    "cycle_annotation_complete": complete,
                })
    order = {split: i for i, split in enumerate(("train", "valid", "test"))}
    records.sort(key=lambda row: (order[row["official_split"]], row["class_id"], row["sample_id"]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data/raw/repcount_a/LLSP")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a")
    args = parser.parse_args()
    records = make_records(args.root.resolve())
    if not records:
        raise ValueError("没有找到目标类别的视频")
    counts = Counter((row["official_split"], row["action"]) for row in records)
    summary = {
        "dataset_root": str(args.root.resolve()), "classes": CLASSES,
        "video_count": len(records),
        "counts": {split: {action: counts[split, action] for action in CLASSES}
                   for split in ("train", "valid", "test")},
        "cycle_annotation_warnings": sum(not row["cycle_annotation_complete"] for row in records),
        "missing_count_labels": sum(row["count"] is None for row in records),
        "split_scope": "official_video_split; not verified subject/source-video independent",
        "fitness_aqa": "application_pending; not used for this stage",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest: {manifest}")
    print("REPCOUNT_MANIFEST_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
