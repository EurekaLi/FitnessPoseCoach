"""只改训练索引：按已核实的原视频 ID 隔离来源，原数据与骨架缓存保持不变。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("squat", "push_up", "jump_jack")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mapping(path: Path) -> dict[str, dict]:
    """读取官方 data_ours 表的原始/缓存值；Excel 错误值保留，不伪造来源。

    仅需标准库解析这个固定格式的 XLSX，避免为了训练新增表格软件依赖。
    stuid 的身份语义未证实，不能据此宣称跨人物划分。
    """
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = next(s for s in workbook.find("s:sheets", NS) if s.attrib["name"] == "data_ours")
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in relationships if r.attrib["Id"] == rel_id)
        sheet_path = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(t.text or "" for t in si.iterfind(".//s:t", NS))
                      for si in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        rows = []
        for row in ET.fromstring(archive.read(sheet_path)).iterfind(".//s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                value = cell.findtext("s:v", "", NS)
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in cell.iterfind(".//s:t", NS))
                values[column] = value
            rows.append(values)
    headers = rows[0]
    if [headers.get(c) for c in ("B", "C", "D", "E", "F")] != ["class", "vid_st", "vid", "stuid", "rename"]:
        raise ValueError("官方映射表结构变化，需要先确认列含义")
    result = {}
    for row in rows[1:]:
        name = row.get("F", "")
        if not name or name in result:
            raise ValueError(f"映射表名称为空或重复：{name}")
        result[name] = {"action": row.get("B"), "source_video_id": row.get("D", ""),
                        "original_clip": row.get("C"), "stuid_unverified": row.get("E")}
    return result


def assign_sources(manifest: list[dict], mapping: dict) -> tuple[list[dict], list[dict]]:
    """test > valid > train：优先保护现有留出集，而不是把留出来源搬回训练。"""
    grouped, excluded = defaultdict(list), []
    for item in manifest:
        meta = mapping.get(item["sample_id"].split("/", 1)[1])
        reason = "no_mapping" if meta is None else "invalid_source_id"
        if meta is None or not re.fullmatch(r"[A-Za-z0-9_-]{11}", meta["source_video_id"]):
            excluded.append({**item, "reason": reason})
            continue
        if meta["action"] != item["action"]:
            raise ValueError(f"映射表动作与清单冲突：{item['sample_id']}")
        grouped[meta["source_video_id"]].append({**item, **meta})
    assigned = []
    for source, records in sorted(grouped.items()):
        official = {r["official_split"] for r in records}
        split = "test" if "test" in official else "valid" if "valid" in official else "train"
        assigned.extend({**r, "source_group": source, "training_split": split} for r in records)
    return sorted(assigned, key=lambda r: r["sample_id"]), excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/manifest.jsonl")
    parser.add_argument("--mapping", type=Path, default=PROJECT_ROOT / "data/raw/repcount_a/LLSP/original_data/filename_mapping.xlsx")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/normalized_v1/windows_index_train_valid.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/source_split_v1")
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({r["sample_id"] for r in manifest}) != len(manifest):
        raise ValueError("视频清单重复")
    assigned, excluded = assign_sources(manifest, read_mapping(args.mapping))
    lookup = {r["sample_id"]: r for r in assigned}
    raw_index = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    windows, seen, ready = [], set(), set()
    for row in raw_index:
        if row["window_id"] in seen:
            raise ValueError(f"窗口重复：{row['window_id']}")
        seen.add(row["window_id"])
        ready.add(row["sample_id"])
        meta = lookup.get(row["sample_id"])
        if meta is None or meta["training_split"] == "test":
            continue
        if any(row[k] != meta[k] for k in ("action", "class_id", "official_split")):
            raise ValueError(f"窗口与视频标注不一致：{row['sample_id']}")
        windows.append({**row, "source_group": meta["source_group"], "training_split": meta["training_split"]})
    # 测试视频尚未提取骨架也能参与来源隔离，但不要求此时读取测试图像或窗口。
    missing = []
    for row in manifest:
        if row["official_split"] not in ("train", "valid"):
            continue
        summary = args.input.parent / row["sample_id"] / "summary.json"
        if not summary.is_file() or json.loads(summary.read_text(encoding="utf-8")).get("status") != "ok":
            missing.append(row["sample_id"])
    if missing:
        raise ValueError(f"训练/验证缓存不完整，先补齐：{missing}")
    groups = defaultdict(set)
    for row in assigned:
        groups[row["source_group"]].add(row["official_split"])
    counts = {s: {a: sum(r["training_split"] == s and r["action"] == a for r in windows) for a in CLASSES}
              for s in ("train", "valid")}
    if any(n == 0 for per_class in counts.values() for n in per_class.values()):
        raise ValueError(f"分组后有集合/类别没有窗口：{counts}")
    report = {"status": "ok", "policy": "verified_source_only; test > valid > train; no window random split",
              "fingerprints": {k: sha256(p) for k, p in (("manifest", args.manifest), ("mapping", args.mapping), ("input_index", args.input))},
              "input_videos": len(manifest), "input_windows": len(raw_index),
              "verified_videos": len(assigned), "excluded_videos": len(excluded),
              "excluded_reasons": dict(Counter(r["reason"] for r in excluded)),
              "official_cross_split_sources": [{"source_group": k, "official_splits": sorted(v)}
                                                for k, v in groups.items() if len(v) > 1],
              "assigned_video_counts": {s: dict(Counter(r["action"] for r in assigned if r["training_split"] == s))
                                        for s in ("train", "valid", "test")},
              "usable_video_counts": {s: dict(Counter(lookup[k]["action"] for k in sorted({r["sample_id"] for r in windows if r["training_split"] == s})))
                                      for s in ("train", "valid")},
              "window_counts": counts,
              "moved_videos": [{k: r[k] for k in ("sample_id", "source_group", "official_split", "training_split")} for r in assigned
                               if r["official_split"] != r["training_split"]],
              "no_window_verified_train_valid": [r["sample_id"] for r in assigned if r["training_split"] != "test" and r["sample_id"] not in ready],
              "scope": "已知原视频隔离；不是跨人物评估，也不等同于官方全量基准"}
    payloads = {"source_report.json": json.dumps(report, ensure_ascii=False, indent=2),
                "video_assignments.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in assigned),
                "excluded_videos.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in excluded),
                "windows_index.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in windows)}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = args.output / name
        if target.exists():
            old = target.read_text(encoding="utf-8")
            # JSON 对象的键顺序不影响划分含义，不能把展示顺序变化当作实验输入变化。
            same = json.loads(old) == json.loads(content) if target.suffix == ".json" else old == content
            if not same:
                raise FileExistsError(f"输入或结果发生变化，请使用新的 --output，保留原实验划分：{target}")
    for name, content in payloads.items():
        (args.output / name).write_text(content, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("verified_videos", "excluded_reasons", "usable_video_counts", "window_counts", "scope")}, ensure_ascii=False, indent=2))
    print(f"official_cross_split_sources={len(report['official_cross_split_sources'])} moved_videos={len(report['moved_videos'])}")
    print("SOURCE_SPLIT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
