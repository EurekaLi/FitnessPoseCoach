"""训练/验证共享的窗口读取与三分类指标；不依赖额外的机器学习指标库。"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

CLASSES = ("squat", "push_up", "jump_jack")


def load_index(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, groups, samples = set(), {}, {}
    for row in rows:
        split = row["training_split"]
        if split not in ("train", "valid"):
            raise ValueError("训练入口只接受 train/valid，不读取测试集")
        if row["window_id"] in seen:
            raise ValueError("重复窗口")
        seen.add(row["window_id"])
        if row["class_id"] not in range(3) or CLASSES[row["class_id"]] != row["action"]:
            raise ValueError("类别编号与动作名称不匹配")
        for key, table in ((row["source_group"], groups), (row["sample_id"], samples)):
            if key in table and table[key] != split:
                raise ValueError(f"同一视频或来源跨集合：{key}")
            table[key] = split
    for split in ("train", "valid"):
        if {r["class_id"] for r in rows if r["training_split"] == split} != {0, 1, 2}:
            raise ValueError(f"{split} 没有覆盖全部三类")
    return rows


class WindowDataset:
    """每个文件只解压一次并预加载选中窗口，减少 /mnt/d 上反复读取压缩文件的开销。

    当前约一万多窗口仅需几百 MB，默认不派生加载进程，避免复制数据/重复解压。
    """

    def __init__(self, rows: list[dict], root: Path):
        self.rows = rows
        self.x = np.empty((len(rows), 3, 48, 17), dtype=np.float32)
        self.y = np.array([r["class_id"] for r in rows], dtype=np.int64)
        by_path = defaultdict(list)
        for i, row in enumerate(rows):
            by_path[row["cache_path"]].append((i, row["window_index"]))
        for name, selections in by_path.items():
            with np.load(root / name, allow_pickle=False) as cache:
                data = cache["x"]
                if data.ndim != 4 or data.shape[1:] != (3, 48, 17):
                    raise ValueError(f"模型需要 (N,3,48,17)，实际为 {data.shape}：{name}")
                for i, offset in selections:
                    if not 0 <= offset < len(data):
                        raise ValueError(f"窗口索引越界：{name}/{offset}")
                    self.x[i] = data[offset]
        if not np.isfinite(self.x).all():
            raise ValueError("模型输入包含 NaN/Inf")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], index

    def sampling_weights(self) -> np.ndarray:
        """使每类、每段视频的抽样总权重相等，避免长视频的重叠窗口主导训练。"""
        windows_per_video = Counter(row["sample_id"] for row in self.rows)
        video_class = {row["sample_id"]: row["class_id"] for row in self.rows}
        videos_per_class = Counter(video_class.values())
        return np.array([1.0 / (windows_per_video[r["sample_id"]] * videos_per_class[r["class_id"]])
                         for r in self.rows], dtype=np.float64)


def classification_metrics(labels, predictions) -> dict:
    """矩阵行是真值、列是预测；宏 F1 对三类等权，零分母记零。"""
    matrix = np.zeros((3, 3), dtype=np.int64)
    for label, pred in zip(labels, predictions):
        matrix[int(label), int(pred)] += 1
    tp = matrix.diagonal().astype(float)
    precision = tp / np.maximum(matrix.sum(0), 1)
    recall = tp / np.maximum(matrix.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(tp.sum() / max(matrix.sum(), 1)), "macro_f1": float(f1.mean()),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(matrix[i].sum())} for i, name in enumerate(CLASSES)},
            "confusion_matrix": matrix.tolist()}


def summarize_predictions(rows: list[dict], probabilities: np.ndarray) -> dict:
    by_video = defaultdict(list)
    by_source = defaultdict(list)
    for i, row in enumerate(rows):
        by_video[row["sample_id"]].append(i)
        by_source[(row["source_group"], row["class_id"])].append(i)
    videos = []
    for sample, indexes in by_video.items():
        row = rows[indexes[0]]
        mean = probabilities[indexes].mean(0)
        videos.append({"sample_id": sample, "source_group": row["source_group"], "label": row["class_id"],
                       "prediction": int(mean.argmax()), "probabilities": mean.tolist(), "windows": len(indexes)})
    # 这是整段视频汇总指标，不作为“当前时刻”在线输出；每个输入窗口自身仍只含过去帧。
    source_labels, source_preds = [], []
    for (_, label), indexes in by_source.items():
        source_labels.append(label)
        source_preds.append(int(probabilities[indexes].mean(0).argmax()))
    return {"window": classification_metrics([r["class_id"] for r in rows], probabilities.argmax(1)),
            "video": classification_metrics([r["label"] for r in videos], [r["prediction"] for r in videos]),
            "source_action": classification_metrics(source_labels, source_preds), "videos": videos}
