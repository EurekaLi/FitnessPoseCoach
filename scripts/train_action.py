"""训练三分类 ST-GCN++ 适配模型；默认只读来源隔离后的 train/valid，不碰测试集。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "data/processed/repcount_a/source_split_v1/windows_index.jsonl")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="每类最多4个真实窗口，只跑1轮；不是正式效果评估")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error("epochs>=1、batch-size>=2、lr>0")
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from fitness_pose.action_data import CLASSES, WindowDataset, load_index, summarize_predictions
    from fitness_pose.stgcnpp import STGCNPP

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；未自动降级为 CPU。请先确认已激活 fitness-pose")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = False
    rows = load_index(args.index)
    subsets = {split: [r for r in rows if r["training_split"] == split] for split in ("train", "valid")}
    if args.smoke:
        subsets = {split: [r for label in range(3) for r in [x for x in part if x["class_id"] == label][:4]]
                   for split, part in subsets.items()}
    datasets = {s: WindowDataset(part, PROJECT_ROOT) for s, part in subsets.items()}
    output = args.output or PROJECT_ROOT / "outputs/action_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"结果目录已有文件，为避免覆盖请选择新的 --output：{output}")
    output.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else args.epochs
    config = {"architecture": "ST-GCN++ PYSKL architecture adaptation; single person; from scratch",
              "classes": list(CLASSES), "input_shape": ["B", 3, 48, 17], "dropout": 0.1,
              "epochs": epochs, "batch_size": args.batch_size, "seed": args.seed, "smoke_only": args.smoke,
              "optimizer": {"type": "SGD", "lr": args.lr, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
              "scheduler": "cosine by epoch", "selection_metric": "valid.video.macro_f1",
              "sampler": "class-balanced and video-balanced replacement sampling; train only",
              "index_sha256": fingerprint(args.index),
              "code_sha256": {name: fingerprint(PROJECT_ROOT / name) for name in
                              ("scripts/train_action.py", "src/fitness_pose/stgcnpp.py", "src/fitness_pose/action_data.py")},
              "windows": {s: len(d) for s, d in datasets.items()},
              "videos": {s: len({r['sample_id'] for r in part}) for s, part in subsets.items()},
              "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
              "cuda_runtime": torch.version.cuda, "precision": "FP32; no AMP", "workers": 0,
              "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    write_json(output / "config.json", config)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(torch.as_tensor(datasets["train"].sampling_weights()),
                                    num_samples=len(datasets["train"]), replacement=True, generator=generator)
    loaders = {"train": DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler, num_workers=0),
               "valid": DataLoader(datasets["valid"], batch_size=args.batch_size, shuffle=False, num_workers=0)}
    model = STGCNPP().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    print(f"TRAIN_START smoke={args.smoke} windows={config['windows']} videos={config['videos']} "
          f"parameters={sum(p.numel() for p in model.parameters())} output={output}", flush=True)
    best, started = -1.0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for step, (x, y, _) in enumerate(loaders["train"], 1):
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("训练 loss 出现 NaN/Inf；停止，不保存为成功结果")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum().item())
            total += len(y)
            if step == 1 or step % 100 == 0:
                print(f"TRAIN_BATCH epoch={epoch}/{epochs} step={step}/{len(loaders['train'])} "
                      f"loss={loss.item():.4f} grad_norm={float(norm):.3f}", flush=True)
        model.eval()
        probabilities = np.empty((len(datasets["valid"]), 3), dtype=np.float32)
        val_loss = 0.0
        with torch.inference_mode():
            for x, y, indexes in loaders["valid"]:
                x, y = x.to(args.device), y.to(args.device)
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("验证预测出现 NaN/Inf")
                val_loss += criterion(logits, y).item() * len(y)
                probabilities[indexes.numpy()] = torch.softmax(logits, dim=1).cpu().numpy()
        metrics = summarize_predictions(subsets["valid"], probabilities)
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                  "train_loss": total_loss / total, "train_sampled_accuracy": correct / total,
                  "valid_loss": val_loss / len(datasets["valid"]),
                  "valid_window": metrics["window"], "valid_video": metrics["video"],
                  "valid_source_action": metrics["source_action"], "elapsed_s": time.perf_counter() - started}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"EPOCH_OK epoch={epoch}/{epochs} train_loss={record['train_loss']:.4f} "
              f"valid_video_acc={metrics['video']['accuracy']:.4f} "
              f"valid_video_macro_f1={metrics['video']['macro_f1']:.4f}", flush=True)
        score = metrics["video"]["macro_f1"]
        improved = score > best
        best = max(best, score)
        scheduler.step()
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch, "best_macro_f1": best,
                      "config": config, "record": record}
        # 只保留最佳/最后两份，既能部署也不让每轮检查点持续占用磁盘。
        temporary = output / "last.tmp.pt"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
            write_json(output / "best_metrics.json", {"epoch": epoch, "smoke_only": args.smoke, **metrics})
    result = {"status": "smoke_ok" if args.smoke else "training_complete", "epochs": epochs,
              "best_valid_video_macro_f1": best, "elapsed_s": time.perf_counter() - started,
              "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
              "test_evaluated": False, "note": "验证用于选择最佳模型，不是最终测试成绩"}
    write_json(output / "result.json", result)
    print("ACTION_TRAIN_SMOKE_OK" if args.smoke else "ACTION_TRAIN_DONE", flush=True)
    print(f"结果：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
