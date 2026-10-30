"""只读检查Git候选，不遍历被忽略的数据，不删除或暂存文件。"""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DOCS = {"data/README.md", "models/README.md", "releases/README.md", "outputs/.gitkeep"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def main() -> int:
    try:
        # Windows工作区可能属于Administrators组；只对脚本所在仓库临时信任，不改全局配置。
        result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                 "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                                cwd=ROOT, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("需要Git仓库；请先运行git init -b main。", file=sys.stderr)
        return 1
    # cached也含已经误入索引、之后才被ignore的文件，不只检查新文件。
    names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    errors, total = [], 0
    for name in names:
        path, parts = ROOT / name, Path(name).parts
        if path.is_symlink():
            errors.append(f"{name}: 不提交本机符号链接")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        if size > 1024 * 1024:
            errors.append(f"{name}: 超过1MiB，请单独管理产物")
            continue
        if ((parts[0] in ("data", "models", "outputs", "releases") and name not in ALLOWED_DOCS)
                or any(part in (".venv", "venv", "__pycache__") for part in parts)
                or path.suffix.lower() in (".pt", ".pth", ".onnx", ".engine", ".plan", ".npz", ".npy", ".mp4", ".pem", ".key")
                or path.name in (".env", "credentials.json")
                or path.name.startswith(".env.") and path.name != ".env.example"):
            errors.append(f"{name}: 产物/环境/敏感路径不能提交")
        content = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"{name}: 疑似凭据；不回显具体内容")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"REPOSITORY_{'FAILED' if errors else 'OK'} files={len(names)} bytes={total} issues={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
