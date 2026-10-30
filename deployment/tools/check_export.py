"""只用微型CPU网络确认ONNX导出环境，不读取真实权重或测试视频。"""

from importlib import metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    destination = PROJECT_ROOT / "outputs/deployment_environment"
    destination.mkdir(parents=True, exist_ok=True)
    model = torch.nn.Linear(4, 3).eval()
    with torch.no_grad():
        model.weight.fill_(.5)
        model.bias.fill_(.25)
    example = torch.ones(1, 4)
    # 小网络在CPU导出/对齐，不与正在处理测试骨架的GPU任务争用计算资源。
    with tempfile.TemporaryDirectory(prefix="export_check_", dir=destination) as directory:
        path = Path(directory) / "toy.onnx"
        torch.onnx.export(model, (example,), str(path), input_names=["x"], output_names=["y"],
                          opset_version=18, dynamo=True, external_data=False)
        onnx.checker.check_model(onnx.load(str(path)))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = session.run(None, {"x": example.numpy()})[0]
        with torch.inference_mode():
            reference = model(example).numpy()
        np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-5)
    report = {"status": "ok", "executable": sys.executable, "python": sys.version,
              "packages": {dist.metadata["Name"]: dist.version for dist in metadata.distributions()},
              "toy_export": {"opset": 18, "dynamo": True, "cpu_output": output.tolist(),
                             "max_absolute_error": float(np.abs(output - reference).max())},
              "actual_action_model_exported": False, "test_dataset_used": False}
    (destination / "wsl_export.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"EXPORT_ENV_OK python={sys.executable} onnxscript={metadata.version('onnxscript')} error={report['toy_export']['max_absolute_error']}")


if __name__ == "__main__":
    main()
