# FitnessPoseCoach

基于 **YOLOX + RTMPose-M + ST-GCN++** 的单人健身动作识别与计数项目，包含PyTorch训练、来源隔离评估、ONNX导出和Windows摄像头ONNX Runtime CUDA入口。

> 本仓库只发布源码、配置与说明，不包含数据集、训练权重、冻结快照或虚拟环境。克隆后可直接运行合成测试；摄像头需要先恢复完整产物，不是仅装依赖就能启动。

## 方法与实现

```text
摄像头 → YOLOX检测/几何跟踪 → RTMPose-M COCO17骨架
       → 髋部中心化/躯干尺度归一化 → 因果历史窗口
       → ST-GCN++ → 概率平滑/动作确认 → 周期状态机计数
```

- 动作：深蹲、俯卧撑、开合跳。图卷积GCN与时序卷积TCN用于骨架建模；ST-GCN++参考PYSKL纯PyTorch适配，不是全新算法或官方权重等价实现。
- 模型输入`(1,3,48,17)`，通道为归一化x/y和置信度，不是RGB图像；仅用当前及过去数据。
- 在线逻辑：EMA平滑、置信度门控、连续确认、周期计数、同目标短缺失恢复；不补算预热动作。
- 部署：ONNX Runtime GPU、OpenCV、多线程采集、最新帧有界缓冲、实际采集时钟；Q退出/R重置，默认不录制。

## 已完成与边界

2026-10-17来源隔离三类子集测试：61段视频，整段分类100%、窗口分类96.97%、ready覆盖80.74%、计数MAE **9.54次/视频**。整段分类正确不等于实时每个时刻正确，计数仍是原型，分项及验证边界见[实验结果](docs/RESULTS.md)。

真实ONNX已导出，122个验证窗CPU/Windows CUDA类别对齐，三段完整Windows GPU骨架回放保持状态/计数一致。摄像头入口已实现，公开验证视频短连接检查通过；未据此宣称现场长期稳定性或摄像头FPS。

尚未完成真实模型TensorRT FP16加速与经质量标签验证的动作纠错。TensorRT仅有环境/微型引擎检查，不作为完整产品功能宣传。

## 快速开始：CPU测试

建议独立Python3.11环境，无需数据、权重、相机或GPU：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m unittest discover -s deployment/tests -v
```

测试依赖只装在独立测试环境，不与摄像头环境叠装多个cv2发行包。CI使用同样的合成检查。

## 摄像头运行

先按[数据与模型](docs/DATA_AND_MODELS.md)恢复完整产物，再在Windows项目目录执行：

```powershell
.\deployment\windows\setup.ps1 -BasePython python -SkipTensorRT
& '.\deployment\windows\.venv\Scripts\python.exe' -X utf8 deployment/tools/live_camera.py
```

本机已经装好依赖时跳过安装。当前ORT基线不需要TensorRT；参数、训练/导出顺序和排错见[使用说明](docs/GETTING_STARTED.md)。

## 结构

```text
src/fitness_pose/     模型、归一化、分类过滤与计数
scripts/             数据准备、训练、独立测试和回放
deployment/tools/    导出、验证和摄像头入口
deployment/windows/  Windows依赖/配置/启动脚本
deployment/wsl/      导出依赖及约束
configs/             计数实验配置，v6为当前冻结规则
tests/               合成契约测试
docs/                公开使用、结果、产物及发布说明
third_party/         PYSKL来源/修改说明和原许可
tools/               上传前候选文件检查
```

## 发布和许可

执行`python tools/check_repository.py`检查Git候选，不删除数据。上传步骤见[发布说明](docs/PUBLISHING.md)。

本次没有代作者选择顶层开源许可证。PYSKL适配部分保留Apache-2.0原许可及来源说明；数据与预训练模型不随仓库分发，见[许可边界](docs/LICENSING.md)。
