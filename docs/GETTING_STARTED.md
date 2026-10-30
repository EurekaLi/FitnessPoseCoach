# 使用与开发

## 可用模式

1. 阅读源码/CPU合成测试：无需数据、GPU、权重或相机，按README运行两组unittest。
2. 恢复原产物运行：须先满足[完整恢复条件](DATA_AND_MODELS.md)，没有公开托管权重。
3. 重新训练：自行合法获取数据，适配参考路径/ID，完成训练、冻结及导出，不直接沿用原成绩。

## 环境

原实验为WSL Ubuntu24.04、Python3.11.17、PyTorch2.14.1+cu132、torchvision0.29.1+cu132、NumPy2.4.6、ORT GPU1.30.0、RTX5070。Windows用Python3.13.5和deployment/windows的固定依赖；这是已验证组合，不保证任意平台即装即用。

requirements-training.txt不替你选择GPU版PyTorch。先为自己的硬件安装兼容PyTorch，再装辅助依赖；rtmlib单独用`python -m pip install --no-deps rtmlib==0.0.16`，避免拉入CPU ORT及重复cv2发行包。所需numpy/cv2/tqdm及GPU ORT接口由显式依赖提供。

测试环境只装opencv-python-headless，摄像头环境只装opencv-python，不要混装。Windows setup.ps1的-BasePython接受自己的Python路径，不改全局包或WSL。导出依赖与约束见deployment/wsl两个文件。

## 摄像头（完整产物已恢复）

在项目目录的Windows PowerShell运行：

```powershell
.\deployment\windows\setup.ps1 -BasePython python -SkipTensorRT
& '.\deployment\windows\.venv\Scripts\python.exe' -X utf8 deployment/tools/live_camera.py
```

环境已配置好则跳过第一行。不必激活Conda。若ps1受执行策略限制，可自行用Python创建deployment/windows/.venv，按requirements-runtime.txt安装并用--no-deps安装rtmlib；不用修改全局策略。

默认索引0/DSHOW，640×480；可传--camera-index 1或--backend MSMF。不自动改驱动或隐私设置。先关闭占用相机的其他应用，单人全身入镜，等CAMERA_APP_READY及约2.4秒历史积累/动作确认。窗口有焦点时Q/Esc退出，R清空历史、跟踪和次数。

missing表示输入不可用，uncertain表示未达确认门槛，stale_pose表示骨架过旧。默认不存影像或逐帧骨架，只在outputs/camera_runs/新目录写配置与统计。CAMERA_APP_OK表示正常运行/退出，不代表计数准确。软件显示延迟不含相机内部曝光/缓冲，离线吞吐不是相机FPS。

## 排错与复现

有原产物可用verify_online.py重放三段验证缓存，无需每次重复训练/独立测试。公开验证视频的短连接检查（不会开相机）：

```powershell
& '.\deployment\windows\.venv\Scripts\python.exe' -X utf8 deployment/tools/live_camera.py `
  --video data/raw/repcount_a/LLSP/video/valid/stu10_67.mp4 --headless --max-frames 80
```

相机打不开先查设备占用、索引、接口和连接；一直missing先查全身可见性、光照与处理率，不直接降门槛。部分ORT形状算子可在CPU运行，黄色提示不等于CUDA整个失效，也不证明整图都在GPU。

校验失败应恢复原文件，不绕过校验。尚无真实TensorRT模型运行入口，check_devices.py微型引擎检查不能当作完整加速。
