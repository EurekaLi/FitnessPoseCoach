# 数据与模型产物

本仓库只发布源码。数据、标注、姿态缓存、索引、检查点、ONNX、TensorRT引擎、冻结快照、环境与原始报告均默认忽略，不通过Git/LFS分发，现有本地文件保留。没有已托管的模型下载地址。

## 获取来源

- RepCount：从[数据集页面](https://svip-lab.github.io/dataset/RepCount_dataset.html)及[官方TransRAC仓库](https://github.com/SvipRepetitionCounting/TransRAC)按提供方要求获取。当前布局为data/raw/repcount_a/LLSP/，含annotation、video、original_data/filename_mapping.xlsx；不同下载布局须先适配，不能假定标签通用。
- Fitness-AQA：[原申请入口](https://github.com/ParitoshParmar/Fitness-AQA)。本项目未用其质量标签，不宣称已完成监督纠错。
- YOLOX-tiny/RTMPose-M：rtmlib的COCO17 ONNX方案；下载入口和指纹流程在scripts/pose_image.py，缓存到models/pretrained下两个子目录。使用前核对模型和相关训练数据条件。
- ST-GCN++：本项目训练的三分类检查点，不能随意替换成上游通用权重；结构适配来源见third_party/PYSKL_NOTICE.md。

## 恢复原摄像头产物

当前live_camera.py需要：

```text
releases/v6_test_v1/release.json
releases/v6_test_v1/release.sha256
releases/v6_test_v1/snapshot/<清单中的每个文件>
outputs/action_onnx_v1/action_stgcnpp.onnx
outputs/action_onnx_v1/export_report.json
```

verify_release还检查清单中的工作区原文件，包括来源索引、参考缓存摘要、源码及模型。因此只复制ONNX，甚至只复制release目录，都不足以恢复当前入口。须从自己的可信备份恢复清单内全部文件，或合法获取数据后重新准备、训练、冻结和导出。不要删改校验字段来绕过错误。

参考指纹（不是下载地址）：

- best.pt：4ed8f408227935df7efe5026ff074d8598c13502782ee542ac689ab1cf37f0c9
- action_stgcnpp.onnx：337225a3233e904303c5c8a7a51bebbeca03c530549a9a7060c9235909a873e7

## 新实验脚本顺序

prepare_repcount → pose_video（train/valid）→ prepare_action_windows → prepare_source_split → train_action → independent_test freeze/prepare/evaluate → action_onnx export/verify → verify_online → live_camera。

训练/验证/测试按原视频来源隔离，不随机拆相邻窗口；参考训练索引用training_split而非文件夹。冻结后测试仅作报告，不用来挑阈值。

这些脚本仍包含本实验默认路径与参考ID。independent_test.py的候选/参考缓存及18/19/24测试数量约束、action_onnx.py的三段样本ID不是任意数据集的通用配置。新实验需显式适配并使用新版本，不保证一条命令复现原成绩。不要修改现有本地冻结候选。
