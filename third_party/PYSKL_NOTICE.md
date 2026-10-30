# PYSKL 代码来源与适配说明

上游：[kennymckormick/pyskl](https://github.com/kennymckormick/pyskl)，OpenMMLab PYSKL 项目。

Copyright 2018-2019 Open-MMLab. All rights reserved.

许可：Apache License 2.0，全文见本目录 `PYSKL_LICENSE.txt`。

本项目于 2026-10-16 依据下列上游文件编写纯 PyTorch 适配模块 `src/fitness_pose/stgcnpp.py`：

- `pyskl/models/gcns/stgcn.py`：十阶段网络、内外残差、5/8 阶段降采样和通道增长。
- `pyskl/models/gcns/utils/gcn.py` 中 `unit_gcn`：`adaptive='init'`、`conv_pos='pre'`、`with_res=True`。
- `pyskl/models/gcns/utils/tcn.py` 中 `unit_tcn`、`mstcn`：四种膨胀率、池化和 1×1 共六分支。
- `pyskl/utils/graph.py`：COCO17 `spatial` 图和列归一化。
- `pyskl/models/heads/simple_head.py`：关节/时间平均池化后线性分类。
- `configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py`：ST-GCN++ 架构配置。

这是修改后的适配文件，不是原封不动的上游代码，也不是新算法。当前参考上游 main 文件；实际交付源文件的 SHA256 写入每次训练配置以固定本地实现。未固定上游提交，也未宣称逐参数数值等价。

适配差异：单人输入 `(B,3,48,17)`、髋部/躯干归一化、三分类、时间分支 dropout=0.1（首层为0）、从头训练、按视频与类别平衡抽样、SGD 初始学习率0.02、梯度裁剪5、按轮 cosine 调度。去掉注册器、MMCV 构建器和预训练加载，使用 PyTorch 原生层；输入 BN 保持上游的关节优先排列。

验证范围：已检查模块结构与上游设计对应关系；实际前反向行为由 `train_action.py --smoke` 在用户的 WSL 环境验证。没有声称复现 NTU 数据集分数或完成官方权重数值对齐。

论文：Duan et al., *PYSKL: Towards Good Practices for Skeleton Action Recognition*, ACM Multimedia 2022。
