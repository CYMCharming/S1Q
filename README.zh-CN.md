# S1Q：面向 System One 决策模型的低比特量化

[English](README.md) · [完整结果](docs/results.md) · [方法说明](docs/method.md) · [评测设计](docs/evaluation-design.md) · [权重使用](docs/artifacts.md) · [运行环境](docs/runtime.md) · [相关工作](docs/related-work.md)

S1Q 面向 Kev-0.8B、Kev-4B、Kev-9B、NanoJev 和 Laya 等开源 Jev-like 决策模型，研究低比特量化如何影响选择、评分和概率输出。

方法结合激活感知通道缩放、分组裁剪搜索，以及可选的决策输出 Fisher 权重。只用独立开发集选择量化配置，温度校准与量化校准分开，并保留原生决策头和混合架构的递归状态计算精度。原精度、RTN、局部重构消融及选定 S1Q 的逐项预测和指标保存在 `results/`。

目前实现包括 W4/W8 权重量化、A4/A8 模拟量化、INT4/INT8 打包文件，以及逐层临时反量化的低存储参考执行路径。**浮点矩阵乘法仍然存在，不能把这些实验描述为原生 W4A4 内核加速。** 总体压缩率会计入保留的 embedding、决策头和其他参数。

五个模型已经在 A100/A800 上完成真实量化与评测。下表为主测试集准确率（%）；RTN 与 S1Q 的权重范围、分组和激活精度匹配。Kev/Laya 使用 914 个决策，NanoJev 使用 1,023 个 shooting 决策，标签为记录的参考策略动作概率 argmax；不能将不同数据集的准确率当作模型排行榜。

| 模型 | 选定配置 | 原精度 | 匹配 RTN | S1Q | 完整参数存储占原模型 |
|---|---|---:|---:|---:|---:|
| Kev-0.8B | W4，决策 Fisher 加权 | 82.93 | 80.63 | 81.18 | 51.6% |
| Kev-4B | W4，激活感知 | 85.45 | 84.79 | 84.79 | 37.8% |
| Kev-9B | W4A8 模拟 | 86.98 | 84.35 | 86.54 | 36.0% |
| NanoJev | W4，激活感知 | 79.86 | 79.47 | 80.45 | 36.0% |
| Laya | W4，激活感知 | 66.85 | 63.68 | 64.55 | 29.4% |

开发集不支持直接采用 W4A4。S1Q 也没有在所有数据上一致优于 RTN：Kev-0.8B 的迁移集比 RTN 低 2.38 个百分点，多项配对置信区间包含零。[完整结果](docs/results.md)保留了概率指标、负面结果、消融、独立外部测试和显存测量。主数据集在首轮结果用于改进方法后按探索性结果报告，外部 JevBench 使用事先冻结的配置。

现有社区已经有 Laya INT4/INT8 和 Kev INT8 量化，因此本项目不宣称“首个针对这类模型的量化工作”，贡献定位为跨模型量化实现、决策输出适配及可复现研究。

NanoJev 上游数据包涵盖 Maze、Snake、ViZDoom Basic 和 Predict Position；S1Q 最终原生 test（1,023 个决策）和 OOD（1,024 个决策）仅包含 shooting 的 Basic 与 Predict Position。原数据显式提供的 `reference_argmax_compatibility` 标签用于衡量与记录的强化学习参考策略动作 argmax 的一致率，不是人类标注、最优动作真值或观测成功概率。没有显式受支持硬标签的题目被排除。跨游戏回合的重复输入先按连通分组处理，再用固定规则避免校准与测试重叠。

安装与执行命令见 [英文说明](README.md#quick-start)。源码、权重和数据均固定版本；公开仓库提供代码、可复现实验配方、数据标识和派生评测结果，不重新分发许可不明确的原始数据。

[v0.1.0 发布页](https://github.com/CYMCharming/S1Q/releases/tag/v0.1.0)提供三个 Kev 和 Laya 的已评测打包权重。NanoJev 已完成量化，但其模型卡没有单独明确微调权重许可，因此公开复现配方与结果，原权重从上游下载。打包文件仅包含部分线性权重，仍需相同版本的原生模型、分词器及决策头。[详细使用与许可说明](docs/artifacts.md)。

本项目由 [CYMCharming](https://github.com/CYMCharming) 维护，与 TypeSafe 无隶属关系。感谢 [Jev](https://docs.typesafe.ai/introduction)、[Kev](https://github.com/jaredpalmer/kev)、[NanoJev](https://github.com/TianyuCodings/NanoJev) 和 [Laya](https://github.com/NandhaKishorM/laya) 的原作者。
