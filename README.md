# Climber_legged_rl

> **2026-10-27：当前使用真实CAD模型路线。** scripts/train.py默认核验原始CAD/MJCF；动力学和运动未核实前不启动PPO。详见 [真实模型训练](docs/真实模型训练.md)。零位、方向、限幅独立记录在 [实机必测清单](docs/实机必测清单.md) 与 configs/robot/hardware_calibration.yaml。下文近似试训仅为历史实验。

Climber_legged_rl 是轮腿机器人的强化学习训练仓库。现有一个可运行的 **MuJoCo＋PPO近似仿真站立实验**，用于验证闭链、控制、训练与评测流程；Isaac Lab任务和实机模型标定尚未完成。近似实验不代表实机可部署。

## 目标技术栈

- 仿真与任务：Isaac Sim + Isaac Lab
- 向量化 PPO：RSL-RL
- 后续验证：MuJoCo sim2sim
- 策略导出与实机部署：STM32 / STM32Cube.AI，计划使用小型 float32 actor 经 ONNX 转换

版本需与安装好的 Isaac Lab 匹配。请先不要在本仓库内固定 Isaac Sim、Isaac Lab、PyTorch 或 CUDA 版本；收到你的机器环境后再写入锁定文件。

## 目录

```text
Climber_legged_rl/
├── assets/robot/                 # 已导入旧候选USD；与新MJCF尚未核验同步
├── configs/
│   ├── robot/robot.yaml          # 机器人和执行器参数占位
│   ├── env/flat.yaml             # 首个平地任务参数占位
│   └── agent/ppo.yaml            # PPO 参数占位
├── docs/                         # 建模、训练、sim2sim、部署说明
├── scripts/                      # MuJoCo train/play/export；Isaac Lab仍待实现
├── source/climber_legged_rl/     # Python 包和 Isaac Lab 任务
├── sim2sim/                      # 原始MJCF、网格和历史导出
├── deployment/                   # 实机策略接口，待控制链路信息
└── tests/                        # 模型、观测动作和策略导出一致性检查
```

当前已完成真实CAD气弹簧伸缩机构和左侧闭合锚点的独立修正候选，24点及101帧运动核验通过；见 [CAD修正结果](docs/CAD修正结果.md)。逐刚体物性和气弹簧曲线仍缺失，真实动力学训练尚未开始。

## 建设顺序

1. 模型运动核验：原始CAD关节、闭链、装配分支和真实站姿。
2. 动力学修正和站立训练：逐刚体惯量、气弹簧、接触和六通道驱动。
3. 板端输出一致性验证：观测、策略、控制环和限幅。
4. 架空限力测试：实测零位、方向、反馈比例和停机。
5. 带保护的落地测试：先原地站立，再扩展运动任务。

## 开始训练

完整Isaac Lab路线仍需实现与验证：`assets/robot/climber.usdc`为旧候选资产，`sim2sim/climber_mjcf/robot.xml`为保留的原始导出，两者不是已核实同步模型。

首轮独立实验见 [MuJoCo首轮试训](docs/MuJoCo首轮试训.md)：安装 `requirements-mujoco.txt`，先运行模型生成和smoke检查，再显式传入 `--allow-approximate-model` 训练。独立配置为 `configs/robot/mujoco_experiment.yaml`，生成模型位于 `outputs/`。质量分配、碰撞、气弹簧缺失、临时限幅和较低reset姿态均已标注；不要把试训策略或临时值直接用于实机。

## 实验记录要求

每次训练保留配置快照、随机种子、代码提交号、训练日志、评测命令集和 checkpoint。大体积 checkpoint 不提交到 Git；发布时记录模型对应的配置和观测/动作接口版本。

## 待补项

详见 [`docs/待补信息.md`](docs/待补信息.md)。

已选参数及开源依据见 [`docs/参考参数与STM32部署.md`](docs/参考参数与STM32部署.md)。其中试用控制增益和频率尚未通过物理仿真或板端验证。
