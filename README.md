# Climber_legged_rl

Climber_legged_rl 是轮腿机器人的强化学习训练仓库骨架。已导入候选模型，并参考开源仓库制定了首轮控制、观测和 PPO 配置；关节映射、模型物理验证与 Isaac Lab 任务实现尚未完成，因此现在不能启动训练。

## 目标技术栈

- 仿真与任务：Isaac Sim + Isaac Lab
- 向量化 PPO：RSL-RL
- 后续验证：MuJoCo sim2sim
- 策略导出与实机部署：STM32 / STM32Cube.AI，计划使用小型 float32 actor 经 ONNX 转换

版本需与安装好的 Isaac Lab 匹配。请先不要在本仓库内固定 Isaac Sim、Isaac Lab、PyTorch 或 CUDA 版本；收到你的机器环境后再写入锁定文件。

## 目录

```text
Climber_legged_rl/
├── assets/robot/                 # 自有机器人 USD/URDF/网格模型，当前待提供
├── configs/
│   ├── robot/robot.yaml          # 机器人和执行器参数占位
│   ├── env/flat.yaml             # 首个平地任务参数占位
│   └── agent/ppo.yaml            # PPO 参数占位
├── docs/                         # 建模、训练、sim2sim、部署说明
├── scripts/                      # train/play/export 入口，任务定义后补
├── source/climber_legged_rl/     # Python 包和 Isaac Lab 任务
├── sim2sim/                      # MuJoCo 模型与策略回放，待机器人模型
├── deployment/                   # 实机策略接口，待控制链路信息
└── tests/                        # 模型、观测动作和策略导出一致性检查
```

## 建设顺序

1. 补全 `configs/robot/robot.yaml`，提供机器人模型和关节/执行器信息。
2. 在 Isaac Lab 中加载模型，检查质量惯量、关节方向/限位、接触和闭链稳定性。
3. 定义单一平地速度跟踪任务，先用 PPO 训练站立和低速移动。
4. 用实测数据校准电机响应，再逐步加入延迟、动力学随机化和地形课程。
5. 将策略导出到 MuJoCo 做 sim2sim；输入顺序、归一化、动作缩放与控制频率逐项对齐后，再接实机。

## 开始训练

目前已有一套待验证的 Climber 模型：`assets/robot/climber.usdc` 用于 Isaac Lab 候选资产，`sim2sim/climber_mjcf/robot.xml` 用于 MuJoCo 候选模型。它们还没有通过加载和物理一致性检查，因此训练入口仍不会启动。不要直接把其他队伍的机器人配置当作 Climber 的参数。

## 实验记录要求

每次训练保留配置快照、随机种子、代码提交号、训练日志、评测命令集和 checkpoint。大体积 checkpoint 不提交到 Git；发布时记录模型对应的配置和观测/动作接口版本。

## 待补项

详见 [`docs/待补信息.md`](docs/待补信息.md)。

已选参数及开源依据见 [`docs/参考参数与STM32部署.md`](docs/参考参数与STM32部署.md)。其中试用控制增益和频率尚未通过物理仿真或板端验证。
