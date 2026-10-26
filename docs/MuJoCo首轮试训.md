# MuJoCo 首轮近似试训

> 历史实验：2026-10-27用户要求真实模型，当前路线见 [真实模型训练](真实模型训练.md)。本页近似模型和checkpoint不用于当前训练；复现须显式指定 --backend mujoco。

## 适用范围

这是原始 MJCF 的独立近似仿真实验，不是 Isaac Lab 已经完成，也不是实机可用策略。用户于2026-10-26同意继续推进后，先用 MuJoCo CPU物理＋Stable-Baselines3 PPO验证模型、训练、评测和checkpoint链路。小型MLP默认在CPU更新；不声称是GPU并行物理训练。服务器的4090可用于渲染，PyTorch CUDA已验证可用。

原始 `sim2sim/climber_mjcf/robot.xml`、网格和旧USD均保留。生成文件写入被Git忽略的 `outputs/mujoco_experiment/`，不污染原始CAD导出。生成XML引用当前机器的绝对mesh路径，换机器后应重新运行生成器。

## 本次实际结果

| 阶段 | 实际累计交互步数 | 评测 |
| --- | ---: | --- |
| 零动作对照 | 0 | 平均0.472秒触发终止 |
| 首轮PPO | 100352 | 10回合平均0.626秒 |
| 续训后最终策略 | 1101824 | 独立20回合均达到5秒上限 |
| 同一最终策略长测 | 不新增训练步数 | 独立5回合均达到30秒上限 |

长测未触发数值警告，最大闭链端点误差约0.069mm。不过30秒末端平面位移约1.03～1.05m，前向速度RMSE约0.036m/s：**这说明获得了不倒基线，不是严格原地站立**。还没有非零速度跟踪、抗大扰动、粗糙地形或实机测试。最终actor的TorchScript导出与PPO确定性输出的随机/轨迹对比误差均为0。

后续优先修正轮胎碰撞和模型动力学，收紧零速任务的速度/位置漂移评测；不要仅凭奖励上涨或timeout率就宣称完成站立。

## 明确的近似项

- 整机15kg、每侧5.14kg、每轮0.52kg沿用已提供数据；将每侧非轮刚体质量和惯量等比例缩放到该侧剩余4.62kg，保留局部COM。中央刚体设为差额4.72kg。这是实验质量分配方法，不代表已核实base就是4.72kg。左右缩放因子分别约7.304和5.148，说明源模型的分项质量仍需机械核对。
- 原始薄壳mesh指定shell惯性算法后可编译；所有CAD mesh仅作视觉。碰撞暂用两个半径55mm的球体和机身盒体，不模拟连杆/链条碰撞、真实轮胎宽度或形变。
- 保留六个闭链约束；新增自由机身，只驱动确认的四个腿关节和两个车轮，不给被动关节额外电机。
- 气弹簧力暂未建模；450N不被误写成N/m刚度。
- 临时腿输出力矩上限10Nm、轮输出上限2Nm；这些是试训参数，不是额定/峰值硬件声明。速度保护和关节范围也仅为实验边界。
- 有界数值求解没有找到满足135mm髋轴到轮轴高度及当前导出范围的初值，不能据此证明实机不可达。首轮改用110mm目标；求得左约108.8mm、右110mm。不是让用户重测站姿，也不是编码器零位标定。
- 根节点绕z转pi，使模型命名的left落在世界+y侧；机体-source-x暂作为前方。此坐标约定需要后续与CAD/IMU核验。
- 第一轮命令恒为零，只学习5秒平地站立，尚未训练前进、转向、台阶、跳跃。重置只有小角度与小速度扰动，不代表大扰动鲁棒性。

完整配置见 `configs/robot/mujoco_experiment.yaml`。原始 `robot.yaml` 和Isaac Lab计划不被这些实验值覆盖。

## 安装和检查

在Python3.10隔离环境安装 `requirements-mujoco.txt`，不要覆盖Isaac Lab自己的依赖。服务器环境路径为 `/root/gpufree-data/.venvs/climber-validation/bin/python`。

```bash
cd /root/gpufree-data/Climber_legged_rl
source /root/gpufree-data/.venvs/climber-validation/bin/activate
python scripts/prepare_mujoco_experiment.py
python scripts/smoke_mujoco.py --steps 1000
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/check_training_readiness.py --backend mujoco
```

smoke验证Gym接口、观测/动作、接触、重置、力矩限幅与有限数值；它不要求随机策略已经站稳。报告绑定config、生成模型和环境代码的SHA256。改这些文件后需重跑smoke，不能沿用旧检查结果。

`check_training_readiness.py` 默认仍检查原Isaac Lab路线，可能返回阻塞；应明确区分 `--backend mujoco` 的近似实验与完整目标路线。

## 训练与恢复

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u scripts/train.py \
  --backend mujoco --allow-approximate-model --device cpu \
  --num-envs 8 --total-timesteps 100000 --run-name example_standing
```

默认拒绝未显式允许近似模型的训练。所有run名必须唯一，防止覆盖。8个进程各跑MuJoCo，策略100Hz、物理500Hz；PPO按完整rollout取样，实际步数可能略大于请求数。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u scripts/train.py \
  --backend mujoco --allow-approximate-model --num-envs 8 --total-timesteps 1000000 \
  --run-name example_continuation \
  --resume outputs/mujoco_runs/example_standing/final_policy.zip
```

恢复时 `--total-timesteps` 指新增步数。生成模型以及mass/physics/control/task/reward配置必须相同，不把改变动力学当作无缝断点恢复。

每个run有config、模型、来源代码hash、metadata、CSV/TensorBoard、monitor日志、每1万步checkpoint和最终策略；续训还每5万步评测并保存best。评测失败/数值发散会记录failed状态，不静默继续。训练进程不是调度器，不会在达到预算后自动无限续训。

## 评测、回放与导出

```bash
MUJOCO_GL=egl python scripts/play.py \
  --checkpoint outputs/mujoco_runs/example_standing/final_policy.zip \
  --episodes 20 --seed 40000 \
  --report outputs/mujoco_runs/example_standing/heldout.json \
  --video outputs/mujoco_runs/example_standing/playback.mp4

python scripts/export.py \
  --checkpoint outputs/mujoco_runs/example_standing/final_policy.zip \
  --output outputs/mujoco_runs/example_standing/export
```

回放默认隐藏CAD网格、显示碰撞简化体；不能把球形碰撞视为实际轮胎模型。20回合评测使用独立种子，但扰动分布仍与训练相同；timeout表示撑过5秒、不触发当前终止条件，不等同于长期静止或实机成功。

TorchScript导出含24维输入/6维动作顺序、缩放、数值reset角及模型hash，并对随机观测和真实轨迹检查输出一致性。尚未完成ONNX/CubeAI验证，禁止直接部署实机。
