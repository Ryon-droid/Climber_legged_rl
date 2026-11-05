# GPU 物理与速度估计器

**最新训练进度：**速度估计第十三轮完成 245760 步，best 仍为起始 actor；新种子无扰站姿/站姿安排推扰各 19/20，固定起身/起身后受推各 20/20，旧难例仍失败。第十四轮降低 PPO 学习率、增加成功轨迹回放，正式尚未启动。见 [第十三轮评估与第十四轮训练](速度估计第十三轮评估与第十四轮训练.md)。

**后续训练已完成：**速度估计第一轮 best 已在独立复测中完成起身约 2.09 s，无推扰起身/站姿各 20/20；final 的起身后偏航仍超标。下一轮采用 best，命令和完整评估见 [速度估计第一轮评估与第二轮训练](速度估计第一轮评估与第二轮训练.md)。下文记录实现时的初始验证结果。

已接入 MuJoCo Warp CUDA 物理、有监督速度估计器，以及对应的训练、CPU 独立评估、MuJoCo 回放、checkpoint 继续训练和 TorchScript 导出。**目前推荐继续使用 CPU 物理。GPU 实测尚未提速；速度估计器候选仍需重新学习起身，不能替换已验收的联合策略。**

## 实测结果

现有 CAD XML SHA256 为 `796095694e5a8eaa73257f808779d1cd6b3322bbd998bb1e991522eccde24823`，本次沿用它的 16.4634566 kg 质量、六个闭链、450 N 气弹簧、155–220 mm 销轴中心距及电机限幅。物理周期 1 ms，控制周期 10 ms。

估计器使用旧策略生成的 16 个完整回合：24000 帧训练、6144 帧验证，按完整回合划分。验证集用于选择预训练 epoch，另用新种子检查闭环。验证速度 RMSE 为 **0.00562 m/s**，分轴 `[0.00588, 0.00385, 0.00673]` m/s；始终预测零速度的基线为 **0.02131 m/s**。这些数值只说明教师轨迹上的精度。

CPU 闭环检查使用 seed 112000、112001，无推扰，两组 20 s 站姿均成功，趴姿 **0/2**，在 10 s 起身期限终止。1024 步短程 PPO 后仍为上述结果。换成估计速度会改变旧策略反馈，冻结原起身分支不能保证保留效果。因此三个新配置均设置 `training.freeze_rising_actor: false`，允许重新学习起身；原有四个历史观测对照仍默认冻结起身分支。

GPU 的 0.2 s 站姿/趴姿开环响应与 CPU 相比，qpos 最大差异约 `1.1e-5`、qvel 约 `5.3e-4`、电机力矩约 `0.0013 N·m`，闭链误差小于 0.6 mm。禁用 CPU `mj_step` 后仍能积分，双世界不同动作和独立 reset 检查也通过。旧 CPU 策略在 GPU 上的单种子 20 s 站稳通过，但起身在 10 s 超时，不能认为两个后端的长轨迹等价。

4090 吞吐量如下，包含主机任务计算，排除首次编译，不包含 PPO 更新/独立评估；会随机器负载变化：

| 后端 | 环境数 | 每秒控制转移数 |
| --- | ---: | ---: |
| CPU 单环境 | 1 | 578 |
| CPU 多进程 | 8 | 2846 |
| MuJoCo Warp CUDA | 1 | 218 |
| MuJoCo Warp CUDA | 8 | 582 |
| MuJoCo Warp CUDA | 16 | 691 |

CUDA 执行碰撞、约束求解、积分，以及每个物理子步的 PD/电机限幅。奖励、起身阶段、观测、reset 和控制步末的 `mj_forward` 仍复用 CPU。主机线程、状态复制和任务逻辑限制吞吐量，因此这条路径目前不能加快正式训练。扩大到数百/数千环境，需要进一步把任务计算搬到 GPU。当前开放 GPU 1–64 环境，CPU 1–14 环境。

完整报告：

- `outputs/cad_balance_estimator/pretrained/pretrain_report.json`
- `outputs/cad_balance_estimator/validation/summary.json`
- `outputs/cad_balance_gpu/gpu_validation.json`
- `outputs/cad_balance_gpu_estimator/validation/summary.json`

本次只运行了监督预训练和 CPU 1024 / GPU 512 步 PPO 集成验证，没有启动新的正式长训练，未中断先前的历史观测训练。

## 估计器与接口

五帧历史，每帧 36 维，从旧到新，相邻 10 ms、首尾跨度 40 ms。估计器每帧只取角速度/重力投影、四个主动关节位置、六个电机速度和上一指令，共 22×5=110 维。网络 `110 → 128 ELU → 64 ELU → 3`，输出机身坐标系三维速度，单位 m/s。

训练环境是 183 维：180 维历史＋3 维真实速度标签。每帧 28:30 的真实平面速度置零；actor 内再次屏蔽这些槽位，仅在最新帧填入估计 vx/vy。真实速度仅用于 critic 和监督损失，改变标签不能改变动作。PPO 使用停止梯度后的估计值；独立 Adam 使用 MSE，默认每轮 rollout 更新 4 epoch、batch 512、学习率 0.001。

普通 `PPO.load` 可加载 checkpoint 推理；继续训练使用项目 runner，它选择 `VelocityEstimationPPO` 并恢复第二个优化器。导出 actor 只接收 **180 维**，不需要真实标签；另有 `velocity_estimator.ts`，180 维输入、3 维 m/s 输出。随机输入及真实轨迹的导出误差均为零。

actor 仍使用原有的相对起点位移、航向、髋部高度和阶段状态。估计器不使用这些量；速度估计没有自动解决它们在实机上的获取问题。

## 命令

独立 GPU 环境已安装：`/root/gpufree-data/.venvs/climber-gpu`。原验证环境未替换。新环境安装 `requirements-gpu.txt`，版本锁定 MuJoCo/MuJoCo Warp 3.17.0、Warp 1.18.0，需要 NVIDIA CUDA 驱动。

CPU 物理＋估计器的候选训练，默认使用本次预训练 checkpoint，重新适应反馈：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python -u scripts/train_balance_history.py --variant estimator
```

重新采集并预训练时选新目录；可用 `--parent PATH` 指定匹配的 36/180 维教师，用 `--dataset PATH/teacher_velocity.npz` 复用校验过教师和模型来源的数据：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/prepare_velocity_estimator.py \
  --episodes 16 --epochs 80 --output outputs/cad_balance_estimator/pretrained_new
```

使用新预训练结果训练时，增加 `--parent outputs/cad_balance_estimator/pretrained_new/pretrained_policy.zip`。

独立验证：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/validate_velocity_estimator.py
/root/gpufree-data/.venvs/climber-gpu/bin/python scripts/validate_gpu_physics.py
/root/gpufree-data/.venvs/climber-gpu/bin/python scripts/validate_velocity_estimator.py \
  --config configs/robot/cad_balance_gpu_estimator.yaml --steps 512
```

GPU 实验训练选项：

```bash
# GPU 物理＋历史策略
/root/gpufree-data/.venvs/climber-gpu/bin/python -u scripts/train_balance_history.py --variant gpu

# GPU 物理＋速度估计器
/root/gpufree-data/.venvs/climber-gpu/bin/python -u scripts/train_balance_history.py --variant gpu_estimator
```

GPU runner 要求与当前模型、后端源码匹配的 `gpu_validation.json`；缺失/过期会提示先验证。CPU/GPU 之间切换使用 warm start，不能当作相同后端 resume。独立选模评估仍在 CPU MuJoCo 上完成，并注明后端。

查看预训练候选的站姿：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/play_getup.py \
  --config configs/robot/cad_balance_estimator.yaml \
  --checkpoint outputs/cad_balance_estimator/pretrained/pretrained_policy.zip \
  --reset-mode standing --push-episode-probability 0 --seed 112000 --loop
```

导出：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/export.py \
  --checkpoint outputs/cad_balance_estimator/pretrained/pretrained_policy.zip \
  --output outputs/cad_balance_estimator/pretrained/export
```

GPU 时间在每个控制步上传时按整数步数重定位，随后执行十次实际积分并检查返回时间，容差随 float32 精度缩放。发散、时间重置、接触/约束容量溢出均会终止。摩擦参数按世界独立上传，电机饱和和超额定时长仍按每个物理子步统计。
