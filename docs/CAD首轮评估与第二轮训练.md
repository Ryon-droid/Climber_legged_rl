# CAD 首轮评估与第二轮训练

> **已失效的站姿假设（2026-10-28 后续修正）：**用户确认主体应水平。首轮及原第二轮使用了错误的主体参考系，本文倾角、320 mm 站姿及从旧 best warm-start 的命令仅保留为历史记录，不应继续执行。原第二轮已停止。修正结果与从零训练命令见 [主体水平姿态修正](主体水平姿态修正.md)。

评估日期：2026-10-28（北京时间）。首轮已完成；第二轮完整训练尚未启动。

## 首轮是否结束

- 运行：`outputs/mujoco_runs/cad_dynamics_standing_20261028_201934/`。
- 开始于 2026-10-28 20:19:34，完成于 20:28:25；`status.json` 为 `completed`，训练进程已退出。
- 实际采样 1,001,472 步；PPO 按完整 rollout 采样，所以略超过请求的 1,000,000 步。
- 当前任务是平地原地站立，前进/转向指令均为零；首轮单回合上限 5 秒。

## 独立评估

对 best 与 final 使用相同的留出种子 80000–80019，各评估 20 回合，每回合延长到 30 秒。这里的通过仅表示该模型、平地、小初始扰动下未触发跌倒等终止条件。

| 指标 | best（750,000 步） | final（1,001,472 步） |
|---|---:|---:|
| 站满 30 秒 | 20/20 | 20/20 |
| 平均回报（相同首轮奖励） | 7550.30 | 7469.07 |
| 回合结束时平面位移：均值 | 3.09 cm | 34.32 cm |
| 回合结束时平面位移：最大值 | 7.15 cm | 39.54 cm |
| 前进速度 RMSE：回合均值 | 0.0114 m/s | 0.0180 m/s |
| 双轮同时接地比例：回合均值 | 99.51% | 99.50% |
| 全部回合最大闭链误差 | 0.562 mm | 0.562 mm |
| MuJoCo 数值警告 | 0 | 0 |

结束位移不是全程最大偏移或累计路程。推荐从 `best/best_model.zip` 接着训练，因为 final 的静止漂移明显更大；训练步数增加并不保证策略持续改善。

原始报告：

- `outputs/cad_dynamics/best_heldout_30s_20ep.json`
- `outputs/cad_dynamics/final_heldout_30s_20ep.json`

## 站立质量仍有不足

额外对 best 使用种子 81000–81004、5 回合 × 30 秒，记录姿态与力矩：

- 平均机身倾角 **13.69°**，说明目前学会了持续不倒，站姿还不够直。
- 髋部目标世界高度 **0.378 m**，平均偏高 **6.86 cm**，实际约 **0.447 m**。
- 非轮部件接地比例为 0，没有检测到靠其他部件撑地。
- 六路电机、每个策略周期末采样的限矩占比合计平均 **33.20%**；这是电机与时间的联合占比，不是整机三分之一时间才限矩。
- 另取种子 81000 的 5 秒检查，左右后腿关节的限矩采样占比分别 **97.4% / 97.6%**，平均力矩约 **9.95 Nm**，接近当前 10 Nm 上限。前腿关节也接近当前 1.57 rad 上限，需要继续观察姿态与力矩，不能把存活时间视为高质量站立。

报告：`outputs/cad_dynamics/best_posture_audit.json`、`outputs/cad_dynamics/best_motor_audit.json`。当前评估未包含外力推扰、斜坡、摩擦/质量随机化和行走，不能由这些结果判断真机表现。

## 第二轮设置

配置：`configs/robot/cad_dynamics_round2.yaml`。

- 继续原地站立，改善姿态、高度和漂移；不加入前进、跳跃或爬台阶。
- 使用首轮 750,000 步 best 权重，再采样约 **2,000,000 步**；新轮训练步数从 0 计数。
- 单回合 5 → **20 秒**；初始倾角标准差 0.008 → **0.02 rad**，初始速度扰动标准差 0.005 → **0.01**。这属于初始化扰动，不是训练中施加推力。
- upright 权重 1 → **4**，hip_height 1 → **3**，linear_tracking 1 → **2**，yaw_tracking 0.5 → **1**，action_rate 0.02 → **0.03**。
- 学习率 **1e-4**，熵系数 **0.002**。
- 质量、惯量、CAD、弹簧、摩擦、关节范围、控制增益和力矩限值保持当前值。后腿持续限矩应在本轮继续跟踪，不通过任意提高力矩上限掩盖。

代码新增 `--warm-start`，允许在动力学与控制完全相同的前提下更换任务/奖励。`--resume` 用于任务与奖励不变的续跑，本轮请使用 `--warm-start`。保留策略和优化器状态，应用新配置中的学习率、熵系数与随机种子。

已完成：36 项测试通过；第二轮模型构建、2000 步 smoke 与 readiness 检查通过；第二轮 XML 与首轮完全一致。另已完成 **2048 步** warm-start 短跑，最终评估 10/10 回合站满 20 秒，验证加载与训练流程；短跑不证明第二轮已经改善站姿。第二轮初始化设置下的旧 best 基线也通过 10/10 × 20 秒，报告为 `outputs/cad_dynamics/round2_initial_audit.json`。

## 开始第二轮

在项目根目录执行；该命令在前台运行，运行目录不能已存在，重复启动时更换 `--run-name`：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/gpufree-data/.venvs/climber-validation/bin/python -u scripts/train.py \
  --backend cad-dynamics \
  --config configs/robot/cad_dynamics_round2.yaml \
  --warm-start outputs/mujoco_runs/cad_dynamics_standing_20261028_201934/best/best_model.zip \
  --total-timesteps 2000000 --num-envs 8 --device cpu \
  --run-name cad_dynamics_standing_round2_20261028
```

后台运行时，可在命令开头加 `nohup env`，末尾追加 `> outputs/cad_dynamics_round2/train.log 2>&1 &`。

第二轮状态与输出：

```bash
cat outputs/mujoco_runs/cad_dynamics_standing_round2_20261028/status.json
tail -f outputs/mujoco_runs/cad_dynamics_standing_round2_20261028/progress.csv
```

`scripts/training_status.py` 仍跟踪首轮的 `active_training.json`，不会因手动执行上述命令自动切换。配置或环境实现修改后，应先重做构建和 smoke：

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/prepare_cad_dynamics.py --config configs/robot/cad_dynamics_round2.yaml
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/smoke_mujoco.py --config configs/robot/cad_dynamics_round2.yaml --steps 2000
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/check_training_readiness.py --backend cad-dynamics --config configs/robot/cad_dynamics_round2.yaml
```

## 第二轮结束后的评估

best、final 均运行以下命令，分别更换 checkpoint 与输出文件名；先保持同一组种子比较。还应使用首轮相同配置复测，以区分初始化扰动变化造成的差异。奖励权重已改变，跨轮不要直接比较总回报，应比较倾角、高度误差、位移、存活率与限矩占比。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/audit_standing_policy.py \
  --checkpoint outputs/mujoco_runs/cad_dynamics_standing_round2_20261028/best/best_model.zip \
  --episodes 20 --horizon-seconds 30 --seed 83000 \
  --output outputs/cad_dynamics_round2/best_quality_30s.json
```

建议把 20/20 × 30 秒存活、平均倾角小于 5°、髋高误差小于 2 cm、结束平面位移小于 5 cm 作为下一轮观察目标，同时要求限矩比例明显下降；这些是训练目标，不是已达到的结果或真机放行标准。如果站姿和限矩仍未改善，应先检查 450 N 恒力弹簧、关节范围与可达站姿的匹配关系，再决定后续课程。
