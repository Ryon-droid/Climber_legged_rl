# CAD动力学站立训练

> **第五轮已完成（2026-10-29 00:40:07）：**best 独立 30 回合全部原地站满 60 秒，腿长均值 134.87 mm；final 24/30。最新第六轮轻推扰训练及回放指令见 [第五轮评估与第六轮推扰训练](第五轮评估与第六轮推扰训练.md)。

> **第四轮原地训练已完成（2026-10-28 23:19:09）：**独立 30 回合测试 best 原地合格 29/30，final 28/30。仍有初态恢复失败和约 10 mm 腿长偏差。最新第五轮配置及回放指令见 [第四轮原地评估与第五轮训练](第四轮原地评估与第五轮训练.md)。

> **原地站稳目标：**第三轮存在一轮近乎静止、另一轮驱动绕圈的问题。已更新第四轮的位置/朝向奖励、观测和验收，最新命令见 [原地站稳训练修正](原地站稳训练修正.md)。站满时长不等同于原地站稳。

> **第三轮已完成（2026-10-28 22:08:42）：**独立 30 回合、每回合 30 秒评估，best 站满 21/30，final 站满 13/30。第四轮配置与最新训练、回放指令见 [第三轮评估与第四轮训练](第三轮评估与第四轮训练.md)。下文原首轮及旧第三轮状态为历史记录。

> **135 mm 新站姿训练结果：**`cad_dynamics_midpoint135_20261028` 已于 2026-10-28 21:30:48 完成，但独立 30 回合评估全部主体触地，平均存活不足 1 秒，尚未学会站立。已准备第三轮恢复训练配置与回放命令，见 [135毫米训练评估与第三轮](135毫米训练评估与第三轮.md)。

> **最新尺寸确认：**135 mm 为每侧两个实物电机轴心中点到轮毂中心的侧视直线距离，非髋轮垂距。已据此调整水平站姿和奖励，原水平训练停止保存。查看 [135毫米中点距离校正](135毫米中点距离校正.md)。

> **最新修正：主体应与地面平行。**此前 CAD 根坐标被误当作主体参考，首轮及原第二轮站姿不合要求。已停止原第二轮，修正安装板参考、135 mm 初始站姿和姿态奖励；38 项测试通过。旧策略不能直接续训，查看 [主体水平姿态修正](主体水平姿态修正.md)。本文后续 320 mm 描述与旧训练命令为历史记录。

更新：首轮已于 **2026-10-28 20:28:25（北京时间）** 完成，实际采样 1,001,472 步。独立评估结果、站姿问题及已验证的第二轮命令见 [CAD首轮评估与第二轮训练](CAD首轮评估与第二轮训练.md)。第二轮完整训练尚未启动。

2026-10-28：用户要求接受当前质量等模型参数，完成代码并直接开始训练。本次新增独立的 `cad-dynamics` 路线，使用修正CAD继续构建自由底盘动力学，不使用旧15 kg近似实验模型或旧checkpoint。

## 已实现

- 原23个刚体、511个CAD网格和六个闭链保留，惯性参数不缩放；五项最新质量仍为base 12.7 kg、左右轮各0.54 kg、左右link2各0.89 kg，模型合计16.463456595727138 kg。
- 启用浮动基座、重力、真实CAD网格与地面接触，以及四腿、双轮共六路输出力矩驱动。装配内部邻接件可能互相接触，首轮仅启用机器人与地面的碰撞，不启用自碰撞。
- 两侧气弹簧滑动自由度施加450 N标称向外推力；这是标称恒力模型，行程、力长曲线、摩擦与阻尼没有被宣称为实测值。
- 24维观测、6维动作、带限幅的腿部PD/车轮速度环、姿态/高度奖励、跌倒/接触/闭链终止、定期评测与checkpoint。
- 数值警告、非有限状态和仿真时间重置会使训练报错退出并保存中断checkpoint，不把数值发散当作有效训练样本。
- 训练前核对配置、模型、环境代码、构建代码和参数来源哈希，拒绝复用过期检查报告。

当前COM、惯量、其他18项质量接受为本轮仿真参数。摩擦、阻尼、控制增益和电机限幅集中于 `configs/robot/cad_dynamics.yaml`；腿力矩上限10 Nm、轮力矩上限2 Nm沿用历史仿真试用设置，不是新增硬件测量。

## reset与135 mm姿态

135 mm髋轮垂距的闭链运动学求解继续通过。但是将当前完整CAD网格全部用于地面接触后，这个姿态的base最低点会低于轮底约126.7 mm。250 mm仍有约11.7 mm穿地；求解至320 mm后base最低点净空约58.3 mm。

因此本轮训练reset使用闭链可行且轮接地点位于整机下方的320 mm髋轮垂距，左右轮沿x方向对齐当前整机COM。记录保存在 `outputs/cad_dynamics/model_report.json` 的 `pose_attempts` 和 `rest_height_hip_above_wheel_m`。这是本轮仿真站姿，不宣称已复现135 mm实物落地站姿；若要求该低站姿，应进一步核对base中网格的归属或CAD坐标。不会通过关闭主体碰撞掩盖穿地。

## 本次验证与训练

- 31项测试通过，包括质量/惯量保留、六驱动映射、闭链、reset净空、气弹簧力、动作限幅和无数值警告地面接触。
- 零动作与随机动作各2000步smoke通过；这证明可采集有效物理转移，不代表零动作已能站稳。
- CLI端到端4096步PPO已完成并产生最终策略和评测，平均存活约0.773 s。该短跑只验证训练通路。
- 已启动 `cad_dynamics_standing_20261028_201934`，目标1000000步、8个环境、从零训练。MuJoCo物理为CPU多进程，当前小型MLP使用CPU训练。
- 每10000步保存checkpoint，每50000步评测5个episode并保存best；最终评测10个episode。episode上限5 s。训练已经启动不代表已经收敛，结果以评测为准。

## 查看当前进度

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/training_status.py
tail -f outputs/cad_dynamics/cad_dynamics_standing_20261028_201934.log
```

进程号和启动参数：`outputs/cad_dynamics/active_training.json`。本次运行目录：`outputs/mujoco_runs/cad_dynamics_standing_20261028_201934/`，内含 `status.json`、`progress.csv`、`checkpoints/`、`best/`、配置和模型快照。后台进程不依赖当前终端，但机器重启或进程退出后不会自动续训。

## 重建与重新训练

```bash
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/repair_cad_model.py
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/prepare_cad_dynamics.py
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/smoke_mujoco.py --config configs/robot/cad_dynamics.yaml --steps 2000
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/check_training_readiness.py --backend cad-dynamics
/root/gpufree-data/.venvs/climber-validation/bin/python scripts/train.py --backend cad-dynamics --total-timesteps 1000000 --num-envs 8 --device cpu --run-name NEW_UNIQUE_NAME
```

原 `cad` / `cad-repaired` 命令继续用于机构诊断，默认不偷偷开启长训练。新增路线显式使用 `--backend cad-dynamics`，不需要旧近似模型参数开关。

评测/回放和TorchScript导出仍使用 `scripts/play.py`、`scripts/export.py`，指向本次运行的checkpoint即可。STM32/CubeAI和真机控制未在本次任务中运行。
