# 第十轮PPO评估与第十一轮示范回放

> **用户最新澄清：起身只有一种固定趴姿。**后续应围绕实物固定初态训练，不能再把360/370/375 mm三个仿真示例当成任务要求。固定趴姿的前后水平距离尚待确认；当前训练入口会在配置仍有多个初态时明确停止，不会擅自挑选其中一个。下文三姿态成绩保留为历史评估；下一轮指令待固定初态、示范及检查同步更新后再运行。

2026-10-29，第十轮运行 `cad_getup_stand_round10_ppo_refine_20261029_180330` 已完成，实际采样102400步。不是仍在训练，也不是起身任务已全部解决。

## 质量结论

独立评估使用seed=98500，每组20回合、每回合最多20秒，全程执行模型输出、无推扰。

|模型|从趴姿起身并保持到20秒|独立站姿保持20秒|说明|
|---|---:|---:|---|
|第十轮final，新增独立评估|14/20|19/20|370 mm初态6次全部失败|
|第十轮best，保留第0步父模型|13/20|17/20|引用相同权重父模型此前seed=98200评估，非本轮新增40回合|

两行种子不同，不能以14大于13、19大于17认定提升。趴姿实际只有三个确定初态；final本次抽到360/370/375 mm分别3/6/11次，通过3/0/11次。均衡看仍只通过三种中的两种；这20回合不是20种独立物理扰动。

final的360/375 mm起身判定分别约2.49/3.03秒，最终位移44.2/47.4 mm。370 mm则10秒超时，最大位移546.8 mm、最大倾角68.8度，明显不能作为可靠起身模型。站姿20回合有1回合主体触地。两组MuJoCo数值警告均为0；没有数值警告不等于控制合格。

训练时5回合随机趴姿选模成绩：0/2/4/6/8/10万步分别80/60/20/40/80/20%，站姿选模均100%。8万步虽与父模型成功率打平，奖励更低，未替换best。对8万步三种趴姿逐一复测也只有360/375 mm成功，370 mm侧滚失败。

逐项核对第十轮best与父模型：只有探索参数log_std变化，确定性动作均值参数完全相同，best仍为第0步。不能把保留父模型称为本轮学会了新动作。

证据：
- [训练状态](../outputs/mujoco_runs/cad_getup_stand_round10_ppo_refine_20261029_180330/status.json)
- [本轮final独立评估](../outputs/cad_getup_stand_round10_ppo_refine/assessment_final_20.json)
- [best权重对应关系](../outputs/cad_getup_stand_round10_ppo_refine/assessment_best_lineage.json)
- [80k检查点370 mm复测](../outputs/cad_getup_stand_round10_ppo_refine/assessment_80k_0.37.json)

## 为什么有三种趴姿

配置 `rearward_offsets_m: [0.36, 0.37, 0.375]` 表示电机轴心中点到轮心的前后水平偏移。三种都是主体水平、腹部朝下、腿向后伸，仅后伸程度不同。它们是仿真选择的初态，不是用户确认的三组实测尺寸，也不是135 mm站姿侧视距离或155–220 mm气弹簧长度。每个起身回合从中随机选一个，同一个模型完成起身与站稳。

若实物趴下总由同一机械限位确定，应该先以该实测初态集中学习，再逐渐增加附近变化；目前没有擅自认定其中哪个值等于实际限位，也没有修改这些数。

## 下一轮指令

新增 `configs/robot/cad_getup_stand_round11_ppo_replay.yaml` 和入口：

```bash
cd /root/gpufree-data/Climber_legged_rl
bash scripts/train_getup_ppo_replay.sh
```

本轮从第十轮best初始化，12万步、8环境、PPO学习率5e-6，探索标准差0.02，clip=0.05，target KL=0.005；仍70%趴姿、30%站姿，不加推扰。每个PPO rollout更新完成后额外做4次actor示范回放更新，batch512、学习率1e-6，目的是约束已学会的动作，尝试降低遗忘；这是下一轮实验，不保证一定提高。

回放数据来自174527轮示范文件的前24000条成功专家样本，不混入后续失败学生轨迹标签。数据哈希、物理XML、控制映射和成功专家记录在训练前核验。回放只用于训练，运行时仍是一个actor输出六路动作，不调用教师或替换动作。

每2万步选模时逐一测试360/370/375 mm三种初态，加10回合站姿，减少随机抽样遗漏370 mm的情况。初始父模型先保存为best，新权重必须超过它才替换；final单独保存。日志保留剩余时间，位于 `outputs/cad_getup_stand_round11_ppo_replay/`；策略输出位于 `outputs/mujoco_runs/cad_getup_stand_round11_ppo_replay_时间戳/`。

训练入口已准备；本次不会自动启动完整12万步。模型质量、弹簧、力矩边界及起身验收阈值不变。

已通过24项针对性测试（示范回放3项、选模4项、策略迁移17项）、1000步零/随机动作smoke，以及实际CAD环境512步PPO加示范更新检查。512步验证确认初始动作保留、actor发生更新、保存后能用普通PPO加载播放；不作为起身成功率证据。验证记录在 `outputs/cad_getup_stand_round11_ppo_replay/validation/update_check.json`。

## 当前MuJoCo回放

打开已保留的第十轮best，固定看能起身的360 mm姿态：

```bash
cd /root/gpufree-data/Climber_legged_rl
bash scripts/play_latest_getup.sh --rearward-offset-m 0.36
```

查看仍失败的370 mm姿态：

```bash
bash scripts/play_latest_getup.sh --rearward-offset-m 0.37
```

不带offset参数会在三种初态中抽样。该入口明确固定第十轮best，下一轮训练后不会自动改用未经评估的新权重；MuJoCo窗口需要在有图形桌面的终端运行。
