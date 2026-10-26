# 实际CAD机构修正结果

2026-10-27。用户要求使用真实模型，且确认底盘左右对称。本次生成独立的运动学修正候选，不覆盖原CAD、MJCF或已有训练结果。这里的“通过”只针对已列出的运动学测试，不能替代真实动力学认证。

## 已完成的具体修正

1. 原导出把左侧缸筒、活塞杆、上端头合在一个刚体，右侧杆part_433没有归属。现按原始STL恢复两侧四个真实零件：左part_582/583为缸筒侧，part_581/580为杆侧；右part_434/435为缸筒侧，part_433/432为杆侧。共恢复1个遗漏件，剩余72个仍需归属核对。
2. 两侧建立销轴转动+缸杆滑动+上端销轴闭合的机构，分别连接一级、二级连杆。以左侧已有CAD销轴和用户确认的对称关系构造右侧，缸/杆STL主轴与安装轴做数值一致性检查。
3. closure_007/008保留名称、重接实际杆端与二级连杆安装点；没有简单删除气弹簧约束。
4. 连续运动测试发现左closure_003的原锚点落在后连杆中间、偏离相邻小销轴体的轴心；右closure_r_005使用杆端销轴。现依据已有CAD小销轴位置及对侧拓扑，把左闭合锚点移到对应端部。其余刚体位姿、明确写入的非弹簧惯量、原几何、关节范围和四个主机构闭链均保留。
5. 弹簧拆分惯量按真实STL积分及源导出已有密度重新分摊，明确标记为未校准物性；没有将质量缩放为15 kg。原导出约7.733 kg，恢复杆后约7.736 kg，仍不足以训练真实15 kg底盘。

## 输出和复现

    /root/gpufree-data/.venvs/climber-validation/bin/python scripts/repair_cad_model.py
    MUJOCO_GL=egl /root/gpufree-data/.venvs/climber-validation/bin/python scripts/validate_cad_repair.py --video outputs/cad_repaired/cad_motion.mp4
    /root/gpufree-data/.venvs/climber-validation/bin/python -m unittest discover -s tests -p 'test_*.py' -v

- 配置：configs/robot/cad_repair.yaml。
- 修正XML：outputs/cad_repaired/robot_cad_repaired.xml。
- 改动证据、零件/安装点/哈希及单点测试：outputs/cad_repaired/repair_report.json。
- 连续101帧运动核验：outputs/cad_repaired/motion_validation.json、prescribed_motion.npz。
- 实际CAD指定轨迹回放：outputs/cad_repaired/cad_motion.mp4（不是PPO策略或有重力站立演示）。

24个腿部单电机±0.02/0.05/0.1 rad检查点全部通过。两侧135 mm髋轮垂距复现，六个闭链均保留。连续101帧小幅同步运动，闭链端点最大残差约4.4e-9 m、左右轮x/z位置差约1.3e-8 m，非首帧最大配置步长约0.00252。小销轴辅助体的自由自转在指定轨迹求解时连续保留，不把数值求解的自由角跳变写成真实运动。

## 尚未完成的真实模型部分

这是固定基座、无重力、无碰撞、无电机输出的机构诊断副本。滑动关节不设未知的真实行程边界，也不施加未知的真实气弹簧力。它仍需真实安装关系核对、气弹簧行程/力曲线、逐刚体质量/COM/惯量及驱动输出包络，再建立自由底盘接触动力学。

现有指定轨迹需要的两眼中心距约163.3～166.2 mm，只是模型在该轨迹的几何需求，不是气弹簧已测行程，也不是强行给实物指定的新尺寸。

scripts/train.py --backend cad-repaired 和就绪检查可生成修正报告，但明确返回2，阻止未校准模型启动训练。默认cad仍核验未改动的源导出，用于保留问题来源。

用户确认目前没有逐刚体物性，商家不提供气弹簧资料。可自行收集的最低数据、录入表和校验见 [无厂家资料的参数补齐](无厂家资料的参数补齐.md)。不需要重复提供整机15 kg、每侧5.14 kg和135 mm站姿。

最终复核：24项回归测试通过；回放与当前生成XML的SHA256一致，原始XML的SHA256保持不变。当前完成的是CAD运动学修正候选，真实动力学和PPO训练仍受实际参数缺失阻塞。
