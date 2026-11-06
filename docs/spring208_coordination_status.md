# 困难趴姿协调修正：当前状态

2026-11-06：已加入训练专用的左右腿协调示范、近站立阶段稳定示范、分阶段纠正学习及原生策略独立评估脚本。

示范控制仅用于生成训练标签，最终评估直接运行学生网络，不增加运行时示范控制器，不放宽起身验收条件。站立网络、速度估计器等受保护部分保持冻结。气弹簧最大销轴中心距仍为 208 mm，最小为 155 mm。

本次运行 `spring208_coordination_20261106_2038` 已按用户要求于北京时间约 20:40 停止。完成了阶段 0、1、2，并保留初始、阶段最终及阶段最佳检查点。独立测试尚未运行，不能据此认定泛化效果已经改善，也未替换此前保留的最佳策略。

运行状态：`outputs/spring208_coordination_fix/spring208_coordination_20261106_2038.json`。
检查点和阶段记录：`outputs/mujoco_runs/spring208_coordination_20261106_2038/`。
这些运行产物保留在训练设备，不提交到 Git。

脚本入口：

- `scripts/improve_spring208_coordination.py`：示范方案检查、纠正标签收集和网络学习。
- `scripts/assess_spring208_coordination.py`：父策略与候选策略的配对独立评估。
- `scripts/check_coordination_regressions.py`：已知失败情形的示范诊断，不是学生独立评估。
- `scripts/run_spring208_coordination.py`：显式启动一次有界训练及后续评估；不会自动延长训练。

上述脚本依赖训练设备上已经准备好的模型产物和父策略检查点；仅克隆代码不会包含这些被忽略的运行文件。当前没有在运行的纠正训练，后续训练需用户重新安排。
