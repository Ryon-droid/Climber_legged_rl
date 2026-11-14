"""Report native paired results, actual costs and protected-actor contracts."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'scripts')]
from stable_baselines3 import PPO
from scipy.stats import binomtest
from supermimic_pilot import PARENT,CFG,load_config,write_json,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
    folder=args.folder;torch.set_num_threads(1)
    report=json.loads((folder/'comparison.json').read_text())
    states=[]
    for line in (folder/'workflow.log').read_text().splitlines():
        try:item=json.loads(line)
        except json.JSONDecodeError:continue
        if 'state' in item:states.append(item)
    initial_cost=next(r['seconds'] for r in states if r['state']=='world_model_policy_training' and r['cycle']==0)
    timing=copy.deepcopy(report['timing'])
    # The first coordinator version started this arm's stopwatch after initial
    # selection. Charge that measured cost too, matching both PPO arms.
    if 'initial_selection_seconds' not in timing['world_model']:
        timing['world_model']['initial_selection_seconds']=initial_cost
        timing['world_model']['inclusive_seconds']+=initial_cost
    baseline=PPO.load(PARENT,device='cpu')
    frozen={};actor_changed={};paired={};action_std={}
    configured_std=torch.log(torch.tensor(load_config(CFG)['training']['action_std']))
    rows={name:json.loads((folder/f'test_{name}.json').read_text())['episodes']
          for name in ('parent','world_model','ppo_current','ppo_reference')}
    def key(r):return r['split'],r['offset_m'],r['mode'],r['push'],r['seed']
    parent_lookup={key(r):r for r in rows['parent']}
    names=('world_model','ppo_current','ppo_reference')
    for name in names:
        model=PPO.load(folder/name/'selected_policy.zip',device='cpu')
        protected=[];changed=[]
        for k,v in baseline.policy.state_dict().items():
            other=model.policy.state_dict()[k]
            if k.startswith(('mlp_extractor.policy_net.standing_','mlp_extractor.policy_net.velocity_estimator.')):
                if not torch.equal(v,other):protected.append(k)
            if k.startswith(('mlp_extractor.policy_net.rising_net.','mlp_extractor.policy_net.rising_head.')):
                if not torch.equal(v,other):changed.append(k)
        expected_std=baseline.policy.log_std if name=='world_model' else configured_std
        checkpoints=list((folder/name).glob('step_*.zip')) if name!='world_model' else []
        for checkpoint in checkpoints:
            candidate=PPO.load(checkpoint,device='cpu')
            if not torch.equal(expected_std,candidate.policy.log_std):
                raise RuntimeError('Action standard deviation changed during training: '+str(checkpoint))
        if not torch.equal(expected_std,model.policy.log_std):
            raise RuntimeError('Selected policy changed its configured action standard deviation')
        action_std[name]=dict(values=model.policy.log_std.detach().exp().tolist(),
            same_as_parent=torch.equal(baseline.policy.log_std,model.policy.log_std),
            expected_configuration_verified=True,training_checkpoints_checked=len(checkpoints))
        if protected:raise RuntimeError('Protected selected-policy parameters changed: '+str(protected))
        frozen[name]=True;actor_changed[name]=bool(changed)
        wins=losses=0
        for r in rows[name]:
            if r['split']!='test':continue
            b=parent_lookup[key(r)]
            wins+=int(r['success'] and not b['success']);losses+=int(b['success'] and not r['success'])
        paired[name]=dict(wins=wins,losses=losses,
            exact_two_sided_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)
    world_curves=json.loads((folder/'world_model/curve.json').read_text())
    data=dict(report,analyzed_timing=timing,protected_actor_tensors_identical=frozen,
        selected_rising_actor_changed=actor_changed,paired_fresh_vs_parent=paired,
        action_standard_deviation=action_std,
        post_training_action_diagnostics={str(i):json.loads((folder/f'action_probe_model_{i}.json').read_text()) for i in (0,3)},
        world_model_candidate_selection_keys=[r['selection']['key'] for r in world_curves],
        reached_target_time_seconds={name:None for name in names},
        target_time_note='Only a validated target crossing can establish time-to-target; none is inferred from model losses or an unchanged selected parent.')
    write_json(folder/'comparison_analyzed.json',data)
    titles=dict(parent='原策略',ppo_current='当前 PPO 续训',ppo_reference='参考轨迹＋PPO',world_model='小型世界模型方案')
    lines=['# 起身算法对照：首轮结果','',
        '2026-11-09。仿真筛选实验，单训练种子；世界模型方法为 SuperMimic 思路的小型适配，并非官方完整复现。','',
        '结论：本轮没有验证出比保留的原 PPO 策略效果更好、达到目标更快的方案。继续保留原策略；不自动切换，也不自动追加训练。','',
        '| 方法 | 新种子困难趴姿 | 原始趴姿 | 站立/推扰 | 新增交互 | 流水线耗时 | 达到目标 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for name in ('parent','ppo_current','ppo_reference','world_model'):
        s=report['summaries'][name]
        success=[f"{s[k]['success']}/{s[k]['count']}" for k in ('folded','nominal','standing')]
        t=timing.get(name)
        minutes=f"{t['inclusive_seconds']/60:.2f} 分钟" if t else '已有策略'
        samples=f"{t['total_native_transitions']:,}" if t else '—'
        passed='是' if report['passed_95_percent_gate'][name] else '否'
        lines.append('| '+' | '.join([titles[name],*success,samples,minutes,passed])+' |')
    lines.extend(['','目标：困难趴姿至少 95%，原始趴姿全部成功，站立无退步且没有数值警告。测试为已知留出几何配全新种子。',
        '', '耗时包含采集、模型拟合、在线交互、策略优化和选模；最终测试另计。已把世界模型组初次选模时间补入，原始计时文件保留。首次代码实现、调试及前期诊断拟合不计入可重复流水线时间。',
        '', '三组都保留初始策略参与选模。因此“最终结果相同”可能表示更新全部失败、选回了原策略。', '',
        '选出的起身网络是否真正改变：'+json.dumps(actor_changed,ensure_ascii=False)+'。',
        '', '世界模型四次候选的选模指标（原始趴姿成功数、最差姿态成功数、其他趴姿成功数）：'+str(data['world_model_candidate_selection_keys'])+'。',
        '', '所有选出策略的站立网络、速度估计器均逐张量核对，与原策略保持一致。PPO 两组训练开始时按现行配置把探索标准差从腿 0.015/轮 0.15 改为腿 0.0015/轮 0.015，之后所有保存的训练检查点保持固定；世界模型选回的原策略保留原标准差。最终测试使用确定性动作，该标准差不参与动作计算。最终测试没有示范控制器或世界模型替代动作。',
        '', '原始趴姿沿用无额外扰动的确定性配置，8 次为重复核验；随机鲁棒性主要由 48 个扰动趴姿及 16 个站立/推扰回合衡量。三组训练均不施加外部推力。',
        '', '若未达到目标，就不能根据一次训练用时更短宣称更快达标。单训练种子也不足以证明算法稳定优越。',
        '', '配对困难趴姿相对原策略的胜/负：'+json.dumps(paired,ensure_ascii=False)+'。',
        '', '诊断：在 28 个训练状态上做 336 次原生动作扰动查询。首版模型的动作响应方向余弦中位数为 -0.089，后续为 0.603；但相对导数误差仍约 0.985，奖励对动作的梯度方向中位数仍为 -0.052。能预测已有轨迹，不等于能准确指导动作更新。这是本次退步的可能原因，不能据此证明完整 SuperMimic 方法无效。',
        '', '下一步建议：先增加成功与失败边缘状态附近的受限动作扰动数据，并验证模型对动作变化的响应；优先采用可解释的物理奖励计算，检查奖励梯度。通过这些检查后，再决定是否做多个随机种子的完整对照。当前不值得直接扩大世界模型训练预算。',
        '', '详细结果：`comparison_analyzed.json`；模型预测验证：`world_validation_*.json`；原生几何检查：`world_geometry_*.json`；动作响应诊断：`action_probe_model_*.json`。'])
    (folder/'首轮对照结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:data[k] for k in ('summaries','analyzed_timing','selected_rising_actor_changed','paired_fresh_vs_parent')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
