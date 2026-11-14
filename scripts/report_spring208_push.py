"""Write the completed push experiment's outcome without modifying checkpoints."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);args=p.parse_args()
    folder=args.folder
    report=json.loads((folder/'assessment.json').read_text())
    before=json.loads((folder/'parent_test.json').read_text())
    after=json.loads((folder/'selected_test.json').read_text())
    labels=dict(standing='无推扰站立',push_0='2～5 N 推扰',push_1='5～10 N 推扰',push_2='10～15 N 推扰',
                nominal_getup='正常趴姿起身',difficult_getup='困难趴姿起身',regression_standing='历史偏航场景')
    lines=['# 抗推扰专项训练结果','',
        '本轮通过预设抗推扰验收，可进入低速移动专项。' if report['gate_passed'] else
        '本轮未通过预设抗推扰验收，不能宣称已经完成抗推扰训练。',
        '', '| 测试组 | 原策略 | 本轮候选 |', '| --- | --- | --- |']
    for key,label in labels.items():
        a=report['parent_summary'][key];b=report['selected_summary'][key]
        lines.append(f"| {label} | {a['success']}/{a['count']} | {b['success']}/{b['count']} |")
    failures=[dict(spec=r['spec'],reason=r['reason'],maximum=r['maximum'],
                   final_errors=r['final_errors'],push_metrics=r['push_metrics'])
              for r in after['episodes'] if not r['success']]
    def key(r):
        s=r['spec'];return s['group'],s['seed'],s.get('offset_m'),s['push']
    old={key(r):r for r in before['episodes']}
    lost=[r['spec'] for r in after['episodes'] if old[key(r)]['success'] and not r['success']]
    gained=[r['spec'] for r in after['episodes'] if not old[key(r)]['success'] and r['success']]
    physical=dict(warnings=sum(r['warnings'] for r in after['episodes']),
        max_spring_upper_mm=max(max(r['spring_upper_mm']) for r in after['episodes']),
        max_spring_lower_mm=max(max(r['spring_lower_mm']) for r in after['episodes']),
        total_applied_pushes=sum(r['push_metrics']['push_count'] for r in after['episodes']),
        recovered_pushes=sum(r['push_metrics']['recovered_push_count'] for r in after['episodes']))
    report.update(independent_failures=failures,paired_lost_cases=lost,paired_gained_cases=gained,
                  physical_checks=physical)
    (folder/'assessment_analyzed.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines.extend(['',f"新增训练交互：{report['new_training_steps']:,} 步；训练、分段筛选和配对验收共 {report['seconds']/60:.1f} 分钟。",'',
        '训练只更新站立网络，起身网络与速度估计器逐张量保持不变。选模与最终测试使用不同种子，最终测试不参与选模。',
        '', '物理检查：'+json.dumps(physical,ensure_ascii=False)+'。',
        '', f"同组配对测试：新成功 {len(gained)} 个，原来成功而本轮失败 {len(lost)} 个。",'',
        '推力持续 0.05～0.10 秒、每回合两次；恢复期限 4 秒且需连续稳定 0.5 秒。验收分别要求三档推扰及无推扰站立至少 95%，正常趴姿全部成功、已成功的站立与起身场景不退步、零数值警告及限位穿透不超过 1 mm。',
        '', '困难趴姿至少 95% 的目标仍单独记录。正常趴姿为同一确定性姿态的重复核验。一次训练种子和有限仿真样本不能保证实机表现。',
        '', '候选文件：`selected_policy.zip`。原保留策略文件未覆盖；没有实机部署。',
        '', '后续移动采用前进、后退、左转、右转共用速度指令策略，先做 ±0.10 m/s、±0.30 rad/s 的单轴动作与停车验收。已准备保持当前模型和观测接口的适配代码；没有启动旧 220 mm 模型训练。',
        '', '失败回合、推力方向、脉冲恢复情况、配对退步以及物理检查详见 `assessment_analyzed.json`。'])
    (folder/'抗推扰专项结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('gate_passed','new_training_steps','seconds','parent_summary','selected_summary','physical_checks')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
