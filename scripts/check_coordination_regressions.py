"""Verify corrective demonstrations on the previously inspected episodes."""
from improve_spring208_coordination import *


if __name__=='__main__':
    torch.set_num_threads(1)
    recipe=json.loads((OUT/'teacher_probe.json').read_text())['ranking'][0]['recipe']
    config=load_config(ROOT/'configs/robot/cad_spring208_phase3_test.yaml')
    offset=min(config['getup']['rearward_offsets_m'])
    jobs=[(recipe,offset,seed,None,1.,False,'test','folded',False) for seed in (445000,445001,445002)]
    rows=[r[0] for r in parallel(jobs)]
    report=dict(teacher_only=True,inspected_regression_cases=True,episodes=rows,
        successes=sum(r['success'] for r in rows),count=len(rows))
    (OUT/'teacher_regressions.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
