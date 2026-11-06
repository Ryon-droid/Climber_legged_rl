"""Identify unmet acquisition criteria without changing the policy or success gate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'source')]
from review_spring208_failures import *


def main():
    torch.set_num_threads(1)
    policy=PPO.load(RUN/'best/best_model.zip',device='cpu')
    config=load_config(CFG)
    reports=[]
    snapshot_jobs={'roll_failure':[.30,.60,1.50,2.06], 'timeout_failure':[10.], 'success_reference':[2.50]}
    snapshots=[]
    for slug,seed,title in CASES:
        env=make_mujoco_env(ROOT/config['experiment']['output_directory']/'robot_experiment.xml',CFG)
        task=env.unwrapped
        task.push.config=dict(task.push.config,episode_probability=0.)
        env.pose_bank=[p for p in env.pose_bank if abs(p['rearward_offset_m']-.3342953692058811)<1e-10]
        obs,_=env.reset(seed=seed,options={'reset_mode':'folded'})
        for geom in range(task.model.ngeom):
            body=int(task.model.geom_bodyid[geom])
            while body:
                name=task.model.body(body).name or ''
                if name.startswith(('left','right')):
                    task.model.geom_rgba[geom,:3]=([.16,.47,.82] if name.startswith('left') else [.88,.47,.13])
                    break
                body=int(task.model.body_parentid[body])
        renderer=mujoco.Renderer(task.model,height=270,width=480)
        camera=mujoco.MjvCamera();camera.distance=1.1;camera.elevation=-15
        vector=-task._rotation()[:,0]
        camera.azimuth=float(np.degrees(np.arctan2(vector[1],vector[0])))
        option=mujoco.MjvOption();option.geomgroup[1]=1
        rows=[]
        try:
            while True:
                action=policy.predict(obs,deterministic=True)[0]
                obs,_,done,truncated,info=env.step(action)
                lengths=np.asarray(info['virtual_leg_lengths_m'])
                _,base=contacts(task)
                criteria=dict(base_clear=not base, both_wheels=info['wheel_contacts']==2,
                    leg_length=bool(max(abs(lengths-task.target_virtual_leg_length))<=task.getup['leg_tolerance_m']),
                    hip_height=bool(abs(info['hip_height_m']-task.target_hip_height)<=task.getup['height_tolerance_m']),
                    tilt=bool(info['tilt_rad']<=task.getup['tilt_tolerance_rad']),
                    displacement=bool(info['planar_displacement_m']<=config['task']['stationary_max_displacement_m']),
                    heading=bool(abs(info['heading_error_rad'])<=config['task']['stationary_max_heading_rad']),
                    planar_speed=bool(info['planar_speed_m_s']<=config['task']['linear_tracking_sigma_m_s']),
                    yaw_rate=bool(abs(info['yaw_rate_rad_s'])<=config['task']['yaw_tracking_sigma_rad_s']))
                seconds=task.steps*task.dt
                pitch,roll=task._attitude()
                rows.append(dict(seconds=seconds,heading_deg=float(np.degrees(info['heading_error_rad'])),
                    displacement_mm=info['planar_displacement_m']*1000,
                    planar_speed_m_s=info['planar_speed_m_s'],yaw_rate_deg_s=float(np.degrees(info['yaw_rate_rad_s'])),
                    criteria=criteria,settled_seconds=task.settled_steps*task.dt,acquired=bool(task.acquired)))
                if any(abs(seconds-m)<1e-8 for m in snapshot_jobs[slug]):
                    shape=geometry(task)
                    camera.lookat[:]=np.mean(np.vstack([shape['motor_midpoints_world_m'],shape['wheel_centers_world_m']]),axis=0)
                    camera.lookat[2]=.16
                    renderer.update_scene(task.data,camera=camera,scene_option=option)
                    snapshots.append((slug,title,seconds,renderer.render().copy(),float(np.degrees(roll)),float(np.degrees(info['heading_error_rad']))))
                if done or truncated:break
        finally:
            renderer.close();env.close()
        source=json.loads((OUT/(slug+'.json')).read_text())
        if abs(source['seconds']-seconds)>1e-8 or source['success']!=bool(info['combined_success']):
            raise RuntimeError('Acceptance diagnostic changed trajectory')
        settled=[r for r in rows if r['seconds']>=1.5 and not r['acquired']]
        counts={k:sum(not r['criteria'][k] for r in settled) for k in rows[0]['criteria']}
        reports.append(dict(slug=slug,title=title,seed=seed,seconds=seconds,
            success=bool(info['combined_success']),max_settled_seconds=max(r['settled_seconds'] for r in rows),
            failed_gate_samples_after_1p5_before_acquisition=counts,
            last=rows[-1],trace=rows))
        print(slug,counts,'last',rows[-1],flush=True)
    (OUT/'acceptance_diagnostics.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False)+'\n')
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family']=FontProperties(fname=FONT).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(3,1,figsize=(11,8),constrained_layout=True)
    colors=['#b52d52','#7d5bb7','#28825c']
    for report,color in zip(reports,colors):
        t=[r['seconds'] for r in report['trace']]
        for ax,key in zip(axes,['heading_deg','planar_speed_m_s','settled_seconds']):
            ax.plot(t,[r[key] for r in report['trace']],label=report['title'],color=color)
            ax.grid(alpha=.2);ax.legend();ax.set_xlabel('仿真时间 / 秒')
    axes[0].axhline(5,color='#555555',ls='--',label='允许朝向误差 ±5°')
    axes[0].axhline(-5,color='#555555',ls='--')
    axes[0].set_ylabel('朝向误差 / 度');axes[0].legend()
    axes[1].axhline(.05,color='#555555',ls='--');axes[1].set_ylabel('平面速度 / m/s')
    axes[2].axhline(1,color='#555555',ls='--');axes[2].set_ylabel('连续满足起身条件 / 秒')
    fig.suptitle('为什么看起来站起，仍被判为超时：逐项核对成功条件')
    fig.savefig(OUT/'acceptance_diagnostics.png',dpi=130);plt.close(fig)
    canvas=Image.new('RGB',(960,3*350),'#f4f6fa');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,20);small=ImageFont.truetype(FONT,17)
    labels={('roll_failure',.3):'起身早期：右轮明显抬起',('roll_failure',.6):'0.6 秒：两轮重新接地',
        ('roll_failure',1.5):'1.5 秒：左右腿开始明显分化',('roll_failure',2.06):'2.06 秒：横滚越界，终止',
        ('timeout_failure',10.):'超时案例：已站起，但朝向未达标',('success_reference',2.5):'成功对照：满足稳定站立判据'}
    for i,(slug,title,seconds,frame,roll,heading) in enumerate(snapshots):
        x,y=(i%2)*480,(i//2)*350
        draw.text((x+10,y+6),labels[(slug,round(seconds,2))],font=font,fill='#182237')
        canvas.paste(Image.fromarray(frame),(x,y+35))
        draw.text((x+10,y+310),f't={seconds:.2f}s  横滚 {roll:+.1f}°  朝向 {heading:+.1f}°',font=small,fill='#26344a')
    canvas.save(OUT/'failure_keyframes.jpg',quality=94)


if __name__=='__main__':main()
