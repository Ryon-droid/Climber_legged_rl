"""Reproduce assessed failures; record native-policy rollouts and bilateral traces."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
import imageio_ffmpeg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from stable_baselines3 import PPO
from climber_legged_rl.cad_dynamics import validate_artifacts
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import load_config
from climber_legged_rl.spring_geometry import eye_center_distance
from climber_legged_rl.stance_geometry import virtual_leg_geometry

RUN = ROOT/'outputs/mujoco_runs/spring208_phase3_20261106_1639'
OUT = ROOT/'outputs/spring208_failure_review'
CFG = ROOT/'configs/robot/cad_spring208_phase3_test.yaml'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BLUE, ORANGE = '#2879d0', '#e17b22'
CASES = [('roll_failure',445000,'侧倾失败'), ('timeout_failure',445001,'起身超时'),
         ('success_reference',445002,'同姿态成功对照')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(env):
    return virtual_leg_geometry(env.model,env.data,env.motor_midpoints,env.wheel_centers,env.chassis_axes)


def contacts(env):
    wheels=[False,False]
    base=False
    for contact in env.data.contact:
        pair={contact.geom1,contact.geom2}
        if env.ground_id not in pair:
            continue
        base |= bool(pair.intersection(env.base_geom_ids))
        for geom in pair.intersection(env.wheel_geom_ids):
            body=int(env.model.geom_bodyid[geom])
            for i, wheel in enumerate(env.wheel_ids):
                wheels[i] |= body==wheel
    return wheels,base


def capture_state(env, seconds, torque, peak, saturation, command, targets, wheel_targets, dt):
    shape=geometry(env)
    pitch,roll=env._attitude()
    wheels,base=contacts(env)
    return dict(seconds=seconds, pitch_deg=float(np.degrees(pitch)),roll_deg=float(np.degrees(roll)),
        q_rad=env.data.qpos[env.q_indices].tolist(), qvel_rad_s=env.data.qvel[env.v_indices].tolist(),
        targets_rad=targets.tolist(), wheel_targets_rad_s=wheel_targets.tolist(),
        action=command.tolist(), mean_torque_Nm=torque.tolist(),peak_abs_torque_Nm=peak.tolist(),
        saturation_fraction=saturation.tolist(),wheel_contacts=wheels,base_contact=base,
        spring_mm=[1000*eye_center_distance(env.model,env.data,side) for side in ('left','right')],
        wheel_z_mm=(shape['wheel_centers_world_m'][:,2]*1000).tolist(),
        virtual_leg_mm=(shape['side_view_distance_m']*1000).tolist(),
        hip_height_mm=float(np.mean(env.data.xpos[env.hip_ids,2])*1000),
        acquired=bool(env.acquired), policy_phase=float(env._phase()),
        kp=float(env.control['leg_kp']), kd=float(env.control['leg_kd']),
        torque_limits_Nm=env.torque_limits.tolist(),sample_dt=dt)


def draw_frame(env,renderer,cameras,options,row,title,seed,encoder,frames,endpoint=False):
    canvas=Image.new('RGB',(1280,570),'#f6f8fb')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,20)
    small=ImageFont.truetype(FONT,17)
    for i,camera in enumerate(cameras):
        shape=geometry(env)
        camera.lookat[:]=np.mean(np.vstack([shape['motor_midpoints_world_m'],shape['wheel_centers_world_m']]),axis=0)
        camera.lookat[2]=.16
        renderer.update_scene(env.data,camera=camera,scene_option=options)
        canvas.paste(Image.fromarray(renderer.render().copy()),(i*640,38))
        draw.text((i*640+12,44),'侧面视角' if i==0 else '前后方向视角',font=small,fill='#182237')
    draw.rectangle((0,0,1280,37),fill='#182237')
    draw.text((12,5),f'{title}  |  t={row["seconds"]:.2f}s  |  seed={seed}  |  0.5 倍速 / 仿真复现',font=font,fill='white')
    draw.text((16,411),'蓝色：左侧',font=font,fill=BLUE)
    draw.text((245,411),'橙色：右侧',font=font,fill=ORANGE)
    draw.text((485,411),f'横滚 {row["roll_deg"]:+.1f}°   俯仰 {row["pitch_deg"]:+.1f}°',font=font,fill='#182237')
    q=np.degrees(row['q_rad'][:4])
    for side,index,color in [('左',0,BLUE),('右',2,ORANGE)]:
        y=447+(index//2)*31
        draw.text((16,y),f'{side}腿前 / 后关节：{q[index]:+.1f}° / {q[index+1]:+.1f}°',font=small,fill=color)
        draw.text((445,y),f'轮接地：{"是" if row["wheel_contacts"][index//2] else "否"}',font=small,fill=color)
        draw.text((665,y),f'轮心高度：{row["wheel_z_mm"][index//2]:.1f} mm',font=small,fill=color)
        draw.text((960,y),f'弹簧：{row["spring_mm"][index//2]:.1f} mm',font=small,fill=color)
    draw.text((16,522),'角度为模型关节坐标；颜色仅辅助辨认。终止帧后不继续施加动作。',font=small,fill='#526176')
    if endpoint:
        draw.text((940,522),'回合终止 / 画面定格',font=small,fill='#b52929')
    encoder.send(np.asarray(canvas))
    for timepoint in (0.,.6,1.02,1.5,2.,3.,6.,10.):
        if abs(row['seconds']-timepoint)<1e-7:
            frames.append((row['seconds'],canvas.copy()))
    return canvas


def simulate(slug,seed,title,reference,policy):
    config=load_config(CFG)
    xml=ROOT/config['experiment']['output_directory']/'robot_experiment.xml'
    env=make_mujoco_env(xml,CFG)
    task=env.unwrapped
    env.pose_bank=[p for p in env.pose_bank if abs(p['rearward_offset_m']-reference['offset_m'])<1e-10]
    task.push.config=dict(task.push.config,episode_probability=0.)
    obs,reset=env.reset(seed=seed,options={'reset_mode':'folded'})
    # These visual-only colors do not change geometry, contact, forces or policy observations.
    for geom in range(task.model.ngeom):
        body=int(task.model.geom_bodyid[geom])
        side=None
        while body:
            name=task.model.body(body).name or ''
            if name.startswith('left'): side='left';break
            if name.startswith('right'): side='right';break
            body=int(task.model.body_parentid[body])
        if side:
            task.model.geom_rgba[geom,:3]=([.16,.47,.82] if side=='left' else [.88,.47,.13])
    axes=task.data.xaxis[[task.model.joint(n).id for n in config['control']['leg_joints']]].copy()
    lateral=task._rotation()[:,1]
    axis_sign=np.sign(axes@lateral)
    renderer=mujoco.Renderer(task.model,height=360,width=640)
    cameras=[]
    rot=task._rotation()
    for vector in (rot[:,1],-rot[:,0]):
        camera=mujoco.MjvCamera()
        camera.distance=1.10
        camera.azimuth=float(np.degrees(np.arctan2(vector[1],vector[0])))
        camera.elevation=-12
        cameras.append(camera)
    options=mujoco.MjvOption(); options.geomgroup[1]=1
    encoder=imageio_ffmpeg.write_frames(str(OUT/(slug+'.mp4')),(1280,570),fps=25,
        codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=7,
        macro_block_size=2,output_params=['-movflags','+faststart'])
    encoder.send(None)
    trace=[];frames=[]
    torques=[]
    original=task._observe_motor_torque
    def observe(torque):
        original(torque)
        torques.append(torque.copy())
    task._observe_motor_torque=observe
    zero=np.zeros(6)
    row=capture_state(task,0.,zero,zero,zero,zero,task.data.qpos[task.q_indices[:4]].copy(),zero[:2],0.)
    trace.append(row)
    draw_frame(task,renderer,cameras,options,row,title,seed,encoder,frames)
    try:
        while True:
            action=policy.predict(obs,deterministic=True)[0]
            command=np.clip(action.astype(np.float32),-1.,1.)
            targets=task._leg_targets(command).copy()
            wheel_targets=command[4:]*task.control['wheel_action_scale_rad_s']
            torques.clear()
            obs,_,done,truncated,info=env.step(action)
            torque=np.asarray(torques)
            if len(torque)!=config['control']['decimation']:
                raise RuntimeError('Incomplete motor torque sampling')
            saturation=np.mean(np.abs(torque)>=task.torque_limits[None,:]-1e-7,axis=0)
            row=capture_state(task,task.steps*task.dt,torque.mean(axis=0),np.abs(torque).max(axis=0),
                saturation,command,targets,wheel_targets,task.dt)
            trace.append(row)
            # Success reference is simulated for all 20 seconds but video shows the first 6 seconds.
            if task.steps%2==0 and (slug!='success_reference' or row['seconds']<=6.):
                last_frame=draw_frame(task,renderer,cameras,options,row,title,seed,encoder,frames)
            if done or truncated:
                if slug!='success_reference':
                    last_frame=draw_frame(task,renderer,cameras,options,row,title,seed,encoder,frames,endpoint=True)
                for _ in range(25):encoder.send(np.asarray(last_frame))
                break
    finally:
        encoder.close();renderer.close();env.close()
    if (info['termination_reason'] or 'timeout')!=reference['reason'] or abs(row['seconds']-reference['seconds'])>1e-8 or bool(info['combined_success'])!=reference['success']:
        raise RuntimeError('Rendered episode does not reproduce independent assessment')
    report=dict(title=title,slug=slug,seed=seed,offset_m=reference['offset_m'],
        checkpoint_sha256=sha(RUN/'best/best_model.zip'),model_sha256=sha(xml),
        simulation_only=True,rendering_colors_only=True,reset=reset,matched_assessment=True,
        success=bool(info['combined_success']),reason=info['termination_reason'] or 'timeout',
        seconds=row['seconds'],getup_seconds=info['getup_seconds'],
        joint_names=config['control']['leg_joints']+config['control']['wheel_joints'],
        joint_axes_world_at_reset=axes.tolist(),joint_axis_lateral_sign=axis_sign.tolist(),
        peak_torque_Nm=info['peak_torque_Nm'],over_rated_seconds=info['over_rated_seconds'],trace=trace)
    (OUT/(slug+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/(slug+'.csv')).open('w',newline='') as f:
        fields=['seconds','pitch_deg','roll_deg','q_rad','qvel_rad_s','targets_rad','mean_torque_Nm','wheel_contacts','spring_mm']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in trace)
    choices=[0.,.6,1.02,1.5,2.,row['seconds'] if slug!='success_reference' else 3.]
    chosen=[]
    for moment in choices:
        matching=min(frames,key=lambda item:abs(item[0]-moment)) if frames else None
        if matching and matching not in chosen: chosen.append(matching)
    # The final actual failure frame is retained even if it falls between capture times.
    if slug!='success_reference': chosen=chosen[:5]+[(row['seconds'],last_frame)]
    sheet=Image.new('RGB',(1280,570*len(chosen)), 'white')
    for i,(_,frame) in enumerate(chosen):sheet.paste(frame,(0,570*i))
    sheet.save(OUT/(slug+'_frames.jpg'),quality=90)
    print(json.dumps({k:v for k,v in report.items() if k not in ('trace','reset')},ensure_ascii=False),flush=True)
    return report


def make_plot(reports):
    matplotlib.rcParams['font.family']=FontProperties(fname=FONT).get_name()
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    matplotlib.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(6,3,figsize=(17,17),constrained_layout=True)
    for column,report in enumerate(reports):
        rows=[r for r in report['trace'] if r['seconds']<=min(report['seconds'],6.)]
        t=np.array([r['seconds'] for r in rows])
        q=np.degrees([r['q_rad'][:4] for r in rows])
        torque=np.array([r['mean_torque_Nm'][:4] for r in rows])
        axis_sign=np.array(report['joint_axis_lateral_sign'])
        dq=(q-q[0])*axis_sign
        axes[0,column].set_title(f'{report["title"]} / seed={report["seed"]}\n终止 {report["seconds"]:.2f}s / '+('成功' if report['success'] else report['reason']))
        axes[0,column].plot(t,[r['roll_deg'] for r in rows],label='横滚',color='#9f234b')
        axes[0,column].plot(t,[r['pitch_deg'] for r in rows],label='俯仰',color='#667085')
        axes[0,column].axhline(np.degrees(.35),ls=':',color='#9f234b',alpha=.5)
        axes[0,column].axhline(-np.degrees(.35),ls=':',color='#9f234b',alpha=.5)
        for rowindex,j in [(1,0),(2,1)]:
            axes[rowindex,column].plot(t,dq[:,j],color=BLUE,label='左侧')
            axes[rowindex,column].plot(t,dq[:,j+2],color=ORANGE,label='右侧')
        for j,color,label in [(0,BLUE,'左前'),(1,BLUE,'左后'),(2,ORANGE,'右前'),(3,ORANGE,'右后')]:
            axes[3,column].plot(t,torque[:,j]*axis_sign[j],color=color,ls='-' if j%2==0 else '--',label=label)
        axes[3,column].axhline(20,ls=':',color='#888888')
        axes[3,column].axhline(-20,ls=':',color='#888888')
        for j,color,label in [(0,BLUE,'左轮'),(1,ORANGE,'右轮')]:
            axes[4,column].plot(t,[r['wheel_z_mm'][j] for r in rows],color=color,label=label)
            axes[5,column].step(t,[float(r['wheel_contacts'][j])+(1.3 if j==0 else 0) for r in rows],where='post',color=color,label=label)
        axes[5,column].set_yticks([0,1,1.3,2.3],['右离地','右接地','左离地','左接地'])
        for rowindex in range(6):
            ax=axes[rowindex,column]
            for boundary in (.6,1.02,2.):
                if boundary<=t[-1]:ax.axvline(boundary,color='#94a3b8',ls=':',alpha=.55)
            ax.grid(alpha=.18);ax.legend(fontsize=8,loc='best');ax.set_xlabel('仿真时间 / 秒')
    for i,label in enumerate(['姿态角 / 度','前关节相对初态转角 / 度','后关节相对初态转角 / 度',
        '控制周期平均力矩 / N·m','轮心高度 / mm','轮地接触（分行显示）']):axes[i,0].set_ylabel(label)
    fig.suptitle('相同困难趴姿：左右腿动作与失败过程\n蓝色左侧、橙色右侧；关节转角/力矩按同一底盘横向轴对齐；虚线为 0.6s 峰值窗、1.02s PD 切换、2s 相位饱和。',fontsize=15)
    fig.savefig(OUT/'bilateral_diagnostics.png',dpi=130)
    fig.savefig(OUT/'bilateral_diagnostics.pdf')
    plt.close(fig)


def summarize(report):
    rows=report['trace'];t=np.array([r['seconds'] for r in rows])
    sign=np.array(report['joint_axis_lateral_sign'])
    q=np.array([r['q_rad'][:4] for r in rows]);dq=(q-q[0])*sign
    diff=np.degrees(dq[:,:2]-dq[:,2:])
    def first(predicate):
        matches=[r for r in rows if predicate(r)]
        return matches[0]['seconds'] if matches else None
    def sample(moment):
        i=int(np.argmin(abs(t-moment)))
        r=rows[i]
        return dict(seconds=r['seconds'],roll_deg=r['roll_deg'],pitch_deg=r['pitch_deg'],
            left_minus_right_front_rear_deg=diff[i].tolist(),
            q_deg=np.degrees(r['q_rad'][:4]).tolist(),torque_Nm=r['mean_torque_Nm'][:4],
            wheel_z_mm=r['wheel_z_mm'],wheel_contacts=r['wheel_contacts'],spring_mm=r['spring_mm'],
            acquired=r['acquired'],hip_height_mm=r['hip_height_mm'],phase=r['policy_phase'])
    return dict(slug=report['slug'],title=report['title'],seed=report['seed'],reason=report['reason'],
        seconds=report['seconds'],success=report['success'],joint_axis_lateral_sign=sign.tolist(),
        first_roll_over_5deg=first(lambda r: abs(r['roll_deg'])>5),
        first_roll_over_10deg=first(lambda r: abs(r['roll_deg'])>10),
        max_abs_front_rear_lr_delta_deg=np.abs(diff).max(axis=0).tolist(),
        samples=[sample(moment) for moment in (0.,.6,1.02,1.5,2.,report['seconds'])])


def main():
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=True)
    assessment=json.loads((RUN/'phase3_assessment.json').read_text())
    if sha(RUN/'best/best_model.zip')!=assessment['selected_checkpoint_sha256']:
        raise RuntimeError('Checkpoint mismatch')
    validate_artifacts(CFG)
    policy=PPO.load(RUN/'best/best_model.zip',device='cpu')
    reports=[]
    for slug,seed,title in CASES:
        reference=next(r for r in assessment['episodes'] if r['model']=='selected' and r['split']=='test'
            and r['offset_m']<.335 and r['seed']==seed)
        reports.append(simulate(slug,seed,title,reference,policy))
    make_plot(reports)
    summary=[summarize(r) for r in reports]
    (OUT/'review_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
