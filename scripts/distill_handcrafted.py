"""Direct imitation distillation: handcrafted expert -> PPO policy.

Expert (validated 10/10 success on the 208 mm folded pose):
  leg action  a[:4] = clip(1.0 * (nominal_legs - q) / 0.6, -1, 1)
  wheel action a[4:] = clip(185.37 * pitch + 12.38 * pitch_rate, -10, 10) / 10
The env applies wheel_action_sign itself, so no manual sign here.

Standing trajectories are labelled by the initial policy (self-imitation) so
behaviour cloning does not forget the already working standing controller.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source"))
import numpy as np, torch
from stable_baselines3 import PPO
from climber_legged_rl.mujoco_env import make_mujoco_env
from climber_legged_rl.mujoco_model import ROOT, load_config

LEG_GAIN = 1.0
WHEEL_KP = 185.3709835995987
WHEEL_KD = 12.381648013013905
WHEEL_KPOS = -22.98302737207376
WHEEL_KVEL = 46.682530363210674


def expert_action(env):
    q = env.data.qpos[env.q_indices[:4]]
    leg_a = np.clip(LEG_GAIN * (env.nominal_legs - q) / env.dynamic['incremental_target_scale_rad'], -1, 1)
    pitch, _ = env._attitude()
    angular, linear, _ = env._kinematics()
    displacement, _ = env._hold_errors()
    local_x = float((env._rotation().T @ np.r_[displacement, 0.])[0])
    fwd = float(np.clip(WHEEL_KP * pitch + WHEEL_KD * angular[1]
                        + WHEEL_KPOS * local_x + WHEEL_KVEL * linear[0], -10, 10))
    w = float(np.clip(fwd / env.control['wheel_action_scale_rad_s'], -1, 1))
    return np.r_[leg_a, w, w]


def collect(env, student, standing_label_policy, n_folded, n_standing, seconds=20.0,
            folded_weight=2.0, noise=0.0, seed=0):
    """Folded rollouts come from the expert (student=None) or the student (DAgger);
    every folded state is labelled by the expert. Standing states are labelled by
    the known-good standing policy so behaviour cloning keeps that skill."""
    rng = np.random.default_rng(seed)
    xs, ys, ws, report = [], [], [], []
    for i in range(n_folded):
        obs, _ = env.reset(seed=50000 + i, options={'reset_mode': 'folded'})
        info = None
        for _ in range(round(seconds / env.dt)):
            label = expert_action(env)
            if student is not None:
                action, _ = student.predict(obs, deterministic=True)
                if noise:
                    action = np.clip(np.asarray(action, dtype=float) + rng.normal(0., noise, 6), -1., 1.)
            else:
                action = label
            xs.append(obs.copy()); ys.append(label.copy())
            ws.append(folded_weight if not env.acquired else 1.0)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc: break
        report.append(dict(mode='folded', seed=50000 + i, acquired=bool(env.acquired),
                           success=bool(info.get('combined_success')),
                           seconds=env.steps * env.dt, reason=info.get('termination_reason')))
    for i in range(n_standing):
        obs, _ = env.reset(seed=60000 + i, options={'reset_mode': 'standing'})
        info = None
        for _ in range(round(seconds / env.dt)):
            label, _ = standing_label_policy.predict(obs, deterministic=True)
            xs.append(obs.copy()); ys.append(np.asarray(label, dtype=float))
            ws.append(1.0)
            obs, r, term, trunc, info = env.step(label)
            if term or trunc: break
        report.append(dict(mode='standing', seed=60000 + i, success=bool(info.get('combined_success')),
                           seconds=env.steps * env.dt, reason=info.get('termination_reason')))
    folded_ok = sum(1 for r in report if r['mode'] == 'folded' and r['success'])
    standing_ok = sum(1 for r in report if r['mode'] == 'standing' and r['success'])
    print(f"COLLECT folded_success={folded_ok}/{n_folded} standing_success={standing_ok}/{n_standing}", flush=True)
    print("COLLECT detail: " + json.dumps(report), flush=True)
    return xs, ys, ws


def leg_augmentation(env, x, n, seed):
    """The rising leg channel is a known saturating proportional map of the four leg
    angles, so cover the whole joint box instead of only visited states. Without this
    the actor extrapolates badly as soon as the rise drifts off the demonstration path.
    Observation layout: indices 8..11 hold qpos(leg) - nominal_legs."""
    rng = np.random.default_rng(seed)
    base = np.asarray(x, dtype=np.float32)[rng.integers(len(x), size=n)]
    low, high = env.leg_low, env.leg_high
    nominal = env.nominal_legs
    q = np.where(rng.random((n, 4)) < 0.7,
                 np.clip(nominal + rng.normal(0., 0.35, (n, 4)), low, high),
                 rng.uniform(low, high, (n, 4)))
    base[:, 8:12] = q - nominal
    y = np.clip((nominal - q)/env.dynamic['incremental_target_scale_rad'], -1., 1.)
    return base, y


def fit(policy, x, y, w, updates, batch_size, lr, seed, symmetry=10.0, leg_aug=None, leg_weight=5.0):
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=policy.device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=policy.device)
    p = np.asarray(w, dtype=float)
    # Emphasise states where the expert saturates: those carry the closed-chain
    # branch-switch corrections that plain regression under-weights.
    p = p * (1.0 + 3.0 * (np.abs(np.asarray(y)) > 0.5).mean(axis=1))
    p /= p.sum()
    opt = torch.optim.Adam(policy.policy.parameters(), lr=lr)
    if leg_aug is not None:
        xa = torch.as_tensor(np.asarray(leg_aug[0], dtype=np.float32), device=policy.device)
        ya = torch.as_tensor(np.asarray(leg_aug[1], dtype=np.float32), device=policy.device)
    t0 = time.monotonic()
    for u in range(updates):
        for g in opt.param_groups: g['lr'] = lr * (1 - 0.8 * u / updates)
        idx = rng.choice(len(xt), size=batch_size, p=p)
        # This policy is a DiagGaussian (symmetric Box action space, no tanh squash), so
        # predict() returns the clipped raw mean: regress that mean directly.
        m = policy.policy.get_distribution(xt[idx]).distribution.mean
        loss = torch.mean((m - yt[idx]) ** 2)
        # Keep the two wheel commands identical: asymmetric wheels yaw the robot and
        # violate the get-up heading / yaw-rate acquisition gate.
        loss = loss + symmetry * torch.mean((m[:, 4] - m[:, 5]) ** 2)
        if leg_aug is not None:
            j = rng.integers(len(xa), size=batch_size // 2)
            ma = policy.policy.get_distribution(xa[j]).distribution.mean
            loss = loss + leg_weight * torch.mean((ma[:, :4] - ya[j]) ** 2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0); opt.step()
        if (u + 1) % 500 == 0 or u + 1 == updates:
            rem = (time.monotonic() - t0) * (updates - u - 1) / (u + 1)
            print(f"IMIT update={u+1}/{updates} loss={float(loss):.6f} remaining={rem:.0f}s", flush=True)


def short_eval(env, policy, mode, n=20, seconds=20.0, seed_base=52000):
    succ = acq = 0; lens = []; reasons = []
    for i in range(n):
        obs, _ = env.reset(seed=seed_base + i, options={'reset_mode': mode})
        info = None
        for _ in range(round(seconds / env.dt)):
            a, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        succ += int(bool(info.get('combined_success')))
        acq += int(bool(info.get('getup_acquired')))
        lens.append(env.steps * env.dt)
        reasons.append(info.get('termination_reason') or ('acquired' if info.get('getup_acquired') else 'timeout'))
    return succ, acq, float(np.mean(lens)), sorted(set(reasons))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('configs/robot/cad_balance_spring208_imitation.yaml'))
    ap.add_argument('--initial-checkpoint', type=Path, required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--updates', type=int, default=8000)
    ap.add_argument('--dagger-rounds', type=int, default=2)
    ap.add_argument('--dagger-updates', type=int, default=3000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--learning-rate', type=float, default=0.0003)
    ap.add_argument('--folded-episodes', type=int, default=6)
    ap.add_argument('--standing-episodes', type=int, default=4)
    args = ap.parse_args()
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    folder = ROOT / cfg['experiment']['output_directory']
    env = make_mujoco_env(folder / 'robot_experiment.xml', cfg_path)
    policy = PPO.load(str(args.initial_checkpoint), device='cpu')
    run_dir = ROOT / 'outputs/mujoco_runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"IMITATION_DIRECTORY={run_dir}", flush=True)

    s, a, l, r = short_eval(env, policy, 'folded', 20)
    print(f"BASELINE folded success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)
    s, a, l, r = short_eval(env, policy, 'standing', 20)
    print(f"BASELINE standing success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)

    x, y, w = collect(env, None, policy, args.folded_episodes, args.standing_episodes)
    leg_aug = leg_augmentation(env, x, 40000, seed=777)
    fit(policy, x, y, w, args.updates, args.batch_size, args.learning_rate, seed=202611, leg_aug=leg_aug)
    policy.save(run_dir / 'policy_after_imitation.zip')
    s, a, l, r = short_eval(env, policy, 'folded', 20)
    print(f"AFTER_IMIT folded success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)
    s, a, l, r = short_eval(env, policy, 'standing', 20)
    print(f"AFTER_IMIT standing success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)

    for stage in range(1, args.dagger_rounds + 1):
        x2, y2, w2 = collect(env, policy, policy, args.folded_episodes, args.standing_episodes,
                             noise=0.05, seed=1000 + stage)
        x.extend(x2); y.extend(y2); w.extend(w2)
        leg_aug = leg_augmentation(env, x, 40000, seed=777 + stage)
        fit(policy, x, y, w, args.dagger_updates, args.batch_size, args.learning_rate,
            seed=202611 + stage, leg_aug=leg_aug)
        policy.save(run_dir / f'policy_after_dagger_{stage}.zip')
        s, a, l, r = short_eval(env, policy, 'folded', 20)
        print(f"AFTER_DAGGER_{stage} folded success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)
        s, a, l, r = short_eval(env, policy, 'standing', 20)
        print(f"AFTER_DAGGER_{stage} standing success={s}/20 acquired={a} mean_s={l:.2f} reasons={r}", flush=True)
    env.close()


if __name__ == '__main__':
    main()
