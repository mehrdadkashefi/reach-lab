import argparse
import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from effectors import make_effector
from tasks import make_task
from controllers import GRUController, ModularGRU


def list_of_float(s): return [float(x) for x in s.split(',')]
def list_of_int(s):   return [int(x) for x in s.split(',')]

p = argparse.ArgumentParser()
# --- main / training ---
p.add_argument("--effector", choices=["point_mass", "arm_torque", "arm26"], default="arm26")
p.add_argument("--task", choices=["delayed_reach", "delayed_reach_posture"], default="delayed_reach")
p.add_argument("--arch", choices=["gru", "modular"], default="gru")
p.add_argument("--n-batch", type=int, default=600)
p.add_argument("--batch-size", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--effort-w", type=float, default=1e-3)
p.add_argument("--snap-every", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--track", action="store_true", help="log metrics to Weights & Biases")
p.add_argument("--wandb-project", default="arm-rnn")
# --- effector overrides (kwargs) ---
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--vis-delay-ms", type=float, default=70)
p.add_argument("--pro-delay-ms", type=float, default=25)
# --- task overrides (kwargs) ---
p.add_argument("--steps", type=int, default=100, help="delayed_reaching episode length")
p.add_argument("--go-range", type=list_of_int, default=[20, 50], help="delayed_reaching go window")
# delayed_reach_posture timing (ms); None -> task defaults
p.add_argument("--init-range-ms",  type=list_of_int, default=None)
p.add_argument("--delay-range-ms", type=list_of_int, default=None)
p.add_argument("--move-ms",        type=int,         default=None)
p.add_argument("--final-range-ms", type=list_of_int, default=None)
p.add_argument("--null-value",     type=float,       default=None)
p.add_argument("--final-input",    choices=["null", "target"], default=None)
p.add_argument("--prob-no-go",     type=float,       default=None, help="fraction of no-go trials")
# --- controller config ---
p.add_argument("--hidden-dim", type=int, default=128, help="baseline gru hidden size")
# modular overrides: leave as None to use ModularGRU's own defaults
p.add_argument("--module-size",  type=list_of_int,   default=None)
p.add_argument("--vision-mask",  type=list_of_float, default=None)
p.add_argument("--proprio-mask", type=list_of_float, default=None)
p.add_argument("--task-mask",    type=list_of_float, default=None)
p.add_argument("--output-mask",  type=list_of_float, default=None)
p.add_argument("--spectral-scaling", type=float, default=None)
p.add_argument("--connectivity", type=list_of_float, default=None,
               help="flattened 3x3 (row = receiver); overrides the module default")
args = p.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")
torch.manual_seed(args.seed)

# ----------------------------------------------------------------------------- run directory
run_dir = os.path.join(os.getcwd(), "experiments",
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(run_dir, exist_ok=True)
def out(name): return os.path.join(run_dir, name)
with open(out("config.json"), "w") as f:
    json.dump(vars(args), f, indent=2, default=str)
print(f"saving results to {run_dir}")

# ----------------------------------------------------------------------------- effector + task
eff = make_effector(args.effector, dt=args.dt,
                    vis_delay_ms=args.vis_delay_ms, pro_delay_ms=args.pro_delay_ms).to(device)

if args.task == "delayed_reach":
    rk = {} if args.prob_no_go is None else {'prob_no_go': args.prob_no_go}
    task = make_task(args.task, eff, steps=args.steps, go_range=args.go_range, **rk)
elif args.task == "delayed_reach_posture":
    tk = {}
    if args.init_range_ms  is not None: tk['init_range_ms']  = tuple(args.init_range_ms)
    if args.delay_range_ms is not None: tk['delay_range_ms'] = tuple(args.delay_range_ms)
    if args.move_ms        is not None: tk['move_ms']        = args.move_ms
    if args.final_range_ms is not None: tk['final_range_ms'] = tuple(args.final_range_ms)
    if args.null_value     is not None: tk['null_value']     = args.null_value
    if args.final_input    is not None: tk['final_input']    = args.final_input
    if args.prob_no_go     is not None: tk['prob_no_go']     = args.prob_no_go
    task = make_task(args.task, eff, **tk)
else:
    raise ValueError(f"Invalid task: {args.task}")
print(f"effector: {eff.name}  (input_dim={eff.input_dim}, output_dim={eff.output_dim}, "
      f"vis_d={eff.vis_d}, pro_d={eff.pro_d})")
print(f"task: {task.name}  (episode {task.steps} steps = {task.steps * args.dt:.2f} s)")

# ----------------------------------------------------------------------------- controller
if args.arch == "gru":
    controller = GRUController(eff.input_dim, hidden_dim=args.hidden_dim,
                               output_dim=eff.output_dim, out_bias=eff.out_bias)
    print(f"controller: single GRU, {args.hidden_dim} units")
else:
    # only pass overrides the user actually set; otherwise ModularGRU defaults apply
    mod_kwargs = {}
    if args.module_size     is not None: mod_kwargs['module_sizes']     = args.module_size
    if args.vision_mask     is not None: mod_kwargs['vision_mask']      = args.vision_mask
    if args.proprio_mask    is not None: mod_kwargs['proprio_mask']     = args.proprio_mask
    if args.task_mask       is not None: mod_kwargs['task_mask']        = args.task_mask
    if args.output_mask     is not None: mod_kwargs['output_mask']      = args.output_mask
    if args.spectral_scaling is not None: mod_kwargs['spectral_scaling'] = args.spectral_scaling
    if args.connectivity    is not None: mod_kwargs['connectivity']     = np.array(args.connectivity).reshape(3, 3)
    controller = ModularGRU(eff.input_dim, eff.output_dim, eff.input_layout,
                            out_bias=eff.out_bias, seed=args.seed, **mod_kwargs)
    di, dh, do = controller.density()
    print(f"controller: modular GRU (H={controller.hidden_dim}) | "
          f"mask density  input {di:.2f}  recurrent {dh:.2f}  output {do:.2f}")
controller = controller.to(device)

opt = torch.optim.Adam(controller.parameters(), lr=args.lr)
mse = nn.MSELoss()

# ----------------------------------------------------------------------------- wandb
if args.track:
    import wandb
    wandb.init(project=args.wandb_project, name=os.path.basename(run_dir), config=vars(args))

# fixed eval set (same targets each snapshot)
torch.manual_seed(123)
eval_theta0, eval_inp, eval_desired, eval_perturbation, eval_timestamps = task.make_batch(256)
# "reach" trials = desired start differs from desired end (handles delayed_reaching no-go too)
eval_go = (eval_desired[:, 0] - eval_desired[:, -1]).norm(dim=1) > 1e-6
go_all = torch.where(eval_go)[0].tolist()
hold_all = torch.where(~eval_go)[0].tolist()
eval_go_idx = go_all[:12]
show_idx, labels = [], []
for j in go_all[:3]:
    show_idx.append(j); labels.append('reach')
if hold_all:
    show_idx.append(hold_all[0]); labels.append('hold')
elif len(go_all) > 3:
    show_idx.append(go_all[3]); labels.append('reach')
torch.manual_seed(args.seed)


# ----------------------------------------------------------------------------- plot helpers
def fig_reaches(pos, desired, idxs, title):
    P, D = pos.detach().cpu().numpy(), desired.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    for j in idxs:
        ax.plot(P[j, :, 0], P[j, :, 1], lw=.9)
        ax.plot(D[j, 0, 0], D[j, 0, 1], 'ko', ms=3); ax.plot(D[j, -1, 0], D[j, -1, 1], 'r*', ms=8)
    ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.tight_layout()
    return fig


def fig_diagnostics(states, inp, desired, title):
    t_axis = np.arange(desired.shape[1]) * args.dt
    fig, axes = plt.subplots(len(show_idx), 3, figsize=(14, 2.7 * len(show_idx)), squeeze=False)
    for r, (idx, lab) in enumerate(zip(show_idx, labels)):
        pp = states.pos[idx].detach().cpu().numpy()
        v = states.vel[idx].detach().cpu().numpy()
        a = states.action[idx].detach().cpu().numpy()
        tgt = desired[idx, -1].detach().cpu().numpy()
        gomask = inp[idx, :, 2] > 0.5
        gt = (torch.argmax(gomask.float()).item() * args.dt) if bool(gomask.any()) else None

        ax = axes[r, 0]
        ax.plot(t_axis, pp[:, 0], 'C0', label='x'); ax.plot(t_axis, pp[:, 1], 'C1', label='y')
        ax.axhline(tgt[0], ls='--', c='C0', lw=.7); ax.axhline(tgt[1], ls='--', c='C1', lw=.7)
        if gt is not None: ax.axvline(gt, c='k', ls=':', lw=.8)
        ax.set_ylabel(f"{lab}\nposition (m)")
        if r == 0: ax.legend(fontsize=7, loc='upper right'); ax.set_title("fingertip position (dashed=target)")

        ax = axes[r, 1]
        ax.plot(t_axis, np.linalg.norm(v, axis=1), 'C3')
        if gt is not None: ax.axvline(gt, c='k', ls=':', lw=.8)
        ax.set_ylabel("speed (m/s)")
        if r == 0: ax.set_title("fingertip speed")

        ax = axes[r, 2]
        for m, mname in enumerate(eff.action_names):
            ax.plot(t_axis, a[:, m], lw=1, label=mname)
        if gt is not None: ax.axvline(gt, c='k', ls=':', lw=.8)
        if args.effector == "arm26": ax.set_ylim(-0.02, 1.0)
        ax.set_ylabel("action")
        if r == 0: ax.legend(fontsize=6, ncol=2, loc='upper right'); ax.set_title("control signal")
        if r == len(show_idx) - 1:
            for c in range(3): axes[r, c].set_xlabel("time (s)")
    fig.suptitle(title, y=1.001); fig.tight_layout()
    return fig


def fig_learning_curve(loss_hist, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist, lw=.7, alpha=.4, color='C0', label='loss')
    k = 20
    if len(loss_hist) >= k:
        ma = np.convolve(loss_hist, np.ones(k) / k, mode='valid')
        ax.plot(np.arange(k - 1, len(loss_hist)), ma, color='C1', lw=1.6, label=f'{k}-batch avg')
    ax.set_yscale('log'); ax.set_xlabel('training batch'); ax.set_ylabel('loss (log)')
    ax.legend(); ax.set_title(title); fig.tight_layout()
    return fig


def fig_progress_grid(snapshots, desired, idxs, title):
    ncol = 3
    nrow = int(np.ceil(len(snapshots) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
    D = desired.detach().cpu().numpy()
    for k, (bn, ep, err) in enumerate(snapshots):
        ax = axes[k // ncol, k % ncol]
        P = ep.numpy()
        for j in idxs:
            ax.plot(P[j, :, 0], P[j, :, 1], lw=.9)
            ax.plot(D[j, 0, 0], D[j, 0, 1], 'ko', ms=3); ax.plot(D[j, -1, 0], D[j, -1, 1], 'r*', ms=8)
        ax.set_aspect('equal'); ax.set_title(f'batch {bn}  |  err {err:.1f} cm')
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    for k in range(len(snapshots), nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')
    fig.suptitle(title); fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- train
loss_hist, snapshots = [], []
for i in tqdm(range(args.n_batch)):
    theta0, inp, desired, perturbation, _ = task.make_batch(args.batch_size)
    states = eff.rollout(controller, theta0, inp, perturbation)
    pos_loss = mse(states.pos, desired)
    effort_loss = states.action.pow(2).mean()
    loss = pos_loss + args.effort_w * effort_loss
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_hist.append(loss.item())

    if args.track:
        wandb.log({"loss": loss.item(), "pos_loss": pos_loss.item(),
                   "effort_loss": effort_loss.item()}, step=i)

    if (i + 1) % args.snap_every == 0:
        controller.eval()
        with torch.no_grad():
            ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
        controller.train()
        err = 100 * (ev.pos[:, -1, :] - eval_desired[:, -1, :]).norm(dim=1)[eval_go].mean().item()
        snapshots.append((i + 1, ev.pos.cpu(), err))
        if args.track:
            fr = fig_reaches(ev.pos, eval_desired, eval_go_idx, f"reaches @ batch {i+1} (err {err:.1f} cm)")
            fd = fig_diagnostics(ev, eval_inp, eval_desired, f"diagnostics @ batch {i+1}")
            wandb.log({"eval/endpoint_error_cm": err,
                       "eval/reaches": wandb.Image(fr),
                       "eval/diagnostics": wandb.Image(fd)}, step=i)
            plt.close(fr); plt.close(fd)

print(f"Training complete.  start loss {loss_hist[0]:.5f} -> final loss {loss_hist[-1]:.5f}")
torch.save(controller.state_dict(), out(f"controller_{args.effector}_{args.arch}.pt"))


# ----------------------------------------------------------------------------- evaluate
controller.eval()
with torch.no_grad():
    states = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
final_err = (states.pos[:, -1, :] - eval_desired[:, -1, :]).norm(dim=1)
reach_err = 100 * final_err[eval_go].mean().item()
print(f"mean endpoint error (reach trials): {reach_err:.2f} cm")
wlog = {"eval/final_reach_error_cm": reach_err}
if bool((~eval_go).any()):
    hold_err = 100 * final_err[~eval_go].mean().item()
    print(f"mean endpoint error (hold trials):  {hold_err:.2f} cm")
    wlog["eval/final_hold_error_cm"] = hold_err
if args.track:
    wandb.log(wlog)

print("\n--- sample trials ---")
for idx, lab in zip(show_idx, labels):
    s = eval_desired[idx, 0].cpu().numpy()
    t = eval_desired[idx, -1].cpu().numpy()
    f = states.pos[idx, -1].cpu().numpy()
    sp = states.vel[idx].norm(dim=1)
    print(f"[{lab:5s}] start=({s[0]:+.2f},{s[1]:+.2f})  target=({t[0]:+.2f},{t[1]:+.2f})  "
          f"final=({f[0]:+.2f},{f[1]:+.2f})  err={100*np.linalg.norm(f-t):5.1f}cm  "
          f"peak_speed={sp.max():.2f}m/s")


# ----------------------------------------------------------------------------- final plots
tag = f"{args.effector}/{args.arch}"
lc = fig_learning_curve(loss_hist, f'learning curve ({tag})')
pg = fig_progress_grid(snapshots, eval_desired, eval_go_idx,
                       f'Training progress ({tag}): reaches every {args.snap_every} batches')
td = fig_diagnostics(states, eval_inp, eval_desired, f"Sample trials ({tag}; dotted line = go cue)")
lc.savefig(out('learning_curve.png'), dpi=120, bbox_inches='tight')
pg.savefig(out('training_progress.png'), dpi=120, bbox_inches='tight')
td.savefig(out('trial_diagnostics.png'), dpi=120, bbox_inches='tight')
print(f"\nsaved plots + controller to {run_dir}")

if args.track:
    wandb.log({"learning_curve": wandb.Image(lc),
               "training_progress": wandb.Image(pg),
               "trial_diagnostics": wandb.Image(td)})
    wandb.finish()
plt.close('all')