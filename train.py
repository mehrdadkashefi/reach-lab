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

from utils import fig_reaches, fig_diagnostics, fig_learning_curve


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
p.add_argument("--smooth-w", type=float, default=1e-1, help="weight on the control-smoothness penalty")
p.add_argument("--jerk-w", type=float, default=1e-8,
               help="weight on the hand-path minimum-jerk penalty")
p.add_argument("--hold-w", type=float, default=1e-1, 
               help="weight on speed during stay times.")
p.add_argument("--rate-smooth-w", type=float, default=1e-2,
               help="weight on RNN hidden-activity temporal smoothness")
p.add_argument("--obs-noise", type=float, default=0.0,
               help="std of Gaussian noise on observed body state (vision fingertip + proprio); 0 = off")
p.add_argument("--neural-noise", type=float, default=0.0,
               help="std of Gaussian noise injected into the RNN hidden state each step; 0 = off")
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
mae = nn.L1Loss()

if args.track:
    import wandb
    wandb.init(project=args.wandb_project, name=os.path.basename(run_dir), config=vars(args))

# fixed eval set (same targets each snapshot)
torch.manual_seed(123)
num_eval = 30
eval_theta0, eval_inp, eval_desired, eval_perturbation, eval_timestamps = task.make_batch(num_eval)
torch.manual_seed(args.seed)

# ----------------------------------------------------------------------------- train
loss_hist, snapshots = [], []
for i in tqdm(range(args.n_batch)):
    theta0, inp, desired, perturbation, ts = task.make_batch(args.batch_size)
    states = eff.rollout(controller, theta0, inp, perturbation,
                         obs_noise=args.obs_noise, neural_noise=args.neural_noise)

    # calculate losses
    pos_loss    = mae(states.pos, desired)
    effort_loss = states.action.pow(2).mean()
    smooth_loss = (states.action[:, 1:] - states.action[:, :-1]).pow(2).mean()

    # minimum-jerk
    jerk = (states.pos[:, 3:] - 3 * states.pos[:, 2:-1]
            + 3 * states.pos[:, 1:-2] - states.pos[:, :-3]) / (args.dt ** 3)
    jerk_loss = jerk.pow(2).mean()
    # hidden-activity smoothness
    rate_smooth_loss = (states.hidden[:, 1:] - states.hidden[:, :-1]).pow(2).mean()

    # Speed penalty before go cue
    T  = desired.shape[1]
    tg = torch.arange(T, device=desired.device).unsqueeze(0)         # (1, T)
    hold = ((tg < ts['move_start'].unsqueeze(1)) |                    # before go
            (tg >= ts['final_start'].unsqueeze(1))).float()          # after reach
    hold = torch.maximum(hold, ts['is_no_go'].float().unsqueeze(1))  # all of no-go
    hold_loss = (states.vel.pow(2).sum(-1) * hold).sum() / hold.sum().clamp(min=1.0)
    

    loss = (pos_loss + args.effort_w * effort_loss + args.smooth_w * smooth_loss
            + args.jerk_w * jerk_loss + args.rate_smooth_w * rate_smooth_loss + args.hold_w * hold_loss)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    opt.step()
    loss_hist.append(loss.item())

    if args.track:
        contrib = {'loss_tot':loss, 'pos': pos_loss, 'effort': args.effort_w * effort_loss,
           'smooth': args.smooth_w * smooth_loss, 'jerk': args.jerk_w * jerk_loss,
           'rate_smooth': args.rate_smooth_w * rate_smooth_loss, 'hold': args.hold_w * hold_loss}
        wandb.log({f'{k}': v.item() for k, v in contrib.items()}, step=i)

    if (i + 1) % args.snap_every == 0:
        controller.eval()
        with torch.no_grad():
            ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
        controller.train()
        err = 100 * (ev.pos[:, -1, :] - eval_desired[:, -1, :]).norm(dim=1).mean().item()
        if args.track:
            # randomly select num_eval_to_plot indices from eval_desiered
            fr = fig_reaches(ev.pos, eval_desired, title=f"reaches @ batch {i+1} (err {err:.1f} cm)")
            fd = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"diagnostics @ batch {i+1}",  num_trial=5)
            wandb.log({"eval/endpoint_error_cm": err,
                       "eval/reaches": wandb.Image(fr),
                       "eval/diagnostics": wandb.Image(fd)}, step=i)
            plt.close(fr); plt.close(fd)

print(f"Training complete.  start loss {loss_hist[0]:.5f} -> final loss {loss_hist[-1]:.5f}")
torch.save(controller.state_dict(), out(f"controller_{args.effector}_{args.arch}.pt"))


# ----------------------------------------------------------------------------- final plots
tag = f"{args.effector}/{args.arch}"
lc = fig_learning_curve(loss_hist, f'learning curve ({tag})')
# Final Evaluation
controller.eval()
with torch.no_grad():
    ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
td = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"Sample trials ({tag}; dotted line = go cue)", num_trial=5)
lc.savefig(out('learning_curve.png'), dpi=120, bbox_inches='tight')
td.savefig(out('trial_diagnostics.png'), dpi=120, bbox_inches='tight')
print(f"\nsaved plots + controller to {run_dir}")

if args.track:
    wandb.log({"learning_curve": wandb.Image(lc),
               "trial_diagnostics": wandb.Image(td)})
    wandb.finish()
plt.close('all')