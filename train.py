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
from tasks import make_task, TASKS, task_input_channels
from controllers import GRUController, ModularGRU

from utils import fig_reaches, fig_diagnostics, fig_learning_curve


def list_of_float(s): return [float(x) for x in s.split(',')]
def list_of_int(s):   return [int(x) for x in s.split(',')]

p = argparse.ArgumentParser()
# --- main / training ---
p.add_argument("--effector", choices=["point_mass", "arm_torque", "arm26"], default="arm26")
p.add_argument("--task", choices=["delayed_reach", "delayed_reach_posture", "horizon_sequence",
                                  "hold_posture_pulse", "hold_posture_ramp"],
               default="delayed_reach")
p.add_argument("--desired-profile", choices=["step", "min_jerk"], default="step",
               help="target trajectory the position loss regresses against: 'step' (jump to "
                    "target at go) or 'min_jerk' (straight-line minimum-jerk reach)")
p.add_argument("--arch", choices=["gru", "modular"], default="gru")
# --- effector overrides (kwargs) ---
p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--vis-delay-ms", type=float, default=70)
p.add_argument("--pro-delay-ms", type=float, default=25)
# learning parameters
p.add_argument("--n-batch", type=int, default=600)
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-3)
# Loss weights
p.add_argument("--w-loss-pos", type=float, default=1)
p.add_argument("--w-loss-jerk", type=float, default=1e4)
p.add_argument("--w-loss-action", type=float, default=0.5)
p.add_argument("--w-loss-action-diff", type=float, default=3e-3)
p.add_argument("--w-loss-hidden", type=float, default=3e-4)
p.add_argument("--w-loss-hidden-diff", type=float, default=3e-2)
# extra emphasis on the hold epochs (before go = start posture, and after movement = final hold)
p.add_argument("--w-loss-hold-pos", type=float, default=5.0,
               help="extra position (L1) loss applied only during the start + final hold epochs")
p.add_argument("--w-loss-hold-vel", type=float, default=1.0,
               help="velocity (squared) loss applied only during the start + final hold epochs")
# urgency: push the network to reach fast without shortening the move window
p.add_argument("--w-loss-urgency", type=float, default=0.0,
               help="weight on a time-rising penalty of distance-to-final-target after the go "
                    "cue; encourages a fast, early-peaking reach. 0 = off (default)")
p.add_argument("--urgency-tau-ms", type=float, default=300.0,
               help="time constant (ms) of the urgency ramp; smaller = more urgent / faster reach")
# noise in traininz
p.add_argument("--obs-noise", type=float, default=0.1,
               help="std of Gaussian noise on observed body state (vision fingertip + proprio); 0 = off")
p.add_argument("--neural-noise", type=float, default=0.05,
               help="std of Gaussian noise injected into the RNN hidden state each step; 0 = off")
p.add_argument("--action-noise", type=float, default=0.0,
               help="std of Gaussian noise added to the motor command sent to the plant each "
                    "step (native action units: force/torque, or muscle excitation for arm26); 0 = off")
# random training perturbations (applied to the plant, not observed) -> encourage posture / state feedback
p.add_argument("--perturb-prob", type=float, default=0.0,
               help="fraction of training trials that get a random force/torque pulse; 0 = off")
p.add_argument("--perturb-mag", type=float, default=0.0,
               help="max perturbation magnitude (N for point mass, N.m joint torque for arms); "
                    "each hit trial samples uniform [0, mag]")
p.add_argument("--perturb-dur-ms", type=float, default=100,
               help="duration (ms) of each perturbation pulse")
p.add_argument("--snap-every", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--track", action="store_true", help="log metrics to Weights & Biases")
p.add_argument("--wandb-project", default="arm-rnn")
# --- task overrides (kwargs) ---
p.add_argument("--unified-input", action="store_true",
               help="emit the 10-wide unified instruction stream (3 target slots + go) for "
                    "single-target tasks, so one network can train on all tasks with a fixed "
                    "input_dim. No effect on horizon_sequence (already 10-wide).")
# hold_posture_pulse / hold_posture_ramp
p.add_argument("--onset-range-ms", type=list_of_int, default=None,
               help="hold tasks: random bump onset window (ms), default 500,1000")
p.add_argument("--force-range-n",  type=list_of_float, default=None,
               help="hold tasks: bump force magnitude range (N), default 1,5")
p.add_argument("--dur-range-ms",   type=list_of_int, default=None,
               help="hold_posture_pulse: force pulse duration range (ms), default 100,200")
p.add_argument("--ramp-range-ms",  type=list_of_int, default=None,
               help="hold_posture_ramp: force ramp-up duration range (ms), default 200,600")
p.add_argument("--hold-after-ramp", action="store_true",
               help="hold_posture_ramp: sustain max force after the ramp instead of releasing")
p.add_argument("--center-frac",    type=float, default=None,
               help="hold tasks: fraction of the joint/pos range to sample the start from "
                    "(default 0.4 = central 40%%, keeps the arm clear of its limits)")
p.add_argument("--catch-prob",     type=float, default=None,
               help="hold tasks: fraction of catch trials with no external bump (default 0.3)")
p.add_argument("--go-pulse-ms", type=float, default=150,
               help="go-cue pulse duration (ms): the go input goes to 1 at go onset and back "
                    "to 0 after this long; 0 or negative = sustained cue (old step behaviour)")
p.add_argument("--steps", type=int, default=100, help="delayed_reaching episode length")
p.add_argument("--go-range", type=list_of_int, default=[20, 50], help="delayed_reaching go window")
# delayed_reach_posture timing (ms); None -> task defaults
p.add_argument("--init-range-ms",  type=list_of_int, default=None)
p.add_argument("--delay-range-ms", type=list_of_int, default=None)
p.add_argument("--move-ms",        type=int,         default=None)
p.add_argument("--final-range-ms", type=list_of_int, default=None)
p.add_argument("--final-input",    choices=["null", "target"], default=None)
p.add_argument("--prob-no-go",     type=float,       default=None, help="fraction of no-go trials")
# horizon_sequence (also reuses --init/--delay/--final-range-ms and --prob-no-go above)
p.add_argument("--n-reaches",      type=int,         default=None, help="sequence length (default 7)")
p.add_argument("--dwell-range-ms", type=list_of_int, default=None,
               help="per-reach segment duration range (move + hold at target), ms")
p.add_argument("--horizon-probs",  type=list_of_float, default=None,
               help="P(horizon = 1, 2, 3); default uniform")
p.add_argument("--prob-no-go-reach", type=float,     default=None,
               help="fraction of trials where one random capture gets no go pulse "
                    "(hold at the captured target; sequence aborts)")
# --- controller config ---
p.add_argument("--hidden-dim", type=int, default=128, help="baseline gru hidden size")
# modular overrides: leave as None to use ModularGRU's own defaults
p.add_argument("--module-size",  type=list_of_int,   default=[256, 256, 32])
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
                    vis_delay_ms=args.vis_delay_ms, pro_delay_ms=args.pro_delay_ms,
                    task_dim=task_input_channels(args.task, args.unified_input)).to(device)

perturb_kw = {'perturb_prob': args.perturb_prob, 'perturb_mag': args.perturb_mag,
              'perturb_dur_ms': args.perturb_dur_ms, 'go_pulse_ms': args.go_pulse_ms}
if args.task == "delayed_reach":
    rk = {'desired_profile': args.desired_profile, 'unified_input': args.unified_input, **perturb_kw}
    if args.prob_no_go is not None: rk['prob_no_go'] = args.prob_no_go
    task = make_task(args.task, eff, steps=args.steps, go_range=args.go_range, **rk)
elif args.task == "delayed_reach_posture":
    tk = {'desired_profile': args.desired_profile, 'unified_input': args.unified_input, **perturb_kw}
    if args.init_range_ms  is not None: tk['init_range_ms']  = tuple(args.init_range_ms)
    if args.delay_range_ms is not None: tk['delay_range_ms'] = tuple(args.delay_range_ms)
    if args.move_ms        is not None: tk['move_ms']        = args.move_ms
    if args.final_range_ms is not None: tk['final_range_ms'] = tuple(args.final_range_ms)
    if args.final_input    is not None: tk['final_input']    = args.final_input
    if args.prob_no_go     is not None: tk['prob_no_go']     = args.prob_no_go
    task = make_task(args.task, eff, **tk)
elif args.task in ("hold_posture_pulse", "hold_posture_ramp"):
    hk = {'unified_input': args.unified_input}
    if args.onset_range_ms is not None: hk['onset_range_ms'] = tuple(args.onset_range_ms)
    if args.force_range_n  is not None: hk['force_range_n']  = tuple(args.force_range_n)
    if args.center_frac    is not None: hk['center_frac']    = args.center_frac
    if args.catch_prob     is not None: hk['catch_prob']     = args.catch_prob
    if args.task == "hold_posture_pulse":
        if args.dur_range_ms is not None: hk['dur_range_ms'] = tuple(args.dur_range_ms)
    else:
        if args.ramp_range_ms is not None: hk['ramp_range_ms'] = tuple(args.ramp_range_ms)
        if args.hold_after_ramp:          hk['hold_after_ramp'] = True
    task = make_task(args.task, eff, **hk)
elif args.task == "horizon_sequence":
    sk = {'desired_profile': args.desired_profile, **perturb_kw}
    if args.n_reaches        is not None: sk['n_reaches']        = args.n_reaches
    if args.init_range_ms    is not None: sk['init_range_ms']    = tuple(args.init_range_ms)
    if args.delay_range_ms   is not None: sk['delay_range_ms']   = tuple(args.delay_range_ms)
    if args.dwell_range_ms   is not None: sk['dwell_range_ms']   = tuple(args.dwell_range_ms)
    if args.final_range_ms   is not None: sk['final_range_ms']   = tuple(args.final_range_ms)
    if args.horizon_probs    is not None: sk['horizon_probs']    = tuple(args.horizon_probs)
    if args.prob_no_go       is not None: sk['prob_no_go']       = args.prob_no_go
    if args.prob_no_go_reach is not None: sk['prob_no_go_reach'] = args.prob_no_go_reach
    task = make_task(args.task, eff, **sk)
else:
    raise ValueError(f"Invalid task: {args.task}")
if args.task == "horizon_sequence" and args.w_loss_urgency > 0:
    print("WARNING: --w-loss-urgency penalizes distance to the FINAL target after the first go, "
          "which is wrong for a reach sequence (it fights dwelling at intermediate targets); "
          "consider leaving it at 0 for horizon_sequence.")
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

# fixed eval set (same targets each snapshot); temporarily disable trial-level randomness that
# would blank the signal -- the random plant perturbation on the reach tasks, and the no-bump
# catch fraction on the hold tasks -- so the eval batch is clean and consistent. Only touches
# whichever of these the task actually has.
torch.manual_seed(123)
num_eval = 30
_saved = {}
for _attr in ('perturb_prob', 'catch_prob'):
    if hasattr(task, _attr):
        _saved[_attr] = getattr(task, _attr); setattr(task, _attr, 0.0)
eval_theta0, eval_inp, eval_desired, eval_perturbation, eval_timestamps = task.make_batch(num_eval)
for _attr, _val in _saved.items():
    setattr(task, _attr, _val)
torch.manual_seed(args.seed)


def hold_mask_from_ts(ts, T, device):
    """(batch, T) bool mask that is True during the start hold (before go) and the final hold.

    Uses the per-trial epoch boundaries returned by the task. Works for both tasks:
      - delayed_reach_posture: start hold = t < move_start, final hold = t >= final_start
      - delayed_reach:         start hold = t < go_start  (no explicit final-hold epoch)
    """
    n = ts['episode_end'].shape[0]
    tg = torch.arange(T, device=device).unsqueeze(0)                    # (1, T)
    if 'bump_onset' in ts:                                             # hold_posture_* : hold all episode
        return torch.ones(n, T, dtype=torch.bool, device=device)
    if 'move_start' in ts:                                              # delayed_reach_posture / horizon_sequence
        start_hold = tg < ts['move_start'].to(device).unsqueeze(1)
        end_hold   = tg >= ts['final_start'].to(device).unsqueeze(1)
    else:                                                               # delayed_reach
        start_hold = tg < ts['go_start'].to(device).unsqueeze(1)
        end_hold   = torch.zeros_like(start_hold)
    return (start_hold | end_hold)                                     # (n, T) bool

# ----------------------------------------------------------------------------- train
loss_hist, snapshots = [], []
best_err = float('inf')                                   # lowest eval endpoint error so far
best_path = out(f"controller_{args.effector}_{args.arch}_best.pt")
for i in tqdm(range(args.n_batch)):
    theta0, inp, desired, perturbation, ts = task.make_batch(args.batch_size)
    states = eff.rollout(controller, theta0, inp, perturbation,
                         obs_noise=args.obs_noise, neural_noise=args.neural_noise,
                         action_noise=args.action_noise)

    # position loss
    loss_pos = (states.pos - desired).abs().sum(-1).mean()
    # jerk loss
    _jerk = (states.pos[:, 3:] - 3 * states.pos[:, 2:-1]
        + 3 * states.pos[:, 1:-2] - states.pos[:, :-3])
    loss_jerk = _jerk.pow(2).sum(-1).mean()
    # action loss
    _action  = states.action
    _action_diff = _action[:, 1:] - _action[:, :-1]
    loss_action = _action.pow(2).sum(-1).mean()
    loss_action_diff =  _action_diff.pow(2).sum(-1).mean()
    # hidden acivity loss
    _hidden  = states.hidden
    _hidden_diff = _hidden[:, 1:] - _hidden[:, :-1]
    loss_hidden = _hidden.pow(2).sum(-1).mean()
    loss_hidden_diff = _hidden_diff.pow(2).sum(-1).mean()

    # --- hold-epoch losses: extra emphasis on holding still at the start posture and at the
    #     final location. hmask (batch, T) is True only during those two hold epochs; both
    #     losses are averaged over the held steps so their scale is independent of epoch length.
    T = states.pos.shape[1]
    hmask = hold_mask_from_ts(ts, T, states.pos.device).float()             # (n, T)
    hden  = hmask.sum().clamp(min=1.0)
    pos_err = (states.pos - desired).abs().sum(-1)                          # (n, T)
    vel_sq  = states.vel.pow(2).sum(-1)                                     # (n, T)
    loss_hold_pos = (pos_err * hmask).sum() / hden                         # heavier position at start+end
    loss_hold_vel = (vel_sq  * hmask).sum() / hden                         # be still at start+end

    # --- urgency: penalize still being far from the FINAL target as time passes after the go
    #     cue, with an exponentially-rising weight u(t) = 1 - exp(-(t - t_go)/tau). Movement
    #     time is set by urgency_tau_ms, decoupled from the length of the go/move window, so a
    #     small tau forces a fast, early-peaking reach even in a long window. off when weight 0.
    dev = states.pos.device
    go_key = 'move_start' if 'move_start' in ts else ('go_start' if 'go_start' in ts else None)
    if go_key is None:                                                                # hold tasks: no go cue
        loss_urgency = torch.zeros((), device=dev)
    else:
        go_step = ts[go_key].to(dev)                                                  # (n,)
        tgs = torch.arange(T, device=dev).unsqueeze(0)                                # (1, T)
        tau_u = max(1.0, args.urgency_tau_ms / 1000.0 / args.dt)                      # steps
        dgo = (tgs - go_step.unsqueeze(1)).clamp(min=0)                               # steps since go
        u = (1.0 - torch.exp(-dgo / tau_u)) * (tgs >= go_step.unsqueeze(1)).float()   # (n, T)
        if 'is_no_go' in ts:
            u = u * (~ts['is_no_go']).float().unsqueeze(1)                            # no urgency on no-go
        final_tgt = desired[:, -1:, :]                                                # (n, 1, 2)
        dist_final = (states.pos - final_tgt).abs().sum(-1)                           # (n, T)
        loss_urgency = (dist_final * u).sum() / u.sum().clamp(min=1.0)

    loss = (args.w_loss_pos * loss_pos + args.w_loss_jerk * loss_jerk
            + args.w_loss_action * loss_action + args.w_loss_action_diff * loss_action_diff
            + args.w_loss_hidden * loss_hidden + args.w_loss_hidden_diff * loss_hidden_diff
            + args.w_loss_hold_pos * loss_hold_pos + args.w_loss_hold_vel * loss_hold_vel
            + args.w_loss_urgency * loss_urgency)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    opt.step()
    loss_hist.append(loss.item())

    if args.track:
        contrib = {'loss_tot': loss, 'pos': args.w_loss_pos * loss_pos, 'jerk': args.w_loss_jerk * loss_jerk,
            'muscle': args.w_loss_action * loss_action, 'muscle_diff': args.w_loss_action_diff * loss_action_diff, 'hidden':args.w_loss_hidden * loss_hidden,
            'hidden_diff':  args.w_loss_hidden_diff * loss_hidden_diff,
            'hold_pos': args.w_loss_hold_pos * loss_hold_pos, 'hold_vel': args.w_loss_hold_vel * loss_hold_vel,
            'urgency': args.w_loss_urgency * loss_urgency }

        wandb.log({f'{k}': v.item() for k, v in contrib.items()}, step=i)

    if (i + 1) % args.snap_every == 0:
        controller.eval()
        with torch.no_grad():
            ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
        controller.train()
        err = 100 * (ev.pos[:, -1, :] - eval_desired[:, -1, :]).norm(dim=1).mean().item()
        # keep the checkpoint with the lowest eval endpoint error (so late drift can't ship
        # a worse model than an earlier, better one).
        if err < best_err:
            best_err = err
            torch.save(controller.state_dict(), best_path)
        if args.track:
            # randomly select num_eval_to_plot indices from eval_desiered
            fr = fig_reaches(ev.pos, eval_desired, title=f"reaches @ batch {i+1} (err {err:.1f} cm)")
            fd = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"diagnostics @ batch {i+1}",  num_trial=5)
            wandb.log({"eval/endpoint_error_cm": err,
                       "eval/best_endpoint_error_cm": best_err,
                       "eval/reaches": wandb.Image(fr),
                       "eval/diagnostics": wandb.Image(fd)}, step=i)
            plt.close(fr); plt.close(fd)

print(f"Training complete.  start loss {loss_hist[0]:.5f} -> final loss {loss_hist[-1]:.5f}")
print(f"best eval endpoint error {best_err:.2f} cm  (saved to {os.path.basename(best_path)})")
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