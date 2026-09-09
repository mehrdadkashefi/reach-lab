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
                                  "hold_posture_pulse", "hold_posture_ramp", "pursuit",
                                  "pacman"],
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
# pursuit (rolling preview + go cue, sharing --exec-ms / --preview-ms / --prob-catch / --go-pulse-ms with pacman)
p.add_argument("--pursuit-speed",   type=float, default=None,
               help="pursuit: fixed target speed in normalized workspace units/s (default 0.5)")
p.add_argument("--pursuit-speed-range", type=list_of_float, default=None,
               help="pursuit: min,max for a smooth time-varying target speed (overrides fixed speed)")
p.add_argument("--pursuit-turn-tau-ms", type=float, default=None,
               help="pursuit: heading-smoothness time constant (ms), default 400")
p.add_argument("--pursuit-curviness", type=float, default=None,
               help="pursuit: how sharply the path winds (default 1.0)")
# pacman (isometric force task)
p.add_argument("--exec-ms",         type=int,   default=None, help="pursuit/pacman: profile duration (ms), default 2000")
p.add_argument("--preview-ms",      type=int,   default=None, help="pursuit/pacman: rolling preview lead (ms), default 1000")
p.add_argument("--peak-force-n",    type=float, default=None, help="pacman: peak target force (N), default 8")
p.add_argument("--force-angle-deg", type=float, default=None, help="pacman: fixed force direction (deg), default 90")
p.add_argument("--random-force-dir", action="store_true", help="pacman: randomize force direction per trial")
p.add_argument("--prob-catch",      type=float, default=None, help="pacman: fraction of no-go catch trials, default 0.2")
p.add_argument("--pacman-conditions", type=str, default=None,
               help="pacman: comma-separated subset of conditions to train on (default: all 12)")
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
               help="fraction of trials where one random reach gets no go pulse "
                    "(the hand holds through that segment; the sequence resumes after)")
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
elif args.task == "pursuit":
    pk = {'unified_input': args.unified_input, 'go_pulse_ms': args.go_pulse_ms}
    if args.exec_ms              is not None: pk['exec_ms']       = args.exec_ms
    if args.preview_ms           is not None: pk['preview_ms']    = args.preview_ms
    if args.prob_catch           is not None: pk['prob_catch']    = args.prob_catch
    if args.pursuit_speed        is not None: pk['speed']         = args.pursuit_speed
    if args.pursuit_speed_range  is not None: pk['speed_range']   = tuple(args.pursuit_speed_range)
    if args.pursuit_turn_tau_ms  is not None: pk['turn_tau_ms']   = args.pursuit_turn_tau_ms
    if args.pursuit_curviness    is not None: pk['curviness']     = args.pursuit_curviness
    task = make_task(args.task, eff, **pk)
elif args.task == "pacman":
    mk = {'unified_input': args.unified_input, 'go_pulse_ms': args.go_pulse_ms}
    if args.exec_ms          is not None: mk['exec_ms']        = args.exec_ms
    if args.preview_ms       is not None: mk['preview_ms']     = args.preview_ms
    if args.peak_force_n     is not None: mk['peak_force_n']   = args.peak_force_n
    if args.force_angle_deg  is not None: mk['force_angle_deg'] = args.force_angle_deg
    if args.random_force_dir:             mk['random_dir']     = True
    if args.prob_catch       is not None: mk['prob_catch']     = args.prob_catch
    if args.pacman_conditions is not None:
        mk['conditions'] = [c.strip() for c in args.pacman_conditions.split(",") if c.strip()]
    task = make_task(args.task, eff, **mk)
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
IS_FORCE_TASK = getattr(task, "is_force_task", False)               # isometric force task (pacman)
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
for _attr in ('perturb_prob', 'catch_prob', 'prob_catch'):
    if hasattr(task, _attr):
        _saved[_attr] = getattr(task, _attr); setattr(task, _attr, 0.0)
eval_theta0, eval_inp, eval_desired, eval_perturbation, eval_timestamps = task.make_batch(num_eval)
for _attr, _val in _saved.items():
    setattr(task, _attr, _val)
torch.manual_seed(args.seed)


# ----------------------------------------------------------------------------- train
loss_hist, snapshots = [], []
best_err = float('inf')                                   # lowest eval endpoint error so far
best_path = out(f"controller_{args.effector}_{args.arch}_best.pt")
for i in tqdm(range(args.n_batch)):
    theta0, inp, desired, perturbation, ts = task.make_batch(args.batch_size)
    states = eff.rollout(controller, theta0, inp, perturbation,
                         obs_noise=args.obs_noise, neural_noise=args.neural_noise,
                         action_noise=args.action_noise)

    # main tracking loss: force (isometric pacman) vs position (all other tasks). `desired` is a
    # target FORCE for the force task and a target POSITION otherwise; states.force / states.pos
    # is the matching generated quantity.
    tracked = states.force if IS_FORCE_TASK else states.pos
    # a task may supply a per-timestep mask marking steps that should not be scored -- e.g. the
    # window after an unpreviewed go cue, before the target can physically have been perceived.
    _err = (tracked - desired).abs().sum(-1)                               # (n, T)
    _lm = ts.get('loss_mask') if isinstance(ts, dict) else None
    if _lm is None:
        loss_pos = _err.mean()
    else:
        _lm = _lm.to(_err.device).float()
        loss_pos = (_err * _lm).sum() / _lm.sum().clamp(min=1.0)
    # jerk loss is a *movement*-smoothness penalty, always on position (frozen -> 0 for the
    # isometric force task, whose force profile is meant to have sharp features and must NOT be
    # penalised by the position-tuned jerk weight).
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

    loss = (args.w_loss_pos * loss_pos + args.w_loss_jerk * loss_jerk
            + args.w_loss_action * loss_action + args.w_loss_action_diff * loss_action_diff
            + args.w_loss_hidden * loss_hidden + args.w_loss_hidden_diff * loss_hidden_diff)

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
    opt.step()
    loss_hist.append(loss.item())

    if args.track:
        contrib = {'loss_tot': loss, 'pos': args.w_loss_pos * loss_pos, 'jerk': args.w_loss_jerk * loss_jerk,
            'muscle': args.w_loss_action * loss_action, 'muscle_diff': args.w_loss_action_diff * loss_action_diff, 'hidden':args.w_loss_hidden * loss_hidden,
            'hidden_diff':  args.w_loss_hidden_diff * loss_hidden_diff }

        wandb.log({f'{k}': v.item() for k, v in contrib.items()}, step=i)

    if (i + 1) % args.snap_every == 0:
        controller.eval()
        with torch.no_grad():
            ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
        controller.train()
        if IS_FORCE_TASK:
            # force error (N), averaged over the whole trial
            err = (ev.force - eval_desired).norm(dim=-1).mean().item()
        else:
            err = 100 * (ev.pos[:, -1, :] - eval_desired[:, -1, :]).norm(dim=1).mean().item()
        # keep the checkpoint with the lowest eval error (so late drift can't ship
        # a worse model than an earlier, better one).
        if err < best_err:
            best_err = err
            torch.save(controller.state_dict(), best_path)
        if args.track:
            if IS_FORCE_TASK:
                fr = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"force @ batch {i+1} (err {err:.2f} N)", num_trial=5)
                fd = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"diagnostics @ batch {i+1}", num_trial=5)
            else:
                fr = fig_reaches(ev.pos, eval_desired, title=f"reaches @ batch {i+1} (err {err:.1f} cm)")
                fd = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"diagnostics @ batch {i+1}",  num_trial=5)
            wandb.log({"eval/error": err,
                       "eval/best_error": best_err,
                       "eval/reaches": wandb.Image(fr),
                       "eval/diagnostics": wandb.Image(fd)}, step=i)
            plt.close(fr); plt.close(fd)

print(f"Training complete.  start loss {loss_hist[0]:.5f} -> final loss {loss_hist[-1]:.5f}")
print(f"best eval error {best_err:.3f}  (saved to {os.path.basename(best_path)})")
torch.save(controller.state_dict(), out(f"controller_{args.effector}_{args.arch}.pt"))


# ----------------------------------------------------------------------------- final plots
tag = f"{args.effector}/{args.arch}"
lc = fig_learning_curve(loss_hist, f'learning curve ({tag})')
# Final Evaluation
controller.eval()
with torch.no_grad():
    ev = eff.rollout(controller, eval_theta0, eval_inp, eval_perturbation)
if IS_FORCE_TASK:
    td = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"Sample force trials ({tag})", num_trial=5)
else:
    td = fig_diagnostics(eff, ev, eval_inp, eval_desired, title=f"Sample trials ({tag}; dotted line = go cue)", num_trial=5)
lc.savefig(out('learning_curve.png'), dpi=120, bbox_inches='tight')
td.savefig(out('trial_diagnostics.png'), dpi=120, bbox_inches='tight')
print(f"\nsaved plots + controller to {run_dir}")

if args.track:
    wandb.log({"learning_curve": wandb.Image(lc),
               "trial_diagnostics": wandb.Image(td)})
    wandb.finish()
plt.close('all')