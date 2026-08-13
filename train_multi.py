"""
train_multitask.py -- train one controller across several tasks and measure how much learning
generalizes between them.

Three scenarios (--scenario):
  all           : train on every task in --tasks. Capacity / interference probe: can one network
                  hold all the skills at once?
  single        : train on --train-task only, but validate on every task each eval step. Pure
                  zero-shot transfer: does competence on A confer competence on B?
  leave_one_out : train on every task except --holdout, validate on all. Compositionality probe:
                  can the held-out task be solved from what the others taught?

All three validate on the SAME fixed eval batches for every task, so the per-task curves are
directly comparable across scenarios.

Optionally (--finetune-steps > 0) each eval also runs a short fine-tune probe: from a *copy* of
the current controller, train N steps on each non-trained task and report the error after. Speed
of acquisition is usually a far more sensitive transfer measure than zero-shot error, which is
easily dominated by output-scale mismatch.

Batching. Tasks have different trial lengths (a 7-reach sequence is much longer than a 3 s reach),
so a single stacked batch is impossible without padding. Instead each optimizer step walks the
training tasks round-robin: one task-homogeneous batch per task (its own native T, no padding),
accumulating gradients, then one optimizer step. Each task's loss is averaged over time, so every
task contributes equally per step regardless of trial length; --task-weights re-weights if wanted.

Effectors. Every task shares one effector instance (same type, same 10-wide unified instruction
stream) EXCEPT pacman, which needs its plant held isometric -- it gets a second, identical
effector with isometric=True. Same effector type throughout, so input_dim/output_dim match and one
controller can drive them all.
"""

import argparse
import copy
import datetime
import json
import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from effectors import make_effector
from tasks import make_task, TASKS, UNIFIED_INPUT_CHANNELS
from controllers import GRUController, ModularGRU
from utils import fig_diagnostics, fig_learning_curve


ALL_TASKS = ["delayed_reach", "delayed_reach_posture", "horizon_sequence",
             "hold_posture_pulse", "hold_posture_ramp", "pursuit", "pacman"]


def list_of_str(s):   return [x.strip() for x in s.split(',') if x.strip()]
def list_of_float(s): return [float(x) for x in s.split(',')]


# ----------------------------------------------------------------------------- task construction
def build_tasks(names, args, device):
    """Build every task on a shared effector, except the isometric force task which gets its own.

    Returns (tasks: {name: task}, effectors: {'move': eff, 'iso': eff_or_None}). All tasks use the
    10-wide unified instruction stream so one controller fits them all.
    """
    eff_kw = dict(dt=args.dt, vis_delay_ms=args.vis_delay_ms, pro_delay_ms=args.pro_delay_ms,
                  task_dim=UNIFIED_INPUT_CHANNELS)
    eff_move = make_effector(args.effector, **eff_kw).to(device)
    eff_iso = None
    if any(getattr(TASKS[n], "is_force_task", False) for n in names):
        eff_iso = make_effector(args.effector, **eff_kw).to(device)      # pacman sets isometric=True

    tasks = {}
    for n in names:
        is_force = getattr(TASKS[n], "is_force_task", False)
        eff = eff_iso if is_force else eff_move
        kw = {'unified_input': True}
        if n == "horizon_sequence":
            kw.pop('unified_input')                                     # natively 10-wide
            kw['n_reaches'] = args.n_reaches
        if n in ("pursuit", "pacman"):
            kw['exec_ms'] = args.exec_ms
            kw['preview_ms'] = args.preview_ms
        tasks[n] = make_task(n, eff, **kw)
    return tasks, {'move': eff_move, 'iso': eff_iso}


def task_effector(name, effectors):
    return effectors['iso'] if getattr(TASKS[name], "is_force_task", False) else effectors['move']


# ----------------------------------------------------------------------------- loss / metric
def compute_loss(args, states, desired, is_force):
    """Same loss as train.py: force-tracking for the isometric task, position otherwise. Every term
    is averaged over time, so tasks with longer trials do not get more weight."""
    tracked = states.force if is_force else states.pos
    loss_pos = (tracked - desired).abs().sum(-1).mean()
    _jerk = (states.pos[:, 3:] - 3 * states.pos[:, 2:-1]
             + 3 * states.pos[:, 1:-2] - states.pos[:, :-3])
    loss_jerk = _jerk.pow(2).sum(-1).mean()
    _action = states.action
    loss_action = _action.pow(2).sum(-1).mean()
    loss_action_diff = (_action[:, 1:] - _action[:, :-1]).pow(2).sum(-1).mean()
    _hidden = states.hidden
    loss_hidden = _hidden.pow(2).sum(-1).mean()
    loss_hidden_diff = (_hidden[:, 1:] - _hidden[:, :-1]).pow(2).sum(-1).mean()
    return (args.w_loss_pos * loss_pos + args.w_loss_jerk * loss_jerk
            + args.w_loss_action * loss_action + args.w_loss_action_diff * loss_action_diff
            + args.w_loss_hidden * loss_hidden + args.w_loss_hidden_diff * loss_hidden_diff)


def task_error(states, desired, is_force):
    """Interpretable per-task error: mean force error (N) for the isometric task, final-position
    error (cm) otherwise. Reported alongside the raw loss so curves are readable."""
    if is_force:
        return (states.force - desired).norm(dim=-1).mean().item()
    return 100 * (states.pos[:, -1, :] - desired[:, -1, :]).norm(dim=1).mean().item()


# ----------------------------------------------------------------------------- eval
def make_eval_batches(tasks, args, device, seed=123):
    """One fixed batch per task, reused at every eval so curves are comparable across scenarios and
    scenarios see identical test data. Trial-level randomness that would blank the signal
    (perturbations / catch trials) is disabled, exactly as train.py does for its eval set."""
    batches = {}
    for name, task in tasks.items():
        torch.manual_seed(seed)
        saved = {}
        for attr in ('perturb_prob', 'catch_prob', 'prob_catch'):
            if hasattr(task, attr):
                saved[attr] = getattr(task, attr); setattr(task, attr, 0.0)
        batches[name] = task.make_batch(args.n_eval)
        for attr, val in saved.items():
            setattr(task, attr, val)
    torch.manual_seed(args.seed)
    return batches


def evaluate(controller, tasks, eval_batches, effectors, args):
    """Zero-shot error + loss on every task. Returns {name: {'err':..., 'loss':...}}."""
    controller.eval()
    res = {}
    with torch.no_grad():
        for name, task in tasks.items():
            eff = task_effector(name, effectors)
            is_force = getattr(TASKS[name], "is_force_task", False)
            theta0, inp, desired, pert, ts = eval_batches[name]
            st = eff.rollout(controller, theta0, inp, pert)
            res[name] = {'err': task_error(st, desired, is_force),
                         'loss': compute_loss(args, st, desired, is_force).item()}
    controller.train()
    return res


def finetune_probe(controller, task, name, effectors, args, eval_batch):
    """Adaptation-speed probe: copy the controller, train `--finetune-steps` on `task`, and report
    the error after. Measures how *fast* a task can be picked up, which is usually far more
    sensitive to transfer than zero-shot error. The live controller is never touched."""
    probe = copy.deepcopy(controller)
    probe.train()
    opt = torch.optim.Adam(probe.parameters(), lr=args.finetune_lr)
    eff = task_effector(name, effectors)
    is_force = getattr(TASKS[name], "is_force_task", False)
    for _ in range(args.finetune_steps):
        theta0, inp, desired, pert, ts = task.make_batch(args.finetune_batch)
        st = eff.rollout(probe, theta0, inp, pert, obs_noise=args.obs_noise,
                         neural_noise=args.neural_noise, action_noise=args.action_noise)
        loss = compute_loss(args, st, desired, is_force)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)
        opt.step()
    probe.eval()
    with torch.no_grad():
        theta0, inp, desired, pert, ts = eval_batch
        st = eff.rollout(probe, theta0, inp, pert)
        err = task_error(st, desired, is_force)
    del probe
    return err


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=["all", "single", "leave_one_out"], default="all")
    p.add_argument("--tasks", type=list_of_str, default=ALL_TASKS,
                   help="task set to consider (comma-separated); default = all 7")
    p.add_argument("--train-task", default="pursuit",
                   help="scenario 'single': the one task trained on")
    p.add_argument("--holdout", default="pacman",
                   help="scenario 'leave_one_out': the task excluded from training")
    p.add_argument("--task-weights", type=list_of_float, default=None,
                   help="per-training-task loss weights (comma-separated, order of --tasks); "
                        "default = equal weight for every task")
    p.add_argument("--effector", choices=["point_mass", "arm_torque", "arm26"], default="arm26")
    p.add_argument("--arch", choices=["gru", "modular"], default="gru")
    p.add_argument("--hidden-dim", type=int, default=256)
    # training
    p.add_argument("--n-batch", type=int, default=600, help="optimizer steps (each = one round-robin sweep)")
    p.add_argument("--batch-size", type=int, default=64, help="trials per task per step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    # eval / probes
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--finetune-steps", type=int, default=0,
                   help="if >0, at each eval also fine-tune a copy on each non-trained task for "
                        "this many steps and report the resulting error (adaptation speed)")
    p.add_argument("--finetune-batch", type=int, default=32)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    # shared task knobs
    p.add_argument("--n-reaches", type=int, default=4, help="horizon_sequence length")
    p.add_argument("--exec-ms", type=int, default=2000, help="pursuit/pacman profile duration")
    p.add_argument("--preview-ms", type=int, default=1000, help="pursuit/pacman preview lead")
    # effector
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--vis-delay-ms", type=float, default=70)
    p.add_argument("--pro-delay-ms", type=float, default=25)
    # loss weights (same as train.py)
    p.add_argument("--w-loss-pos", type=float, default=1)
    p.add_argument("--w-loss-jerk", type=float, default=1e4)
    p.add_argument("--w-loss-action", type=float, default=0.5)
    p.add_argument("--w-loss-action-diff", type=float, default=3e-3)
    p.add_argument("--w-loss-hidden", type=float, default=3e-4)
    p.add_argument("--w-loss-hidden-diff", type=float, default=3e-2)
    # noise
    p.add_argument("--obs-noise", type=float, default=0.1)
    p.add_argument("--neural-noise", type=float, default=0.05)
    p.add_argument("--action-noise", type=float, default=0.0)
    p.add_argument("--track", action="store_true", help="log to wandb")
    p.add_argument("--wandb-project", default="reach-lab-multitask")
    args = p.parse_args()

    for n in args.tasks:
        if n not in TASKS:
            p.error(f"unknown task {n!r}; choices: {', '.join(ALL_TASKS)}")

    # which tasks are trained on, in this scenario
    if args.scenario == "all":
        train_names = list(args.tasks)
    elif args.scenario == "single":
        if args.train_task not in args.tasks:
            p.error(f"--train-task {args.train_task!r} not in --tasks")
        train_names = [args.train_task]
    else:
        if args.holdout not in args.tasks:
            p.error(f"--holdout {args.holdout!r} not in --tasks")
        train_names = [n for n in args.tasks if n != args.holdout]
    eval_names = list(args.tasks)                       # always validate on everything
    untrained = [n for n in eval_names if n not in train_names]

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Running on: {device}")

    run_dir = os.path.join(os.getcwd(), "experiments_multitask",
                           f"{args.scenario}_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(run_dir, exist_ok=True)
    def out(name): return os.path.join(run_dir, name)
    with open(out("config.json"), "w") as f:
        json.dump({**vars(args), 'train_names': train_names, 'eval_names': eval_names},
                  f, indent=2, default=str)
    print(f"saving results to {run_dir}")

    # tasks + effectors (pacman gets its own isometric effector)
    tasks, effectors = build_tasks(eval_names, args, device)
    eff_move = effectors['move']
    print(f"effector: {eff_move.name} (input_dim={eff_move.input_dim}, output_dim={eff_move.output_dim})"
          + ("  + a second isometric effector for pacman" if effectors['iso'] is not None else ""))
    print(f"scenario {args.scenario}: training on {train_names}")
    print(f"  validating on {eval_names}" + (f"  (untrained: {untrained})" if untrained else ""))
    for n in eval_names:
        print(f"    {n:22s} {tasks[n].steps:4d} steps = {tasks[n].steps * args.dt:.2f} s")

    # per-task loss weights (equal by default)
    if args.task_weights is None:
        weights = {n: 1.0 for n in train_names}
    else:
        if len(args.task_weights) != len(args.tasks):
            p.error("--task-weights must have one entry per task in --tasks")
        wmap = dict(zip(args.tasks, args.task_weights))
        weights = {n: wmap[n] for n in train_names}

    # controller (one network for every task -- shapes match because all tasks are 10-wide)
    if args.arch == "gru":
        controller = GRUController(eff_move.input_dim, hidden_dim=args.hidden_dim,
                                   output_dim=eff_move.output_dim, out_bias=eff_move.out_bias)
        print(f"controller: single GRU, {args.hidden_dim} units")
    else:
        controller = ModularGRU(eff_move.input_dim, eff_move.output_dim, eff_move.input_layout,
                                out_bias=eff_move.out_bias, seed=args.seed)
        print(f"controller: ModularGRU, {controller.hidden_dim} units")
    controller = controller.to(device)
    opt = torch.optim.Adam(controller.parameters(), lr=args.lr)

    eval_batches = make_eval_batches(tasks, args, device)

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project, name=os.path.basename(run_dir),
                   config={**vars(args), 'train_names': train_names})

    # ------------------------------------------------------------------- train
    history = []                        # list of {'step':.., 'eval':{task: {...}}, 'finetune':{...}}
    loss_hist = []
    for i in tqdm(range(args.n_batch)):
        opt.zero_grad()
        step_losses = {}
        # round-robin: one task-homogeneous batch per task (native T, no padding), accumulate grads
        for name in train_names:
            task = tasks[name]
            eff = task_effector(name, effectors)
            is_force = getattr(TASKS[name], "is_force_task", False)
            theta0, inp, desired, pert, ts = task.make_batch(args.batch_size)
            st = eff.rollout(controller, theta0, inp, pert, obs_noise=args.obs_noise,
                             neural_noise=args.neural_noise, action_noise=args.action_noise)
            loss = compute_loss(args, st, desired, is_force)
            # scale so the *sum* over tasks is the mean task loss (keeps grad magnitude comparable
            # to single-task training regardless of how many tasks are in the mix)
            (weights[name] * loss / max(1e-8, sum(weights.values()))).backward()
            step_losses[name] = loss.item()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
        opt.step()
        loss_hist.append(float(np.mean(list(step_losses.values()))))

        if args.track:
            wandb.log({'loss_mean': loss_hist[-1],
                       **{f'train_loss/{k}': v for k, v in step_losses.items()}}, step=i)

        if (i + 1) % args.eval_every == 0 or i == args.n_batch - 1:
            ev = evaluate(controller, tasks, eval_batches, effectors, args)
            rec = {'step': i + 1, 'eval': ev}
            if args.finetune_steps > 0 and untrained:
                rec['finetune'] = {n: finetune_probe(controller, tasks[n], n, effectors, args,
                                                     eval_batches[n])
                                   for n in untrained}
            history.append(rec)
            msg = "  ".join(f"{n}:{ev[n]['err']:.2f}" for n in eval_names)
            tqdm.write(f"[{i+1:5d}] " + msg)
            if args.track:
                wandb.log({**{f'eval_err/{n}': ev[n]['err'] for n in eval_names},
                           **{f'eval_loss/{n}': ev[n]['loss'] for n in eval_names},
                           **({f'finetune_err/{n}': v for n, v in rec.get('finetune', {}).items()})},
                          step=i)

    # ------------------------------------------------------------------- save
    torch.save(controller.state_dict(), out("controller.pt"))
    with open(out("history.json"), "w") as f:
        json.dump({'train_names': train_names, 'eval_names': eval_names,
                   'untrained': untrained, 'history': history, 'loss_hist': loss_hist},
                  f, indent=2)

    # learning curve + per-task generalization curves
    fig_learning_curve(loss_hist, f"multitask ({args.scenario}) mean train loss").savefig(
        out("learning_curve.png"), dpi=120, bbox_inches="tight")
    steps = [h['step'] for h in history]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for n in eval_names:
        style = '-' if n in train_names else '--'
        ax.plot(steps, [h['eval'][n]['err'] for h in history], style, lw=1.6, label=n)
    ax.set_yscale('log'); ax.set_xlabel("optimizer step"); ax.set_ylabel("eval error (cm / N)")
    ax.set_title(f"{args.scenario}: solid = trained, dashed = untrained")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out("generalization.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

    # final per-task diagnostics
    controller.eval()
    for n in eval_names:
        eff = task_effector(n, effectors)
        theta0, inp, desired, pert, ts = eval_batches[n]
        with torch.no_grad():
            st = eff.rollout(controller, theta0, inp, pert)
        f = fig_diagnostics(eff, st, inp, desired,
                            title=f"{n} ({'trained' if n in train_names else 'UNTRAINED'})",
                            num_trial=min(4, args.n_eval))
        f.savefig(out(f"diag_{n}.png"), dpi=100, bbox_inches="tight"); plt.close(f)

    print("\nfinal per-task error:")
    for n in eval_names:
        tag = "trained" if n in train_names else "UNTRAINED"
        unit = "N" if getattr(TASKS[n], "is_force_task", False) else "cm"
        line = f"  {n:22s} {history[-1]['eval'][n]['err']:8.3f} {unit:2s}  [{tag}]"
        if 'finetune' in history[-1] and n in history[-1]['finetune']:
            line += f"   after {args.finetune_steps}-step finetune: {history[-1]['finetune'][n]:.3f} {unit}"
        print(line)
    print(f"\nsaved to {run_dir}")


if __name__ == "__main__":
    main()