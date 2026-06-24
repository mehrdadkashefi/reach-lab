"""
test.py -- load trained controllers from experiments/ and evaluate them on preset batches.

For each experiment folder (config.json + controller_*.pt) this:
  1. rebuilds the effector, controller, and the *task* the model trained on (from config.json),
  2. builds one or more test batches -- the built-in specs (a center-out reach set and a
     point-to-point set), plus any user-supplied specs,
  3. rolls the controller out and saves figures + raw arrays under <folder>/test/<spec_name>/.

By default both built-in specs are run; restrict with --builtin (e.g. --builtin center_out, or
--builtin none). User specs are added with one or more --spec PATH and named by their filename.

Batches come from the task's own make_batch(spec=...), so training (random) and testing
(explicit) share one code path. Inverse kinematics now lives on the effector
(effector.cartesian_to_joint), and perturbations are built inside the task from the spec.

A spec is a flat dict; only "start" and "target" are required, everything else falls back to
the trained task's parameters. Timing keys are interpreted by whichever task the model used:

    {
      "start_space": "joint",          # "joint" (default) or "cartesian"
      "target_space": "cartesian",     # "cartesian" (default) or "joint"
      "start":  [s, e],                # (dof,) or (n, dof); a single row broadcasts over targets
      "target": [[x, y], ...],         # (2,) or (n, 2)
      "no_go":  false,                 # bool or length-n list

      # delayed_reach timing:
      "go_time": 30, "steps": 100,

      # delayed_reach_posture timing (steps; scalar or length-n):
      "init_steps": 40, "delay_steps": 50, "move_steps": 120, "final_steps": 40,
      "final_input": "null",           # "null" or "target"

      # optional external perturbation added into the physics over [t_start, t_end):
      "perturbation": {"value": [fx, fy], "t_start": 60, "t_end": 80}
    }
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from effectors import make_effector
from tasks import make_task
from controllers import GRUController, ModularGRU
from utils import fig_reaches, fig_diagnostics


# ----------------------------------------------------------------------------- model / task rebuild
def build_controller(cfg, effector):
    """Reconstruct the controller exactly as train.py did, from a saved config dict."""
    if cfg["arch"] == "gru":
        return GRUController(effector.input_dim, hidden_dim=cfg.get("hidden_dim", 128),
                             output_dim=effector.output_dim, out_bias=effector.out_bias)
    mod_kwargs = {}
    if cfg.get("module_size")      is not None: mod_kwargs["module_sizes"]     = cfg["module_size"]
    if cfg.get("vision_mask")      is not None: mod_kwargs["vision_mask"]      = cfg["vision_mask"]
    if cfg.get("proprio_mask")     is not None: mod_kwargs["proprio_mask"]     = cfg["proprio_mask"]
    if cfg.get("task_mask")        is not None: mod_kwargs["task_mask"]        = cfg["task_mask"]
    if cfg.get("output_mask")      is not None: mod_kwargs["output_mask"]      = cfg["output_mask"]
    if cfg.get("spectral_scaling") is not None: mod_kwargs["spectral_scaling"] = cfg["spectral_scaling"]
    if cfg.get("connectivity")     is not None:
        mod_kwargs["connectivity"] = np.array(cfg["connectivity"]).reshape(3, 3)
    return ModularGRU(effector.input_dim, effector.output_dim, effector.input_layout,
                      out_bias=effector.out_bias, seed=cfg.get("seed", 0), **mod_kwargs)


def build_task(cfg, effector):
    """Reconstruct the task the model trained on, from a saved config dict (mirrors train.py)."""
    name = cfg["task"]
    if name == "delayed_reach":
        kw = dict(steps=cfg.get("steps", 100) or 100,
                  go_range=cfg.get("go_range", [20, 50]))
        if cfg.get("prob_no_go") is not None: kw["prob_no_go"] = cfg["prob_no_go"]
        return make_task(name, effector, **kw)
    elif name == "delayed_reach_posture":
        kw = {}
        if cfg.get("init_range_ms")  is not None: kw["init_range_ms"]  = tuple(cfg["init_range_ms"])
        if cfg.get("delay_range_ms") is not None: kw["delay_range_ms"] = tuple(cfg["delay_range_ms"])
        if cfg.get("move_ms")        is not None: kw["move_ms"]        = cfg["move_ms"]
        if cfg.get("final_range_ms") is not None: kw["final_range_ms"] = tuple(cfg["final_range_ms"])
        if cfg.get("final_input")    is not None: kw["final_input"]    = cfg["final_input"]
        if cfg.get("prob_no_go")     is not None: kw["prob_no_go"]     = cfg["prob_no_go"]
        return make_task(name, effector, **kw)
    raise ValueError(f"unknown task in config: {name!r}")


def load_experiment(folder, device):
    """Read config.json + the controller_*.pt in `folder`; return (effector, controller, task, cfg)."""
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)

    eff = make_effector(cfg["effector"], dt=cfg.get("dt", 0.01),
                        vis_delay_ms=cfg.get("vis_delay_ms", 70),
                        pro_delay_ms=cfg.get("pro_delay_ms", 25)).to(device)
    controller = build_controller(cfg, eff).to(device)
    task = build_task(cfg, eff)

    pts = sorted(glob.glob(os.path.join(folder, "controller_*.pt")))
    if not pts:
        raise FileNotFoundError("no controller_*.pt in folder")
    state = torch.load(pts[0], map_location=device)
    controller.load_state_dict(state)
    controller.eval()
    return eff, controller, task, cfg


# ----------------------------------------------------------------------------- built-in specs
# Five points: four corners + center of a rectangle (cartesian fingertip coords, metres).
POINT2POINT_XY = [[-0.36,  0.42], [-0.054, 0.42], [-0.36,  0.21],
                  [-0.054, 0.21], [-0.207, 0.315]]


def _apply_task_timing(cfg, spec):
    """Fill in timing keys to roughly match the task the network trained on (in place)."""
    if cfg.get("task") == "delayed_reach":
        total = int(cfg.get("steps", 100) or 100)
        spec.setdefault("steps", total)
        spec.setdefault("go_time", min(30, total // 3))          # fixed (deterministic) go onset
    else:                                                        # delayed_reach_posture
        spec.setdefault("init_steps", 40);  spec.setdefault("delay_steps", 50)
        spec.setdefault("move_steps", 120); spec.setdefault("final_steps", 40)
        spec.setdefault("final_input", cfg.get("final_input") or "null")
    return spec


def spec_center_out(cfg, effector, n_dirs=8, radius=0.10):
    """Center-out reach set: start from a single central posture and fan out to n_dirs targets
    on a circle. Timing matches the trained task. Returns a flat spec dict."""
    if effector.name == "point_mass":
        lo, hi = effector.pos_range
        center_xy = np.array([[0.5 * (lo + hi)] * 2], dtype=np.float32)
        start_space, start = "cartesian", center_xy
        radius = max(radius, 0.30)                                # point mass works over a wider range
    else:
        sho = 0.5 * (effector.sho_range[0] + effector.sho_range[1])
        elb = 0.5 * (effector.elb_range[0] + effector.elb_range[1])
        center_joint = np.array([[sho, elb]], dtype=np.float32)
        center_xy = effector.joint_to_cart(torch.as_tensor(center_joint)).cpu().numpy()
        start_space, start = "joint", center_joint

    ang = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    targets = center_xy + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)

    spec = {"start_space": start_space, "target_space": "cartesian",
            "start": start[0].tolist(), "target": targets.tolist()}
    return _apply_task_timing(cfg, spec)


def spec_point2point(cfg, effector, points_xy=None):
    """Point-to-point reaches between every *ordered* pair of a small set of cartesian points.

    With the default five points (four corners + centre of a rectangle) this yields 5*4 = 20
    conditions (each point reaches to every other). Start and target are both cartesian; for the
    arms the start is mapped to a joint config via the effector's inverse kinematics. Timing
    matches the trained task. Returns a flat spec dict.
    """
    pts = np.asarray(points_xy if points_xy is not None else POINT2POINT_XY, dtype=np.float32)
    starts = [pts[i] for i in range(len(pts)) for j in range(len(pts)) if i != j]
    targets = [pts[j] for i in range(len(pts)) for j in range(len(pts)) if i != j]
    spec = {"start_space": "cartesian", "target_space": "cartesian",
            "start": np.stack(starts).tolist(), "target": np.stack(targets).tolist()}
    return _apply_task_timing(cfg, spec)


# name -> builder(cfg, effector). Both are run by default; restrict with --builtin.
BUILTIN_SPECS = {"center_out": spec_center_out, "point2point": spec_point2point}


# ----------------------------------------------------------------------------- run one folder
def _run_one_spec(folder, name, spec, eff, controller, task, cfg,
                  obs_noise, neural_noise, num_plot, seed):
    """Roll out a single named spec and save figures + arrays under <folder>/test/<name>/."""
    torch.manual_seed(seed)
    theta0, inp, desired, pert, ts = task.make_batch(spec=spec)

    with torch.no_grad():
        states = eff.rollout(controller, theta0, inp, pert,
                             obs_noise=obs_noise, neural_noise=neural_noise)

    err = 100 * (states.pos[:, -1, :] - desired[:, -1, :]).norm(dim=1).mean().item()
    n = theta0.shape[0]

    out_dir = os.path.join(folder, "test", name)
    os.makedirs(out_dir, exist_ok=True)

    fr = fig_reaches(states.pos, desired,
                     title=f"{cfg['effector']}/{cfg['arch']} {name} (final err {err:.1f} cm)")
    fd = fig_diagnostics(eff, states, inp, desired,
                         title=f"{name} diagnostics ({cfg['effector']}/{cfg['arch']})",
                         num_trial=min(num_plot, n))
    fr.savefig(os.path.join(out_dir, "reaches.png"), dpi=120, bbox_inches="tight")
    fd.savefig(os.path.join(out_dir, "diagnostics.png"), dpi=120, bbox_inches="tight")
    plt.close(fr); plt.close(fd)

    # raw arrays for downstream analysis
    data = {k: v.detach().cpu().numpy() for k, v in vars(states).items()}
    data["inp"] = inp.detach().cpu().numpy()
    data["desired"] = desired.detach().cpu().numpy()
    if pert is not None:
        data["perturbation"] = pert.detach().cpu().numpy()
    for k, v in ts.items():
        data[f"ts_{k}"] = v.detach().cpu().numpy()
    np.savez(os.path.join(out_dir, "states.npz"), **data)

    with open(os.path.join(out_dir, "spec_used.json"), "w") as f:
        json.dump(spec, f, indent=2)

    print(f"      {name:12s} n={n:3d}  final endpoint err {err:.2f} cm  -> {out_dir}")
    return err


def run_folder(folder, named_specs, device, obs_noise, neural_noise, num_plot, seed):
    """Evaluate every (name, spec) in `named_specs` on the model in `folder`.

    Each entry's spec is either a flat dict (user spec) or a callable builder(cfg, effector)
    (built-in spec). Returns {name: final_endpoint_error_cm}.
    """
    eff, controller, task, cfg = load_experiment(folder, device)
    print(f"  {os.path.basename(folder)}: {cfg['effector']}/{cfg['arch']}")
    results = {}
    for name, spec in named_specs:
        s = spec(cfg, eff) if callable(spec) else spec
        results[name] = _run_one_spec(folder, name, s, eff, controller, task, cfg,
                                      obs_noise, neural_noise, num_plot, seed)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default=os.path.join(os.getcwd(), "experiments"))
    ap.add_argument("--spec", action="append", default=None,
                    help="path to a user JSON spec; repeatable. Each is named by its filename "
                         "and run in addition to the built-in specs.")
    ap.add_argument("--builtin", default="center_out,point2point",
                    help="comma-separated built-in specs to run "
                         f"(choices: {', '.join(BUILTIN_SPECS)}; pass 'none' to skip them)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--obs-noise", type=float, default=0.0,
                    help="std of observation noise during testing (default 0 = clean)")
    ap.add_argument("--neural-noise", type=float, default=0.0,
                    help="std of hidden-state noise during testing (default 0 = clean)")
    ap.add_argument("--num-plot", type=int, default=5, help="trials shown in the diagnostics figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None,
                    help="substring filter: only run folders whose name contains this")
    args = ap.parse_args()

    # assemble the list of (name, spec) to run: built-ins (callables) + user specs (dicts)
    named_specs = []
    for name in (s.strip() for s in args.builtin.split(",")):
        if not name or name.lower() == "none":
            continue
        if name not in BUILTIN_SPECS:
            ap.error(f"unknown builtin spec {name!r}; choices: {', '.join(BUILTIN_SPECS)}")
        named_specs.append((name, BUILTIN_SPECS[name]))

    for path in (args.spec or []):
        with open(path) as f:
            s = json.load(f)
        if "trials" in s:
            raise ValueError(
                f"{path}: old trials-list spec format; the new format is a flat dict with "
                "'start'/'target' (and optional timing) keys -- see the test.py docstring.")
        name = os.path.splitext(os.path.basename(path))[0]
        named_specs.append((name, s))

    if not named_specs:
        print("no specs to run (built-ins disabled and no --spec given)")
        return

    folders = sorted(d for d in glob.glob(os.path.join(args.experiments_dir, "*"))
                     if os.path.isfile(os.path.join(d, "config.json")))
    if args.only:
        folders = [d for d in folders if args.only in os.path.basename(d)]
    if not folders:
        print(f"no experiment folders with a config.json under {args.experiments_dir}")
        return

    spec_names = ", ".join(name for name, _ in named_specs)
    print(f"testing {len(folders)} run(s) from {args.experiments_dir}  (specs: {spec_names})")
    for folder in folders:
        try:
            print('--------------------------------')
            print(folder)
            run_folder(folder, named_specs, torch.device(args.device),
                       args.obs_noise, args.neural_noise, args.num_plot, args.seed)
        except Exception as e:
            print(f"  {os.path.basename(folder)}: SKIPPED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()