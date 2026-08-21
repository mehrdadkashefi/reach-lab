"""
test_model.py -- load trained controllers from experiments/ and evaluate them on preset batches.

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
      "go_pulse_steps": 15,            # go-cue pulse length in steps (default: the trained
                                       # task's go_pulse_ms; 0 or negative = sustained cue)

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
from tasks import make_task, TASKS, task_input_channels, PacManTask
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
    # go-cue pulse the model trained with; configs predating the pulse get 0 = sustained cue
    go_pulse_ms = cfg.get("go_pulse_ms", 0)
    unified = cfg.get("unified_input", False)
    if name == "delayed_reach":
        kw = dict(steps=cfg.get("steps", 100) or 100,
                  go_range=cfg.get("go_range", [20, 50]),
                  go_pulse_ms=go_pulse_ms, unified_input=unified)
        if cfg.get("prob_no_go") is not None: kw["prob_no_go"] = cfg["prob_no_go"]
        return make_task(name, effector, **kw)
    elif name in ("hold_posture_pulse", "hold_posture_ramp"):
        kw = {"unified_input": unified}
        if cfg.get("onset_range_ms") is not None: kw["onset_range_ms"] = tuple(cfg["onset_range_ms"])
        if cfg.get("force_range_n")  is not None: kw["force_range_n"]  = tuple(cfg["force_range_n"])
        if cfg.get("center_frac")    is not None: kw["center_frac"]    = cfg["center_frac"]
        if cfg.get("catch_prob")     is not None: kw["catch_prob"]     = cfg["catch_prob"]
        if name == "hold_posture_pulse":
            if cfg.get("dur_range_ms") is not None: kw["dur_range_ms"] = tuple(cfg["dur_range_ms"])
        else:
            if cfg.get("ramp_range_ms")  is not None: kw["ramp_range_ms"]   = tuple(cfg["ramp_range_ms"])
            if cfg.get("hold_after_ramp") is not None: kw["hold_after_ramp"] = cfg["hold_after_ramp"]
        return make_task(name, effector, **kw)
    elif name == "delayed_reach_posture":
        kw = {"go_pulse_ms": go_pulse_ms, "unified_input": unified}
        if cfg.get("init_range_ms")  is not None: kw["init_range_ms"]  = tuple(cfg["init_range_ms"])
        if cfg.get("delay_range_ms") is not None: kw["delay_range_ms"] = tuple(cfg["delay_range_ms"])
        if cfg.get("move_ms")        is not None: kw["move_ms"]        = cfg["move_ms"]
        if cfg.get("final_range_ms") is not None: kw["final_range_ms"] = tuple(cfg["final_range_ms"])
        if cfg.get("final_input")    is not None: kw["final_input"]    = cfg["final_input"]
        if cfg.get("prob_no_go")     is not None: kw["prob_no_go"]     = cfg["prob_no_go"]
        return make_task(name, effector, **kw)
    elif name == "horizon_sequence":
        kw = {"go_pulse_ms": go_pulse_ms}
        if cfg.get("n_reaches")        is not None: kw["n_reaches"]        = cfg["n_reaches"]
        if cfg.get("init_range_ms")    is not None: kw["init_range_ms"]    = tuple(cfg["init_range_ms"])
        if cfg.get("delay_range_ms")   is not None: kw["delay_range_ms"]   = tuple(cfg["delay_range_ms"])
        if cfg.get("dwell_range_ms")   is not None: kw["dwell_range_ms"]   = tuple(cfg["dwell_range_ms"])
        if cfg.get("final_range_ms")   is not None: kw["final_range_ms"]   = tuple(cfg["final_range_ms"])
        if cfg.get("horizon_probs")    is not None: kw["horizon_probs"]    = tuple(cfg["horizon_probs"])
        if cfg.get("prob_no_go")       is not None: kw["prob_no_go"]       = cfg["prob_no_go"]
        if cfg.get("prob_no_go_reach") is not None: kw["prob_no_go_reach"] = cfg["prob_no_go_reach"]
        return make_task(name, effector, **kw)
    elif name == "pursuit":
        kw = {"unified_input": unified, "go_pulse_ms": go_pulse_ms}
        if cfg.get("exec_ms")             is not None: kw["exec_ms"]      = cfg["exec_ms"]
        if cfg.get("preview_ms")          is not None: kw["preview_ms"]   = cfg["preview_ms"]
        if cfg.get("prob_catch")          is not None: kw["prob_catch"]   = cfg["prob_catch"]
        if cfg.get("pursuit_speed")       is not None: kw["speed"]        = cfg["pursuit_speed"]
        if cfg.get("pursuit_speed_range") is not None: kw["speed_range"]  = tuple(cfg["pursuit_speed_range"])
        if cfg.get("pursuit_turn_tau_ms") is not None: kw["turn_tau_ms"]  = cfg["pursuit_turn_tau_ms"]
        if cfg.get("pursuit_curviness")   is not None: kw["curviness"]    = cfg["pursuit_curviness"]
        return make_task(name, effector, **kw)
    elif name == "pacman":
        kw = {"unified_input": unified, "go_pulse_ms": go_pulse_ms}
        if cfg.get("exec_ms")         is not None: kw["exec_ms"]         = cfg["exec_ms"]
        if cfg.get("preview_ms")      is not None: kw["preview_ms"]      = cfg["preview_ms"]
        if cfg.get("peak_force_n")    is not None: kw["peak_force_n"]    = cfg["peak_force_n"]
        if cfg.get("force_angle_deg") is not None: kw["force_angle_deg"] = cfg["force_angle_deg"]
        if cfg.get("random_dir")      is not None: kw["random_dir"]      = cfg["random_dir"]
        if cfg.get("prob_catch")      is not None: kw["prob_catch"]      = cfg["prob_catch"]
        if cfg.get("conditions")      is not None: kw["conditions"]      = cfg["conditions"]
        return make_task(name, effector, **kw)
    raise ValueError(f"unknown task in config: {name!r}")


def load_experiment(folder, device):
    """Read config.json + the controller_*.pt in `folder`; return (effector, controller, task, cfg)."""
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)

    eff = make_effector(cfg["effector"], dt=cfg.get("dt", 0.01),
                        vis_delay_ms=cfg.get("vis_delay_ms", 70),
                        pro_delay_ms=cfg.get("pro_delay_ms", 25),
                        task_dim=task_input_channels(cfg["task"],
                                                     cfg.get("unified_input", False))).to(device)
    controller = build_controller(cfg, eff).to(device)
    task = build_task(cfg, eff)

    pts = sorted(glob.glob(os.path.join(folder, "controller_*.pt")))
    if not pts:
        raise FileNotFoundError("no controller_*.pt in folder")
    state = torch.load(pts[0], map_location=device)
    controller.load_state_dict(state)
    controller.eval()
    return eff, controller, task, cfg


def _folder_layout(folder):
    """Read a run folder's config and normalise the two layouts this repo produces.

    Returns (cfg, task_names, ckpt, is_multitask):
      - train.py runs           : config has 'task';        checkpoint controller_*.pt
      - train_multitask.py runs : config has 'eval_names';  checkpoint controller.pt
    `task_names` is every task the folder can be rolled out on (the trained one, or all the
    multi-task run validated on).
    """
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    is_multi = "task" not in cfg and ("eval_names" in cfg or "train_names" in cfg)
    if is_multi:
        names = list(cfg.get("eval_names") or cfg.get("train_names"))
    else:
        names = [cfg["task"]]
    pts = sorted(glob.glob(os.path.join(folder, "controller*.pt")))
    if not pts:
        raise FileNotFoundError(f"no controller*.pt in {folder}")
    # prefer the '_best' checkpoint when train.py saved one
    ckpt = next((p for p in pts if p.endswith("_best.pt")), pts[0])
    return cfg, names, ckpt, is_multi


def _build_for_task(cfg, name, is_multi, device):
    """Effector + task for one task name, matching how the run was trained. The isometric force
    task gets its own effector (it holds the plant fixed); everything else uses a normal one."""
    unified = True if is_multi else cfg.get("unified_input", False)
    eff = make_effector(cfg["effector"], dt=cfg.get("dt", 0.01),
                        vis_delay_ms=cfg.get("vis_delay_ms", 70),
                        pro_delay_ms=cfg.get("pro_delay_ms", 25),
                        task_dim=task_input_channels(name, unified)).to(device)
    if is_multi:
        kw = {'unified_input': True}
        if name == "horizon_sequence":
            kw = {'n_reaches': cfg.get("n_reaches", 4)}          # natively 10-wide
        if name in ("pursuit", "pacman"):
            kw['exec_ms'] = cfg.get("exec_ms", 2000)
            kw['preview_ms'] = cfg.get("preview_ms", 1000)
        task = make_task(name, eff, **kw)
    else:
        task = build_task({**cfg, "task": name}, eff)            # reuse the trained hyperparameters
    return eff, task


def rollout_experiment(folder, n=5, task=None, spec=None, device="cpu", seed=0,
                       obs_noise=0.0, neural_noise=0.0, verbose=True):
    """Load a trained run and roll it out; returns everything needed for plotting or video.

    Works for both single-task runs (train.py) and multi-task runs (train_multitask.py).

        eff, states, inp, desired, extras = rollout_experiment("experiments/2026-...", n=5)
        video_trials(eff, states, inp, desired, "run.mp4", num_trial=5)

    Args:
        n:     trials to simulate when `spec` is None.
        task:  roll out on a *different* task than the trained one (e.g. to see what a
               pursuit-trained network does on the sequence task). Default None = the task the
               model trained on; for a multi-task folder, its first validated task. The task's
               instruction-stream width must match what the controller was trained with, or a
               clear error is raised rather than a shape mismatch deep in the rollout.
        spec:  optional explicit spec dict instead of a random batch.
    Returns:
        (effector, states, inp, desired, extras); extras has 'controller','task','cfg','theta0',
        'perturbation','timestamps'.
    """
    device = torch.device(device)
    cfg, names, ckpt, is_multi = _folder_layout(folder)
    name = task or names[0]
    if task is not None and task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choices: {', '.join(TASKS)}")

    eff, tsk = _build_for_task(cfg, name, is_multi, device)
    controller = build_controller({**cfg, "task": name}, eff).to(device)
    state = torch.load(ckpt, map_location=device)
    try:
        controller.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint does not fit task {name!r} (input width {eff.task_dim}). A network "
            f"trained on a narrower instruction stream cannot be run on a wider one -- retrain "
            f"with --unified-input to make tasks interchangeable.\n  {e}") from e
    controller.eval()

    if verbose:
        trained = ", ".join(names)
        print(f"[{os.path.basename(folder)}] {'multi-task' if is_multi else 'single-task'} run "
              f"({trained}) | ckpt {os.path.basename(ckpt)}")
        print(f"  rolling out {name}"
              + ("  <-- NOT a trained task (transfer)" if name not in names else "")
              + f" | {tsk.steps} steps = {tsk.steps * cfg.get('dt', 0.01):.2f} s"
              + f" | effector {eff.name}{' (isometric)' if eff.isometric else ''}")

    torch.manual_seed(seed)
    theta0, inp, desired, pert, ts = (tsk.make_batch(spec=spec) if spec is not None
                                      else tsk.make_batch(n))
    with torch.no_grad():
        states = eff.rollout(controller, theta0, inp, pert,
                             obs_noise=obs_noise, neural_noise=neural_noise)
    extras = {'controller': controller, 'task': tsk, 'cfg': cfg,
              'theta0': theta0, 'perturbation': pert, 'timestamps': ts}
    return eff, states, inp, desired, extras


def video_experiment(folder, path="rollout.mp4", tasks=None, n=5, device="cpu", seed=0,
                     obs_noise=0.0, neural_noise=0.0, verbose=True, **video_kw):
    """Roll out a trained run and render videos, one segment of `n` trials per task.

    For a single-task folder this is just that task (override with `tasks=[...]`). For a
    multi-task folder it defaults to every task the run validated on, so you get one artifact
    showing the same network across its whole repertoire. Per-task clips are rendered separately
    (each task has its own trial length and effector) and then concatenated into `path` when the
    format allows it (mp4 + ffmpeg); otherwise the individual clips are kept and returned.

    Extra keyword arguments are passed straight through to utils.video_trials (fps, speed, dark,
    n_units, tail, dpi, ...).
    """
    import subprocess
    import tempfile
    from utils import video_trials

    cfg, names, _, is_multi = _folder_layout(folder)
    todo = list(tasks) if tasks else list(names)
    stem, ext = os.path.splitext(path)
    if verbose:
        print(f"making video for {len(todo)} task(s): {', '.join(todo)}  ({n} trials each)")

    clips = []
    for k, name in enumerate(todo, 1):
        if verbose:
            print(f"[{k}/{len(todo)}] {name}: rolling out ...")
        eff, states, inp, desired, extras = rollout_experiment(
            folder, n=n, task=name, device=device, seed=seed,
            obs_noise=obs_noise, neural_noise=neural_noise, verbose=verbose)
        clip = f"{stem}_{name}{ext}" if len(todo) > 1 else path
        if verbose:
            print(f"[{k}/{len(todo)}] {name}: rendering {n} trials -> {os.path.basename(clip)} ...")
        written = video_trials(eff, states, inp, desired, clip, num_trial=n,
                               task_name=name, **video_kw)
        clips.extend(written)
        if verbose:
            print(f"[{k}/{len(todo)}] {name}: done ({os.path.getsize(written[0]) / 1e6:.1f} MB)")

    if len(clips) > 1 and ext.lower() == ".mp4":
        if verbose:
            print(f"concatenating {len(clips)} clips -> {os.path.basename(path)} ...")
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
                for c in clips:
                    fh.write(f"file '{os.path.abspath(c)}'\n")
                listfile = fh.name
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", listfile, "-c", "copy", path], check=True)
            os.unlink(listfile)
            if verbose:
                print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB); "
                      f"per-task clips kept alongside it")
            return [path] + clips
        except Exception as e:                          # ffmpeg missing / concat failed
            if verbose:
                print(f"  concat skipped ({type(e).__name__}); per-task clips written instead")
    if verbose:
        print("done:", ", ".join(os.path.basename(c) for c in clips))
    return clips


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
    elif cfg.get("task") == "horizon_sequence":
        spec.setdefault("init_steps", 40);  spec.setdefault("delay_steps", 50)
        spec.setdefault("dwell_steps", 50); spec.setdefault("final_steps", 40)
    elif cfg.get("task") in ("hold_posture_pulse", "hold_posture_ramp"):
        pass                                                    # timing sampled by the task itself
    elif cfg.get("task") == "pursuit":
        pass                                                    # timing sampled by the task itself
    elif cfg.get("task") == "pacman":
        pass                                                    # timing/conditions handled by the task
    else:                                                        # delayed_reach_posture
        spec.setdefault("init_steps", 40);  spec.setdefault("delay_steps", 50)
        spec.setdefault("move_steps", 120); spec.setdefault("final_steps", 40)
        spec.setdefault("final_input", cfg.get("final_input") or "null")
    return spec


def spec_center_out(cfg, effector, n_dirs=8, radius=0.10):
    """Center-out reach set: start from a single central posture and fan out to n_dirs targets
    on a circle. Timing matches the trained task. Returns a flat spec dict, or None for tasks
    this spec doesn't apply to (horizon_sequence)."""
    if cfg.get("task") == "horizon_sequence":
        return None
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
    matches the trained task. Returns a flat spec dict, or None for horizon_sequence.
    """
    if cfg.get("task") == "horizon_sequence":
        return None
    pts = np.asarray(points_xy if points_xy is not None else POINT2POINT_XY, dtype=np.float32)
    starts = [pts[i] for i in range(len(pts)) for j in range(len(pts)) if i != j]
    targets = [pts[j] for i in range(len(pts)) for j in range(len(pts)) if i != j]
    spec = {"start_space": "cartesian", "target_space": "cartesian",
            "start": np.stack(starts).tolist(), "target": np.stack(targets).tolist()}
    return _apply_task_timing(cfg, spec)


# name -> builder(cfg, effector). A builder may return None when the spec doesn't apply to the
# trained task (e.g. reach specs on horizon_sequence and vice versa); such specs are skipped.
def spec_sequence(cfg, effector):
    """One fixed target sequence run at horizon 1, 2, and 3 (3 trials), so the same movement
    can be compared across planning horizons. Deterministic (seeded) start + targets sampled
    from the trained workspace; timing at each range's midpoint via the task's own defaults.
    Only applies to horizon_sequence models."""
    if cfg.get("task") != "horizon_sequence":
        return None
    R = int(cfg.get("n_reaches") or 7)
    g = torch.Generator().manual_seed(0)
    if effector.name == "point_mass":
        lo, hi = effector.pos_range
        pts = torch.rand(R + 1, 2, generator=g) * (hi - lo) + lo
        start, targets = pts[0], pts[1:]
        start_space = "cartesian"
        start_val = start.tolist()
    else:
        sho = torch.rand(R + 1, 1, generator=g) * (effector.sho_range[1] - effector.sho_range[0]) \
              + effector.sho_range[0]
        elb = torch.rand(R + 1, 1, generator=g) * (effector.elb_range[1] - effector.elb_range[0]) \
              + effector.elb_range[0]
        joints = torch.cat([sho, elb], dim=1)
        targets = effector.joint_to_cart(joints[1:].to(effector.device)).cpu()
        start_space = "joint"
        start_val = joints[0].tolist()
    spec = {"start_space": start_space, "target_space": "cartesian",
            "start": start_val, "targets": targets.tolist(),
            "horizon": [1, 2, 3]}                          # same sequence, three horizons
    return _apply_task_timing(cfg, spec)


def spec_hold(cfg, effector, n_dirs=8):
    """Hold-posture bump set: hold at the workspace-center posture while a fixed-magnitude bump
    is delivered in each of n_dirs evenly-spaced directions. Onset/duration at the range
    midpoints (deterministic). Only applies to the hold_posture_* tasks."""
    if cfg.get("task") not in ("hold_posture_pulse", "hold_posture_ramp"):
        return None
    if effector.name == "point_mass":
        lo, hi = effector.pos_range
        center = [[0.5 * (lo + hi)] * 2]
        start_space = "cartesian"
    else:
        sho = 0.5 * (effector.sho_range[0] + effector.sho_range[1])
        elb = 0.5 * (effector.elb_range[0] + effector.elb_range[1])
        center = [[sho, elb]]
        start_space = "joint"
    angles = np.linspace(0, 360, n_dirs, endpoint=False).tolist()
    amp = 0.5 * (cfg.get("force_range_n") or [1.0, 5.0])[0] + \
          0.5 * (cfg.get("force_range_n") or [1.0, 5.0])[1]
    spec = {"start_space": start_space, "start": center[0],
            "angle_deg": angles, "amp_n": amp}
    return _apply_task_timing(cfg, spec)


def spec_pursuit(cfg, effector, n_trials=6):
    """A handful of pursuit trials from the workspace-center posture (distinct random paths per
    trial, reproducible via the harness seed). Only applies to the pursuit task."""
    if cfg.get("task") != "pursuit":
        return None
    if effector.name == "point_mass":
        lo, hi = effector.pos_range
        center = [0.5 * (lo + hi)] * 2
        start_space = "cartesian"
    else:
        sho = 0.5 * (effector.sho_range[0] + effector.sho_range[1])
        elb = 0.5 * (effector.elb_range[0] + effector.elb_range[1])
        center = [sho, elb]
        start_space = "joint"
    spec = {"start_space": start_space, "start": [center] * n_trials, "catch": False}
    return _apply_task_timing(cfg, spec)


def spec_pacman(cfg, effector):
    """One trial per force condition from the workspace-center posture, so the network's generated
    force can be compared against every target profile. Only applies to the pacman task."""
    if cfg.get("task") != "pacman":
        return None
    conds = cfg.get("conditions") or list(PacManTask.CONDITIONS)
    if effector.name == "point_mass":
        lo, hi = effector.pos_range
        center = [0.5 * (lo + hi)] * 2
        start_space = "cartesian"
    else:
        sho = 0.5 * (effector.sho_range[0] + effector.sho_range[1])
        elb = 0.5 * (effector.elb_range[0] + effector.elb_range[1])
        center = [sho, elb]
        start_space = "joint"
    spec = {"start_space": start_space, "start": [center] * len(conds),
            "condition": conds, "catch": False}
    return _apply_task_timing(cfg, spec)


BUILTIN_SPECS = {"center_out": spec_center_out, "point2point": spec_point2point,
                 "sequence": spec_sequence, "hold": spec_hold, "pursuit": spec_pursuit,
                 "pacman": spec_pacman}


# ----------------------------------------------------------------------------- run one folder
def _run_one_spec(folder, name, spec, eff, controller, task, cfg,
                  obs_noise, neural_noise, num_plot, seed):
    """Roll out a single named spec and save figures + arrays under <folder>/test/<name>/."""
    torch.manual_seed(seed)
    theta0, inp, desired, pert, ts = task.make_batch(spec=spec)

    with torch.no_grad():
        states = eff.rollout(controller, theta0, inp, pert,
                             obs_noise=obs_noise, neural_noise=neural_noise)

    is_force = getattr(task, "is_force_task", False)
    out_dir = os.path.join(folder, "test", name)
    os.makedirs(out_dir, exist_ok=True)
    n = theta0.shape[0]

    if is_force:
        err = (states.force - desired).norm(dim=-1).mean().item()               # mean force error (N)
        fr = fig_diagnostics(eff, states, inp, desired,
                             title=f"{cfg['effector']}/{cfg['arch']} {name} (force err {err:.2f} N)",
                             num_trial=min(num_plot, n))
        fd = fr
        err_str = f"{err:.2f} N"
    else:
        err = 100 * (states.pos[:, -1, :] - desired[:, -1, :]).norm(dim=1).mean().item()
        fr = fig_reaches(states.pos, desired,
                         title=f"{cfg['effector']}/{cfg['arch']} {name} (final err {err:.1f} cm)")
        fd = fig_diagnostics(eff, states, inp, desired,
                             title=f"{name} diagnostics ({cfg['effector']}/{cfg['arch']})",
                             num_trial=min(num_plot, n))
        err_str = f"{err:.2f} cm"
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
    data['module_size'] = cfg['module_size']
    np.savez(os.path.join(out_dir, "states.npz"), **data)

    with open(os.path.join(out_dir, "spec_used.json"), "w") as f:
        json.dump(spec, f, indent=2)

    print(f"      {name:12s} n={n:3d}  err {err_str}  -> {out_dir}")
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
        if s is None:                                     # builtin not applicable to this task
            print(f"      {name:12s} skipped (not applicable to task {cfg['task']!r})")
            continue
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
    ap.add_argument("--builtin", default="center_out,point2point,sequence,hold,pursuit,pacman",
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
                "'start'/'target' (and optional timing) keys -- see the test_model.py docstring.")
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