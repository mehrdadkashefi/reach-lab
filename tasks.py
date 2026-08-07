"""
tasks.py -- tasks for the effectors.

Each task owns its parameters (overridable via kwargs) and a make_batch(n) method returning:

    theta0       : (n, dof)              initial configuration, sampled via the effector
    inp          : (n, steps, 3)         instruction stream  [target_x, target_y, go]
    desired      : (n, steps, 2)         desired fingertip trajectory
    perturbation : (n, steps, eff.perturbation_dim) external force/torque, or None for no
                   perturbation. xy force for the point mass, signed joint torque for the arms.
                   The effector adds it into the physics each step. To make a perturbing task,
                   build this tensor (e.g. a signed shoulder/elbow torque per timestep) instead
                   of returning None.
    timestamps   : dict of named per-trial epoch boundaries (step indices); for analysis.

desired_profile
---------------
Both tasks accept a `desired_profile` switch controlling the *shape* of the target trajectory
the position loss is regressed against during the reach:

    'step'     : desired jumps to the target the instant the go cue fires (the original
                 behaviour). The path taken to the target is unconstrained.
    'min_jerk' : desired follows the minimum-jerk straight line from start to target across
                 the movement window, then holds at the target. This rewards a straight
                 Cartesian path and a smooth bell-shaped velocity profile.

Only `desired` changes; the instruction stream `inp` (target xy, visibility, go cue) is
identical for both profiles, so the network's inputs are unchanged.
"""

import numpy as np
import torch


def _min_jerk_s(tau):
    """Minimum-jerk position interpolant s(tau) in [0,1], tau clamped to [0,1].
    s = 10 tau^3 - 15 tau^4 + 6 tau^5  (zero vel/acc at both ends)."""
    tau = tau.clamp(0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


# ----------------------------------------------------------------------------- spec helpers
# These let each task's make_batch build an explicit (deterministic) batch from a spec dict,
# in addition to its usual randomly-sampled batch. A spec is a flat dict; only "start" and
# "target" are required, everything else falls back to the task's own parameters. See the
# make_batch docstrings for the accepted keys.
def _as_col(x, n, device, dtype=torch.long):
    """Broadcast a scalar or length-n sequence to a (n, 1) tensor of the given dtype."""
    a = np.array(np.broadcast_to(np.asarray(x), (n,)))   # copy -> writable, contiguous
    return torch.as_tensor(a, device=device).to(dtype).unsqueeze(1)


def _resolve_start_target(eff, spec, device):
    """Read start/target (and their coordinate spaces) from a spec dict.

    Returns (theta0, target_xy, n): theta0 (n, dof) is the joint config used to reset the body;
    target_xy (n, 2) is the cartesian target. start_space / target_space select how the given
    arrays are interpreted ('joint' or 'cartesian'); a single row is broadcast to the other's n.
    """
    start  = torch.as_tensor(np.asarray(spec["start"],  dtype=np.float32))
    target = torch.as_tensor(np.asarray(spec["target"], dtype=np.float32))
    if start.ndim  == 1: start  = start.unsqueeze(0)
    if target.ndim == 1: target = target.unsqueeze(0)
    n = max(start.shape[0], target.shape[0])
    if start.shape[0]  == 1 and n > 1: start  = start.expand(n, -1).contiguous()
    if target.shape[0] == 1 and n > 1: target = target.expand(n, -1).contiguous()
    start, target = start.to(device), target.to(device)

    sspace = spec.get("start_space", "joint")
    if sspace == "joint":
        theta0 = start
    elif sspace == "cartesian":
        theta0 = eff.cartesian_to_joint(start)
    else:
        raise ValueError(f"start_space must be 'joint' or 'cartesian', got {sspace!r}")

    tspace = spec.get("target_space", "cartesian")
    if tspace == "cartesian":
        target_xy = target
    elif tspace == "joint":
        target_xy = eff.joint_to_cart(target)
    else:
        raise ValueError(f"target_space must be 'cartesian' or 'joint', got {tspace!r}")
    return theta0.to(device), target_xy.to(device), n


def _constant_perturbation(eff, pspec, n, T, device):
    """Build an (n, T, perturbation_dim) constant external force/torque from a perturbation spec
    {'value': (pdim,) or (n, pdim), 't_start': int, 't_end': int}, applied over [t_start, t_end).
    Returns None when pspec is None. The effector adds this into the physics each step."""
    if pspec is None:
        return None
    val = torch.as_tensor(np.asarray(pspec["value"], dtype=np.float32), device=device)
    if val.ndim == 1:
        val = val.unsqueeze(0)                       # (1, pdim) -> broadcast over trials
    pert = torch.zeros(n, T, eff.perturbation_dim, device=device)
    pert[:, int(pspec["t_start"]):int(pspec["t_end"]), :] = val.unsqueeze(1)
    return pert


def _random_perturbation(eff, n, T, device, prob, mag, dur_steps):
    """Random training perturbation: brief force/torque pulses applied to the *plant* (never
    shown to the controller), so the network must infer arm state from proprioception and learn
    a state-feedback policy rather than pure feedforward control.

    For a random `prob` fraction of trials, a pulse of random direction and random magnitude in
    [0, mag] is applied over a `dur_steps` window starting at a random onset anywhere in the
    trial (so it can land during the delay/hold as well as the movement). Units are the native
    perturbation units of the effector: xy force (N) for the point mass, signed joint torque
    (N.m) for the arms. Returns (n, T, perturbation_dim), or None when disabled."""
    if prob <= 0.0 or mag <= 0.0:
        return None
    pdim = eff.perturbation_dim
    d = max(1, int(dur_steps))
    hit = (torch.rand(n, device=device) < prob).float()                    # (n,) which trials
    onset = torch.randint(0, max(1, T - d), (n,), device=device)           # random onset step
    direction = torch.randn(n, pdim, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)
    magnitude = torch.rand(n, device=device) * mag                         # small..medium in [0,mag]
    vec = direction * (magnitude * hit).unsqueeze(1)                        # (n, pdim); 0 on un-hit
    tg = torch.arange(T, device=device).unsqueeze(0)                       # (1, T)
    win = ((tg >= onset.unsqueeze(1)) & (tg < (onset + d).unsqueeze(1))).float()   # (n, T)
    return win.unsqueeze(-1) * vec.unsqueeze(1)                            # (n, T, pdim)


class DelayedReaching:
    """Reach to a target after a go cue; some trials are no-go (hold at start)."""
    name = "delayed_reaching"

    def __init__(self, effector, steps=100, go_range=(20, 50), prob_no_go=0.3,
                 desired_profile='step', mj_move_steps=30, go_pulse_ms=150,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100, **kwargs):
        self.effector = effector
        self.steps = steps
        self.go_range = tuple(go_range)
        self.prob_no_go = prob_no_go
        # go-cue input shape: a short pulse of go_pulse_ms at go onset (default 150 ms).
        # None or <= 0 -> sustained cue that stays on after go (the old step behaviour).
        self.go_pulse = (None if (go_pulse_ms is None or go_pulse_ms <= 0)
                         else max(1, round(go_pulse_ms / 1000 / effector.dt)))
        assert desired_profile in ('step', 'min_jerk')
        self.desired_profile = desired_profile
        self.mj_move_steps = int(mj_move_steps)      # min-jerk ramp duration after the go cue
        # random training perturbations (applied to the plant, not observed by the controller)
        self.perturb_prob = float(perturb_prob)
        self.perturb_mag = float(perturb_mag)
        self.perturb_dur = max(1, round(perturb_dur_ms / 1000 / effector.dt))

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional except start/target; scalars broadcast, or give length-n lists):
            start / target           : initial config and target (see start_space/target_space)
            start_space  ('joint')   : 'joint' or 'cartesian'
            target_space ('cartesian'): 'cartesian' or 'joint'
            go_time                  : go-cue onset step (default: midpoint of go_range)
            go_pulse_steps           : go-cue pulse length in steps (default: task's go_pulse_ms;
                                       0 or negative -> sustained cue)
            no_go        (False)     : hold-at-start trials
            steps                    : episode length (default: self.steps)
            perturbation             : {'value', 't_start', 't_end'} or None
        """
        eff, dev, steps = self.effector, self.effector.device, self.steps

        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0  = eff.sample_joint(n)
            target  = eff.joint_to_cart(eff.sample_joint(n))
            go_time = torch.randint(self.go_range[0], self.go_range[1], (n, 1), device=dev)
            nogo    = torch.rand(n, 1, device=dev) < self.prob_no_go
            perturbation = _random_perturbation(eff, n, steps, dev,
                                                self.perturb_prob, self.perturb_mag, self.perturb_dur)
        else:
            theta0, target, n = _resolve_start_target(eff, spec, dev)
            steps   = int(spec.get("steps", self.steps))
            go_def  = (self.go_range[0] + self.go_range[1]) // 2
            go_time = _as_col(spec.get("go_time", go_def), n, dev, torch.long)
            nogo    = _as_col(spec.get("no_go", False), n, dev, torch.bool)
            perturbation = _constant_perturbation(eff, spec.get("perturbation"), n, steps, dev)

        start = eff.joint_to_cart(theta0)
        tgrid = torch.arange(steps, device=dev).unsqueeze(0).expand(n, steps)
        go_mask = (tgrid >= go_time) & ~nogo          # post-go period; drives `desired` below

        # go-cue *input*: a short pulse at go onset (not the sustained go_mask).
        pulse = self.go_pulse
        if spec is not None and spec.get("go_pulse_steps") is not None:
            p = int(spec["go_pulse_steps"])
            pulse = p if p > 0 else None
        go_sig = go_mask if pulse is None else (go_mask & (tgrid < go_time + pulse))

        inp = torch.zeros(n, steps, 4, device=dev)
        inp[:, :, 0:2] = target.unsqueeze(1)
        inp[:, :, 2]   = 1.0              # target always visible in this task
        inp[:, :, 3]   = go_sig.float()

        if self.desired_profile == 'step':
            desired = torch.where(go_mask.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))
        else:  # 'min_jerk': straight-line min-jerk ramp over mj_move_steps after the go cue
            tau = (tgrid - go_time).float() / max(1, self.mj_move_steps)
            s = _min_jerk_s(tau) * (~nogo).float()                       # (n, steps); 0 on no-go
            desired = start.unsqueeze(1) + (target - start).unsqueeze(1) * s.unsqueeze(-1)

        # per-trial epoch timestamps (step indices); not used in training, handy for analysis
        timestamps = {
            'go_start':    go_time.squeeze(-1),                                   # movement onset
            'episode_end': torch.full((n,), steps, dtype=torch.long, device=dev),
            'is_no_go':    nogo.squeeze(-1),
        }
        return theta0, inp, desired, perturbation, timestamps


class DelayedReachPosture:
    """Memory-guided delayed reach with four randomized segments per trial:

        1. initial hold  : hold at the (random) start; target NOT shown (xy = null_value),
                           go = 0, desired = start.            duration ~ init_range_ms
        2. delay         : target shown, but go = 0; still hold at start.
                                                               duration ~ delay_range_ms
        3. movement      : target shown; the go cue fires as a short pulse (go_pulse_ms,
                           default 150 ms) at movement onset, then returns to 0; reach to
                           the target.                         duration = move_ms (fixed)
        4. final hold    : hold at the target. go = 0, desired = target. The instruction xy
                           is either the null_value (final_input='null', default -- mirrors
                           the initial hold, so the network must hold from memory) or the
                           target (final_input='target').     duration ~ final_range_ms

    A fraction `prob_no_go` of trials are no-go: the target is still shown, but the go cue is
    never flipped to 1 and the desired stays at the start posture for the whole episode, so the
    arm must hold and not move.

    The episode length is fixed (sum of each segment's max). Trials are right-aligned, so the
    delay / movement / final-hold durations are honored exactly as sampled and any slack
    extends the (uninformative) initial hold.
    """
    name = "delayed_reach_posture"

    def __init__(self, effector, init_range_ms=(300, 700), delay_range_ms=(300, 700),
                 move_ms=1200, final_range_ms=(300, 700),
                 final_input='null', prob_no_go=0.4, desired_profile='step',
                 go_pulse_ms=150,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100, **kwargs):
        self.effector = effector
        self.prob_no_go = prob_no_go
        # go-cue input shape: a short pulse of go_pulse_ms at movement onset (default 150 ms).
        # None or <= 0 -> sustained cue that stays on for the whole move window (old behaviour).
        self.go_pulse = (None if (go_pulse_ms is None or go_pulse_ms <= 0)
                         else max(1, round(go_pulse_ms / 1000 / effector.dt)))
        assert final_input in ('null', 'target')
        self.final_input = final_input
        assert desired_profile in ('step', 'min_jerk')
        self.desired_profile = desired_profile
        # random training perturbations (applied to the plant, not observed by the controller)
        self.perturb_prob = float(perturb_prob)
        self.perturb_mag = float(perturb_mag)
        self.perturb_dur = max(1, round(perturb_dur_ms / 1000 / effector.dt))

        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.init_lo,  self.init_hi  = ms2steps(init_range_ms[0]),  ms2steps(init_range_ms[1])
        self.delay_lo, self.delay_hi = ms2steps(delay_range_ms[0]), ms2steps(delay_range_ms[1])
        self.move = ms2steps(move_ms)
        self.final_lo, self.final_hi = ms2steps(final_range_ms[0]), ms2steps(final_range_ms[1])
        self.steps = self.init_hi + self.delay_hi + self.move + self.final_hi

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        Random trials are right-aligned with sampled segment lengths (as before). Spec trials
        are left-aligned with the segment lengths given (the final hold stretches to fill the
        episode), the episode length T being the longest init+delay+move+final across trials.

        spec keys (all optional except start/target; scalars broadcast, or give length-n lists):
            start / target           : initial config and target (see start_space/target_space)
            start_space  ('joint')   : 'joint' or 'cartesian'
            target_space ('cartesian'): 'cartesian' or 'joint'
            init_steps / delay_steps / move_steps / final_steps : segment durations in steps
                                       (defaults: each range's midpoint, move = self.move)
            no_go        (False)     : hold-at-start trials
            go_pulse_steps           : go-cue pulse length in steps (default: task's go_pulse_ms;
                                       0 or negative -> sustained cue)
            final_input              : 'null' or 'target' (default: self.final_input)
            perturbation             : {'value', 't_start', 't_end'} or None
        """
        eff, dev = self.effector, self.effector.device
        mid = lambda lo, hi: (lo + hi) // 2

        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = eff.sample_joint(n)
            target = eff.joint_to_cart(eff.sample_joint(n))                 # (n, 2) cartesian
            T = self.steps
            d1 = torch.randint(self.init_lo,  self.init_hi + 1,  (n, 1), device=dev)
            d2 = torch.randint(self.delay_lo, self.delay_hi + 1, (n, 1), device=dev)
            d4 = torch.randint(self.final_lo, self.final_hi + 1, (n, 1), device=dev)
            pre = T - (d1 + d2 + self.move + d4)                            # slack -> initial hold (>=0)
            t1 = pre + d1                                                   # init -> delay
            t2 = t1 + d2                                                    # delay -> move
            t3 = t2 + self.move                                             # move -> final
            nogo = torch.rand(n, 1, device=dev) < self.prob_no_go          # (n, 1)
            final_input = self.final_input
            perturbation = _random_perturbation(eff, n, T, dev,
                                                self.perturb_prob, self.perturb_mag, self.perturb_dur)
        else:
            theta0, target, n = _resolve_start_target(eff, spec, dev)
            init  = _as_col(spec.get("init_steps",  mid(self.init_lo,  self.init_hi)),  n, dev)
            delay = _as_col(spec.get("delay_steps", mid(self.delay_lo, self.delay_hi)), n, dev)
            move  = _as_col(spec.get("move_steps",  self.move),                         n, dev)
            final = _as_col(spec.get("final_steps", mid(self.final_lo, self.final_hi)), n, dev)
            T = int((init + delay + move + final).max().item())
            t1 = init                                                      # init -> delay
            t2 = init + delay                                              # delay -> move
            t3 = init + delay + move                                       # move -> final (-> T)
            nogo = _as_col(spec.get("no_go", False), n, dev, torch.bool)
            final_input = spec.get("final_input", self.final_input)
            perturbation = _constant_perturbation(eff, spec.get("perturbation"), n, T, dev)

        start = eff.joint_to_cart(theta0)                                  # (n, 2)
        tg = torch.arange(T, device=dev).unsqueeze(0)                      # (1, T)
        in_delay = (tg >= t1) & (tg < t2)
        in_move  = (tg >= t2) & (tg < t3)
        in_final = tg >= t3

        # go-cue *input*: a short pulse at movement onset (clipped to the move window);
        # sustained over the whole move window when the pulse is disabled.
        pulse = self.go_pulse
        if spec is not None and spec.get("go_pulse_steps") is not None:
            p = int(spec["go_pulse_steps"])
            pulse = p if p > 0 else None
        go_win = in_move if pulse is None else (in_move & (tg < t2 + pulse))
        go = go_win.float() * (~nogo).float()                             # no go cue on no-go trials
        show_target = in_delay | in_move
        if final_input == 'target':
            show_target = show_target | in_final

        vis = show_target.float().unsqueeze(-1)                # (n, T, 1)
        xy  = target.unsqueeze(1) * vis                        # zero when target hidden
        inp = torch.cat([xy, vis, go.unsqueeze(-1)], dim=-1)   # (n, T, 4)

        if self.desired_profile == 'step':
            reach = (in_move | in_final) & ~nogo                          # no-go trials hold at start
            desired = torch.where(reach.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))
        else:  # 'min_jerk': hold at start, min-jerk straight line across the move window, hold target
            move_dur = (t3 - t2).clamp(min=1)                             # (n, 1) steps
            tau = (tg - t2).float() / move_dur.float()                    # (n, T): <0 pre, 0..1 move, >1 final
            s = _min_jerk_s(tau) * (~nogo).float()                        # (n, T); 0 on no-go (hold start)
            desired = start.unsqueeze(1) + (target - start).unsqueeze(1) * s.unsqueeze(-1)

        # per-trial epoch boundaries (step indices); not used in training, handy for analysis
        timestamps = {
            'init_start':  torch.zeros(n, dtype=torch.long, device=dev),    # initial hold begins
            'delay_start': t1.squeeze(-1).long(),                           # target appears
            'move_start':  t2.squeeze(-1).long(),                           # go onset (movement)
            'final_start': t3.squeeze(-1).long(),                           # movement end / final hold
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'is_no_go':    nogo.squeeze(-1),
        }
        return theta0, inp, desired, perturbation, timestamps


TASKS = {'delayed_reach': DelayedReaching, 'delayed_reach_posture': DelayedReachPosture}

def make_task(name, effector, **kwargs):
    return TASKS[name](effector, **kwargs)