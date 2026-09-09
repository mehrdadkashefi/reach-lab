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

import math

import numpy as np
import torch


# Unified instruction width shared across every task (matches HorizonSequence's stream):
# 3 target slots x [x, y, on] + go. Single-target tasks can opt into this wider layout via
# unified_input=True so one network can be trained on all tasks with a fixed input_dim.
UNIFIED_INPUT_CHANNELS = 10


def _pack_unified(inp4):
    """Remap a native single-target stream (n, T, 4) = [x, y, on, go] into the 10-wide unified
    layout: the target goes in slot 0, slots 1-2 stay off, and the go cue moves to the last
    channel. Leaves the desired trajectory and everything else untouched."""
    n, T, _ = inp4.shape
    u = torch.zeros(n, T, UNIFIED_INPUT_CHANNELS, device=inp4.device, dtype=inp4.dtype)
    u[..., 0:3] = inp4[..., 0:3]                      # slot-0 target: x, y, on
    u[..., -1]  = inp4[..., 3]                        # go -> last channel
    return u


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
    input_channels = 4                    # [target_x, target_y, target_visible, go]
    supports_unified = True               # can emit the 10-wide unified stream instead

    def __init__(self, effector, steps=100, go_range=(20, 50), prob_no_go=0.3,
                 desired_profile='step', mj_move_steps=30, go_pulse_ms=150,
                 unified_input=False,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100, **kwargs):
        self.effector = effector
        self.unified_input = bool(unified_input)
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
        if self.unified_input:
            inp = _pack_unified(inp)

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
    input_channels = 4                    # [target_x, target_y, target_visible, go]
    supports_unified = True               # can emit the 10-wide unified stream instead

    def __init__(self, effector, init_range_ms=(300, 700), delay_range_ms=(300, 700),
                 move_ms=1200, final_range_ms=(300, 700),
                 final_input='null', prob_no_go=0.4, desired_profile='step',
                 go_pulse_ms=150, unified_input=False,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100, **kwargs):
        self.effector = effector
        self.unified_input = bool(unified_input)
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
        if self.unified_input:
            inp = _pack_unified(inp)

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


class HorizonSequence:
    """Delayed *sequence* of reaches with a limited planning horizon.

    Trial structure (segments sampled per trial; random trials are right-aligned so any
    slack extends the uninformative initial hold, like DelayedReachPosture):

        1. initial hold : hand at the start posture, no targets shown, go = 0.
                                                          duration ~ init_range_ms
        2. delay        : the first min(horizon, n_reaches) targets are shown, go = 0;
                          keep holding at the start.       duration ~ delay_range_ms
        3. sequence     : a go *pulse* (go_pulse_ms) fires and the hand reaches target 1.
                          Each reach owns one dwell segment covering both the movement and
                          the hold at the target (the network budgets the time itself). At
                          the end of a segment the target is "captured": it disappears, the
                          remaining targets shift one slot forward (conveyor: slot 1 always
                          holds the current target), and a fresh go pulse cues the next
                          reach.                 duration ~ dwell_range_ms per reach
        4. final hold   : after the last capture all slots are off; hold at the last
                          target.                          duration ~ final_range_ms

    Horizon: each trial runs at a horizon h drawn from {1..3} (horizon_probs). Slot j
    (j = 1..3) shows the (current + j - 1)-th remaining target if j <= h and that target
    exists, else the slot is off (x = y = 0, on = 0). The instruction stream is therefore
    always 3 slots x [x, y, on] + go = 10 channels.

    No-go variants (independent):
        prob_no_go       : the go pulse never fires; the targets stay as in the delay and
                           the hand must hold at the start posture the whole episode.
        prob_no_go_reach : one random reach r (1 <= r <= n_reaches-1) receives no go pulse, so
                           the hand simply stays where it is for that segment -- reach r-1's
                           dwell is effectively extended -- and the sequence then RESUMES
                           normally at reach r+1. Target r is shown during its segment but
                           never reached (it is skipped). This teaches "move only on a pulse",
                           per reach, without ending the trial.

    The desired trajectory jumps to the current target at each go pulse ('step') or ramps
    to it with a minimum-jerk profile over mj_move_steps ('min_jerk'), then holds.
    """
    name = "horizon_sequence"
    n_slots = 3
    input_channels = 3 * n_slots + 1      # 3 x [x, y, on] + go = 10

    def __init__(self, effector, n_reaches=7,
                 init_range_ms=(300, 700), delay_range_ms=(300, 700),
                 dwell_range_ms=(400, 600), final_range_ms=(300, 700),
                 horizon_probs=(1/3, 1/3, 1/3),
                 prob_no_go=0.15, prob_no_go_reach=0.0,
                 desired_profile='step', mj_move_steps=30, go_pulse_ms=150,
                 blind_mode='delay', blind_ms=None,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100, **kwargs):
        self.effector = effector
        # How to handle the window after an *unpreviewed* go cue, during which the target has not
        # reached the controller yet (sensory delay):
        #   'delay' : the desired holds at the previous target across the window, so waiting is
        #             optimal and pre-moving is penalised  (recommended)
        #   'mask'  : the window is dropped from the loss -- removes the cost of guessing but not
        #             its benefit, so the network may still pre-move
        #   'none'  : original behaviour, desired jumps at the capture
        assert blind_mode in ('delay', 'mask', 'none')
        self.blind_mode = blind_mode
        self.blind_steps = (effector.vis_d if blind_ms is None
                            else max(0, round(blind_ms / 1000 / effector.dt)))
        if blind_mode == 'none':
            self.blind_steps = 0
        self.n_reaches = int(n_reaches)
        hp = torch.as_tensor(horizon_probs, dtype=torch.float32)
        assert hp.numel() == self.n_slots and abs(hp.sum().item() - 1.0) < 1e-4, \
            f"horizon_probs must be {self.n_slots} probabilities summing to 1"
        self.horizon_probs = hp
        self.prob_no_go = float(prob_no_go)
        self.prob_no_go_reach = float(prob_no_go_reach)
        assert desired_profile in ('step', 'min_jerk')
        self.desired_profile = desired_profile
        self.mj_move_steps = int(mj_move_steps)
        # go-cue input shape: a short pulse at the initial go and at every capture.
        # None or <= 0 -> sustained cue that stays on for the whole sequence (rarely useful).
        self.go_pulse = (None if (go_pulse_ms is None or go_pulse_ms <= 0)
                         else max(1, round(go_pulse_ms / 1000 / effector.dt)))
        # random training perturbations (applied to the plant, not observed by the controller)
        self.perturb_prob = float(perturb_prob)
        self.perturb_mag = float(perturb_mag)
        self.perturb_dur = max(1, round(perturb_dur_ms / 1000 / effector.dt))

        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.init_lo,  self.init_hi  = ms2steps(init_range_ms[0]),  ms2steps(init_range_ms[1])
        self.delay_lo, self.delay_hi = ms2steps(delay_range_ms[0]), ms2steps(delay_range_ms[1])
        self.dwell_lo, self.dwell_hi = ms2steps(dwell_range_ms[0]), ms2steps(dwell_range_ms[1])
        self.final_lo, self.final_hi = ms2steps(final_range_ms[0]), ms2steps(final_range_ms[1])
        self.steps = (self.init_hi + self.delay_hi
                      + self.n_reaches * self.dwell_hi + self.final_hi)

    # -- spec helpers ------------------------------------------------------------------
    def _resolve_spec_geometry(self, spec, dev):
        """start ((dof,) or (n, dof)) + targets ((n_reaches, 2) or (n, n_reaches, 2)) -> tensors.
        start_space ('joint') / target_space ('cartesian') select interpretation. n is the max
        leading dim over start / targets / horizon / no_go / no_go_reach (singletons broadcast).
        """
        eff, R = self.effector, self.n_reaches
        start   = np.asarray(spec["start"],   dtype=np.float32)
        targets = np.asarray(spec["targets"], dtype=np.float32)
        if start.ndim == 1:   start = start[None]                        # (1, dof)
        if targets.ndim == 2: targets = targets[None]                    # (1, R, 2)
        assert targets.shape[1] == R, \
            f"spec 'targets' must have {R} reaches per trial, got {targets.shape[1]}"
        lead = lambda k: np.asarray(spec[k]).shape[0] if (k in spec and np.asarray(spec[k]).ndim >= 1) else 1
        n = max(start.shape[0], targets.shape[0], lead("horizon"), lead("no_go"), lead("no_go_reach"))
        if start.shape[0]   == 1 and n > 1: start   = np.repeat(start, n, 0)
        if targets.shape[0] == 1 and n > 1: targets = np.repeat(targets, n, 0)

        start_t = torch.as_tensor(start, device=dev)
        if spec.get("start_space", "joint") == "joint":
            theta0 = start_t
        else:
            theta0 = eff.cartesian_to_joint(start_t)

        tt = torch.as_tensor(targets, device=dev)                        # (n, R, 2)
        if spec.get("target_space", "cartesian") == "joint":
            tt = eff.joint_to_cart(tt.reshape(n * R, -1)).reshape(n, R, 2)
        return theta0.to(dev), tt.to(dev), n

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (start/targets required; scalars broadcast, or give length-n lists):
            start / targets          : start config (dof,) and target sequence (n_reaches, 2)
                                       per trial (see start_space/target_space)
            start_space  ('joint')   : 'joint' or 'cartesian'
            target_space ('cartesian'): 'cartesian' or 'joint'
            horizon                  : 1..3 (default 3); scalar or length-n
            init_steps / delay_steps / final_steps : segment durations in steps
                                       (defaults: each range's midpoint)
            dwell_steps              : per-reach segment length; scalar, (n_reaches,) or
                                       (n, n_reaches) (default: dwell range midpoint)
            no_go        (False)     : trial-level no-go (never any go pulse)
            no_go_reach  (-1)        : index (1..n_reaches-1) of the reach whose pulse is
                                       withheld -- the hand holds through that segment and the
                                       sequence resumes at the next reach; -1 = none
            go_pulse_steps           : go-cue pulse length in steps (default: task's
                                       go_pulse_ms; 0 or negative -> sustained cue)
            perturbation             : {'value', 't_start', 't_end'} or None
        """
        eff, dev, R = self.effector, self.effector.device, self.n_reaches
        mid = lambda lo, hi: (lo + hi) // 2

        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0  = eff.sample_joint(n)
            targets = eff.joint_to_cart(eff.sample_joint(n * R)).reshape(n, R, 2)
            T = self.steps
            d1   = torch.randint(self.init_lo,  self.init_hi + 1,  (n, 1), device=dev)
            d2   = torch.randint(self.delay_lo, self.delay_hi + 1, (n, 1), device=dev)
            dseg = torch.randint(self.dwell_lo, self.dwell_hi + 1, (n, R), device=dev)
            d4   = torch.randint(self.final_lo, self.final_hi + 1, (n, 1), device=dev)
            pre  = T - (d1 + d2 + dseg.sum(1, keepdim=True) + d4)       # slack -> initial hold
            t_delay = pre + d1
            t_go    = t_delay + d2
            horizon = torch.multinomial(self.horizon_probs.to(dev), n, replacement=True) + 1
            nogo    = torch.rand(n, device=dev) < self.prob_no_go                     # (n,)
            hit_r   = (torch.rand(n, device=dev) < self.prob_no_go_reach) & ~nogo & (R > 1)
            ridx    = torch.randint(1, max(2, R), (n,), device=dev)
            no_go_reach = torch.where(hit_r, ridx, torch.full_like(ridx, -1))         # (n,)
            pulse = self.go_pulse
            perturbation = _random_perturbation(eff, n, T, dev,
                                                self.perturb_prob, self.perturb_mag, self.perturb_dur)
        else:
            theta0, targets, n = self._resolve_spec_geometry(spec, dev)
            d1 = _as_col(spec.get("init_steps",  mid(self.init_lo,  self.init_hi)),  n, dev)
            d2 = _as_col(spec.get("delay_steps", mid(self.delay_lo, self.delay_hi)), n, dev)
            dw = np.broadcast_to(np.asarray(spec.get("dwell_steps",
                                                     mid(self.dwell_lo, self.dwell_hi))), (n, R))
            dseg = torch.as_tensor(np.array(dw), device=dev).long()                  # (n, R)
            d4 = _as_col(spec.get("final_steps", mid(self.final_lo, self.final_hi)), n, dev)
            T = int((d1 + d2 + dseg.sum(1, keepdim=True) + d4).max().item())
            t_delay = d1                                                # left-aligned
            t_go    = d1 + d2
            horizon = _as_col(spec.get("horizon", self.n_slots), n, dev).squeeze(1)
            horizon = horizon.clamp(1, self.n_slots)
            nogo    = _as_col(spec.get("no_go", False), n, dev, torch.bool).squeeze(1)
            no_go_reach = _as_col(spec.get("no_go_reach", -1), n, dev).squeeze(1)
            pulse = self.go_pulse
            if spec.get("go_pulse_steps") is not None:
                p = int(spec["go_pulse_steps"])
                pulse = p if p > 0 else None
            perturbation = _constant_perturbation(eff, spec.get("perturbation"), n, T, dev)

        start = eff.joint_to_cart(theta0)                               # (n, 2)
        bounds = t_go + dseg.cumsum(1)                                  # (n, R) capture times
        pulse_times = torch.cat([t_go, bounds[:, :-1]], dim=1)          # (n, R) go for reach k
        final_start = bounds[:, -1]                                     # (n,)

        # Two independent no-go variants, both implemented here:
        #   trial no-go (cap = 0): no pulse ever fires and the hand holds at the start posture for
        #       the whole episode; the target display freezes at the beginning of the sequence.
        #   reach no-go (skip = r): reach r alone receives no pulse, so the hand simply stays where
        #       it is for that segment -- reach r-1's dwell is effectively extended -- and the
        #       sequence then RESUMES normally at reach r+1 (target r is skipped, never reached).
        cap = torch.full((n,), R, dtype=torch.long, device=dev)
        cap = torch.where(nogo, torch.zeros_like(cap), cap)
        skip = no_go_reach.clone()                                      # (n,) -1 = no skipped reach
        skip = torch.where(nogo, torch.full_like(skip, -1), skip)       # variants are exclusive
        skip_col = skip.unsqueeze(1)                                    # (n, 1)
        has_skip = (skip >= 0).unsqueeze(1)

        tg = torch.arange(T, device=dev).unsqueeze(0)                   # (1, T)
        cur = (tg.unsqueeze(-1) >= bounds.unsqueeze(1)).sum(-1)         # (n, T) captures so far
        cur_eff = torch.minimum(cur, cap.unsqueeze(1))                  # frozen on no-go variants
        shown_time = tg >= t_delay                                      # targets visible from delay

        # ---- instruction stream: 3 slots x [x, y, on] + go --------------------------------
        inp = torch.zeros(n, T, self.input_channels, device=dev)
        tx, ty = targets[..., 0], targets[..., 1]                       # (n, R)
        for j in range(self.n_slots):
            idx = cur_eff + j                                           # target index in slot j
            on = (idx < R) & (j < horizon.unsqueeze(1)) & shown_time
            idxc = idx.clamp(max=R - 1)
            inp[:, :, 3 * j]     = tx.gather(1, idxc) * on.float()
            inp[:, :, 3 * j + 1] = ty.gather(1, idxc) * on.float()
            inp[:, :, 3 * j + 2] = on.float()

        # go channel: one pulse per cued reach, at t_go and at each capture. A skipped reach (and
        # every reach on a trial no-go) gets no pulse; the rest are unaffected.
        go = torch.zeros(n, T, device=dev)
        if pulse is None:                                               # sustained fallback
            go = ((tg >= t_go) & (cur < cap.unsqueeze(1))
                  & ~(has_skip & (cur == skip_col))).float()
        else:
            for k in range(R):
                pk = pulse_times[:, k:k + 1]                            # (n, 1)
                win = (tg >= pk) & (tg < pk + pulse) & (cur == k)       # clipped to segment k
                win = win & (k < cap).unsqueeze(1) & ~(has_skip & (skip_col == k))
                go = torch.maximum(go, win.float())
        inp[:, :, 3 * self.n_slots] = go

        # ---- which reaches are unpreviewed? ---------------------------------------------------
        # Target k first enters a lit slot during segment max(0, k - h + 1); it is unpreviewed
        # exactly when that is its own segment (horizon 1) and it is not the first reach (whose
        # target is visible throughout the delay period).
        ks = torch.arange(R, device=dev).view(1, R)                     # (1, R)
        unpreviewed = (((ks - horizon.view(n, 1) + 1).clamp(min=0) == ks) & (ks > 0)
                       & (ks < cap.view(n, 1)) & ~(has_skip & (skip_col == ks)))

        # ---- desired trajectory ------------------------------------------------------------
        # The go cue and the new target reach the controller one visual delay after the capture,
        # so during that window an unpreviewed reach CANNOT know where to go. With
        # blind_mode='delay' the desired stays at the previous target across the window and only
        # then switches, which makes waiting optimal: drifting toward the average target is now
        # actively penalised. (Merely masking the window -- blind_mode='mask' -- removes the
        # in-window cost of guessing but not its benefit, since a hand that has drifted centre-ward
        # begins the scored part of the reach closer to the average target.)
        onsets = pulse_times + (self.blind_steps * unpreviewed.long()
                                if self.blind_mode == 'delay' else 0)   # (n, R)
        cur_des = (tg.unsqueeze(-1) >= onsets.unsqueeze(1)).sum(-1)     # (n, T), 0 before reach 0
        didx = (cur_des - 1).clamp(min=0)
        didx = torch.where(has_skip & (didx == skip_col), (skip_col - 1).clamp(min=0), didx)
        didx = torch.minimum(didx, (cap - 1).clamp(min=0).unsqueeze(1)).clamp(min=0, max=R - 1)
        des_x = tx.gather(1, didx)
        des_y = ty.gather(1, didx)
        moved = (cur_des >= 1) & (cap > 0).unsqueeze(1)                  # (n, T)
        desired = torch.where(moved.unsqueeze(-1),
                              torch.stack([des_x, des_y], dim=-1),
                              start.unsqueeze(1))

        if self.desired_profile == 'min_jerk':
            # overwrite each cued segment with a min-jerk ramp prev -> target k, then hold. A
            # skipped reach gets no ramp (it is a hold), and the reach after it starts from the
            # target held during the skip, not from the skipped target.
            for k in range(R):
                pk = onsets[:, k:k + 1]                                 # segment start (n, 1)
                seg_len = (bounds[:, k:k + 1] - pk).clamp(min=1)
                mj = seg_len.clamp(max=self.mj_move_steps).float()
                s = _min_jerk_s((tg - pk).float() / mj)                 # (n, T)
                if k == 0:
                    prev = start                                        # (n, 2)
                else:
                    # normally target k-1; if k-1 was skipped the hand is still at target k-2
                    pidx = torch.full((n,), k - 1, dtype=torch.long, device=dev)
                    pidx = torch.where((skip == k - 1) & (skip >= 0),
                                       pidx - 1, pidx).clamp(min=0)
                    prev = targets[torch.arange(n, device=dev), pidx]    # (n, 2)
                ramp = prev.unsqueeze(1) + (targets[:, k] - prev).unsqueeze(1) * s.unsqueeze(-1)
                mask = ((cur_des - 1 == k) & moved & (k < cap).unsqueeze(1)
                        & ~(has_skip & (skip_col == k)))                # no ramp on a skipped reach
                desired = torch.where(mask.unsqueeze(-1), ramp, desired)

        # ---- sensory-blind loss mask ---------------------------------------------------------
        # `desired` jumps to the new target at the capture, but the whole instruction stream
        # (target AND go cue) reaches the controller one visual delay later, so over that window
        # the loss demands movement toward a target that cannot yet have been perceived. The
        # cheapest way to satisfy it is to guess the average direction and correct -- exactly the
        # spurious pre-movement seen at horizon 1. We therefore exclude that window from the loss
        # for reaches whose target was NOT already visible before the pulse. Nothing pushes the
        # network to move there any more, and the effort penalties make holding still cheapest, so
        # a reaction time emerges rather than being prescribed. Previewed reaches (all of H2/H3,
        # and the first reach of every trial, visible throughout the delay) are untouched, so they
        # can still launch immediately.
        loss_mask = torch.ones(n, T, dtype=torch.bool, device=dev)
        if self.blind_mode == 'mask' and self.blind_steps > 0:
            for k in range(R):
                pk = pulse_times[:, k:k + 1]                            # (n, 1)
                loss_mask &= ~((tg >= pk) & (tg < pk + self.blind_steps)
                               & unpreviewed[:, k:k + 1])

        # per-trial epoch boundaries (step indices); not used in training, handy for analysis
        timestamps = {
            'delay_start':   t_delay.squeeze(-1).long(),                # targets appear
            'move_start':    t_go.squeeze(-1).long(),                   # first go pulse
            'capture_times': bounds.long(),                             # (n, n_reaches)
            'final_start':   final_start.long(),                        # last capture
            'episode_end':   torch.full((n,), T, dtype=torch.long, device=dev),
            'horizon':       horizon.long(),
            'is_no_go':      nogo,
            'no_go_reach':   no_go_reach.long(),
            'loss_mask':     loss_mask,       # False where the target cannot yet be perceived
            'unpreviewed':   unpreviewed,     # (n, R) target was not visible before its own pulse
        }
        return theta0, inp, desired, perturbation, timestamps


class _HoldPostureBase:
    """Shared machinery for the hold-posture tasks: hold at a central start posture while an
    unobserved external force bumps the hand; the target/desired is always the start location,
    so the network learns to resist the bump and return to the start as fast as it can.

    The start posture is sampled from the central `center_frac` of the effector's range so the
    bump doesn't slam the arm into a joint limit. The external force is a Cartesian fingertip
    force (random direction, random magnitude in force_range_n); the effector converts it to its
    native perturbation units (xy force for the point mass, J^T joint torque for the arms) at the
    start posture. The force is applied to the plant only -- never shown in the instruction
    stream. A catch_prob fraction of trials are catch trials with no bump at all (amplitude 0),
    so the network also learns to keep holding when nothing happens. Subclasses implement
    _force_envelope() to shape the force over time.

    Instruction stream (single target = the start location, always visible, no go cue). With
    unified_input=True it is emitted in the 10-wide unified layout (slot 0 = start), otherwise
    in the native 4-wide [x, y, on, go] layout.
    """
    supports_unified = True
    input_channels = 4                    # native; 10 when unified_input=True

    def __init__(self, effector, onset_range_ms=(500, 1000), force_range_n=(1.0, 5.0),
                 settle_ms=700, center_frac=0.4, catch_prob=0.3, unified_input=False,
                 desired_profile='step', **kwargs):
        self.effector = effector
        self.unified_input = bool(unified_input)
        self.center_frac = float(center_frac)
        self.catch_prob = float(catch_prob)              # fraction of no-bump (catch) trials
        self.force_lo, self.force_hi = float(force_range_n[0]), float(force_range_n[1])
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.onset_lo, self.onset_hi = ms2steps(onset_range_ms[0]), ms2steps(onset_range_ms[1])
        self.settle = ms2steps(settle_ms)
        # episode length: latest onset + longest force window + settling time (set by subclass)
        self.steps = self.onset_hi + self._max_force_steps() + self.settle

    # -- subclass hooks --------------------------------------------------------------------
    def _max_force_steps(self):
        """Longest possible force window in steps (for sizing the episode)."""
        raise NotImplementedError

    def _sample_force_params(self, n, dev):
        """Return a dict of per-trial force parameters (subclass-specific), plus 'onset' (n,)."""
        raise NotImplementedError

    def _force_envelope(self, tg, params, dev):
        """(n, T) scalar force magnitude over time from the sampled params. tg is (1, T)."""
        raise NotImplementedError

    def _spec_force_params(self, spec, n, dev):
        """Build force params from a spec dict (explicit onset/duration/amp/angle)."""
        raise NotImplementedError

    # -- shared batch ----------------------------------------------------------------------
    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional; scalars broadcast, or give length-n lists):
            start / start_space      : explicit start posture ((dof,) or (n, dof)); default =
                                       a freshly sampled central posture. start_space 'joint'
                                       (default) or 'cartesian'.
            angle_deg                : bump direction(s) in degrees (default: evenly spread)
            amp_n                    : bump magnitude(s) in N (default: force range midpoint)
            onset_steps              : bump onset step (default: onset range midpoint)
            + subclass-specific timing (dur_steps for pulse; ramp_steps for ramp)
            steps                    : episode length (default: self.steps)
        """
        eff, dev, R = self.effector, self.effector.device, None
        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = eff.sample_center_joint(n, self.center_frac)
            T = self.steps
            params = self._sample_force_params(n, dev)
        else:
            start = spec.get("start")
            if start is None:
                raise ValueError("hold-posture spec needs a 'start' posture")
            start = torch.as_tensor(np.asarray(start, dtype=np.float32), device=dev)
            if start.ndim == 1:
                start = start.unsqueeze(0)
            # n = max leading dim across start and the per-trial bump params (singletons broadcast)
            lead = lambda k: np.asarray(spec[k]).shape[0] if (k in spec and np.asarray(spec[k]).ndim >= 1) else 1
            n = max(start.shape[0], lead("angle_deg"), lead("amp_n"),
                    lead("onset_steps"), lead("dur_steps"), lead("ramp_steps"))
            if start.shape[0] == 1 and n > 1:
                start = start.expand(n, -1).contiguous()
            if spec.get("start_space", "joint") == "cartesian":
                theta0 = eff.cartesian_to_joint(start)
            else:
                theta0 = start
            T = int(spec.get("steps", self.steps))
            params = self._spec_force_params(spec, n, dev)

        start_xy = eff.joint_to_cart(theta0)                             # (n, 2) = hold target

        # external Cartesian force -> native perturbation units, shaped over time
        tg = torch.arange(T, device=dev).unsqueeze(0)                    # (1, T)
        mag = self._force_envelope(tg, params, dev)                      # (n, T) newtons
        ang = params['angle']                                            # (n,)
        fdir = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)      # (n, 2) unit
        pert_unit = eff.cartesian_force_to_perturbation(theta0, fdir)    # (n, pdim) per 1 N
        perturbation = mag.unsqueeze(-1) * pert_unit.unsqueeze(1)        # (n, T, pdim)

        # instruction stream: single target = the start location, always visible, no go cue
        inp = torch.zeros(n, T, 4, device=dev)
        inp[:, :, 0:2] = start_xy.unsqueeze(1)
        inp[:, :, 2]   = 1.0
        if self.unified_input:
            inp = _pack_unified(inp)

        # desired trajectory: hold at the start for the whole episode
        desired = start_xy.unsqueeze(1).expand(n, T, 2).contiguous()

        onset = params['onset']
        timestamps = {
            'bump_onset':  onset.long(),
            'bump_end':    (onset + params['dur']).long(),
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'amp_n':       params['amp'],
            'angle_deg':   ang * 180.0 / np.pi,
            'is_catch':    (params['amp'] == 0),          # no external force this trial
        }
        return theta0, inp, desired, perturbation, timestamps

    # -- helpers for subclasses ------------------------------------------------------------
    def _common_random(self, n, dev):
        """Onset, direction, and amplitude shared by both hold tasks. A catch_prob fraction of
        trials are catch trials: the amplitude is zeroed, so no external force ever arrives and
        the network must simply keep holding."""
        onset = torch.randint(self.onset_lo, self.onset_hi + 1, (n,), device=dev)
        angle = torch.rand(n, device=dev) * (2 * np.pi)
        amp = torch.rand(n, device=dev) * (self.force_hi - self.force_lo) + self.force_lo
        catch = torch.rand(n, device=dev) < self.catch_prob
        amp = amp * (~catch).float()                     # no bump on catch trials
        return onset, angle, amp

    def _common_spec(self, spec, n, dev):
        onset = _as_col(spec.get("onset_steps", (self.onset_lo + self.onset_hi) // 2),
                        n, dev).squeeze(1)
        ang_deg = _as_col(spec.get("angle_deg",
                                   np.linspace(0, 360, n, endpoint=False)), n, dev,
                          torch.float32).squeeze(1)
        amp = _as_col(spec.get("amp_n", 0.5 * (self.force_lo + self.force_hi)),
                      n, dev, torch.float32).squeeze(1)
        return onset, ang_deg * np.pi / 180.0, amp


class HoldPosturePulse(_HoldPostureBase):
    """Hold posture against a brief rectangular force pulse (random onset, direction, magnitude,
    and duration). The force is constant over [onset, onset+dur) then releases to zero."""
    name = "hold_posture_pulse"

    def __init__(self, effector, dur_range_ms=(100, 200), **kwargs):
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.dur_lo, self.dur_hi = ms2steps(dur_range_ms[0]), ms2steps(dur_range_ms[1])
        super().__init__(effector, **kwargs)

    def _max_force_steps(self):
        return self.dur_hi

    def _sample_force_params(self, n, dev):
        onset, angle, amp = self._common_random(n, dev)
        dur = torch.randint(self.dur_lo, self.dur_hi + 1, (n,), device=dev)
        return {'onset': onset, 'angle': angle, 'amp': amp, 'dur': dur}

    def _spec_force_params(self, spec, n, dev):
        onset, angle, amp = self._common_spec(spec, n, dev)
        dur = _as_col(spec.get("dur_steps", (self.dur_lo + self.dur_hi) // 2), n, dev).squeeze(1)
        return {'onset': onset, 'angle': angle, 'amp': amp, 'dur': dur}

    def _force_envelope(self, tg, params, dev):
        onset, dur, amp = params['onset'].unsqueeze(1), params['dur'].unsqueeze(1), params['amp'].unsqueeze(1)
        win = (tg >= onset) & (tg < onset + dur)                         # (n, T)
        return win.float() * amp


class HoldPostureRamp(_HoldPostureBase):
    """Hold posture against a slowly growing force ramp: the force rises linearly from 0 to a
    random maximum over a random ramp duration (random onset), then releases to zero. (Set
    hold_after_ramp=True to instead sustain the max force to the end of the episode -- a
    force-field-style push rather than a transient.)"""
    name = "hold_posture_ramp"

    def __init__(self, effector, ramp_range_ms=(200, 600), hold_after_ramp=False, **kwargs):
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.ramp_lo, self.ramp_hi = ms2steps(ramp_range_ms[0]), ms2steps(ramp_range_ms[1])
        self.hold_after_ramp = bool(hold_after_ramp)
        super().__init__(effector, **kwargs)

    def _max_force_steps(self):
        # if the force is sustained after the ramp, the "window" runs to the end of the episode;
        # size the episode off the ramp itself and let settle cover the post-ramp hold.
        return self.ramp_hi

    def _sample_force_params(self, n, dev):
        onset, angle, amp = self._common_random(n, dev)
        ramp = torch.randint(self.ramp_lo, self.ramp_hi + 1, (n,), device=dev)
        return {'onset': onset, 'angle': angle, 'amp': amp, 'dur': ramp}

    def _spec_force_params(self, spec, n, dev):
        onset, angle, amp = self._common_spec(spec, n, dev)
        ramp = _as_col(spec.get("ramp_steps", (self.ramp_lo + self.ramp_hi) // 2), n, dev).squeeze(1)
        return {'onset': onset, 'angle': angle, 'amp': amp, 'dur': ramp}

    def _force_envelope(self, tg, params, dev):
        onset, ramp, amp = params['onset'].unsqueeze(1), params['dur'].unsqueeze(1), params['amp'].unsqueeze(1)
        frac = ((tg - onset).clamp(min=0).float() / ramp.float()).clamp(max=1.0)   # 0..1 ramp
        active = (tg >= onset)
        if not self.hold_after_ramp:
            active = active & (tg < onset + ramp)                       # release after the ramp
        return active.float() * frac * amp


def _wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def smooth_random_pursuit(n, T, dt, device, q0, move_onset,
                          speed=0.5, speed_range=None,
                          turn_tau_ms=400.0, speed_tau_ms=800.0,
                          curviness=1.0, margin=0.12, wall_gain=8.0,
                          generator=None):
    """Generate n smooth, workspace-spanning random trajectories in a normalized [0,1]^2 box.

    The target moves at a controlled speed in a heading that turns smoothly over time, so the
    path is continuous and never stops once it starts. Speed is either fixed (`speed`, in box
    units per second) or itself a smooth random signal within `speed_range=(min,max)`. Heading
    turns follow a smooth (Ornstein-Uhlenbeck) angular-velocity process with time constant
    `turn_tau_ms`; `curviness` scales how sharply it winds. Near a wall (within `margin`) the
    heading is steered smoothly inward and the speed is smoothly braked to zero, so the target
    decelerates, turns, and re-accelerates without ever leaving the box (no hard clamp -> no kink
    in the path). Before each trial's `move_onset` step the target holds at its start `q0`.

    Args:
        q0:         (n, 2) start position in [0,1]^2 (normalized config space).
        move_onset: (n,) step at which each trial's motion begins.
    Returns:
        q: (n, T, 2) normalized positions in [0,1]^2 (smooth; q[:, t] == q0 for t < move_onset).
    """
    assert q0.shape[1] == 2, "pursuit generator supports dof=2 effectors"
    randn = (lambda *s: torch.randn(*s, device=device, generator=generator))
    rand = (lambda *s: torch.rand(*s, device=device, generator=generator))

    turn_tau = max(dt, turn_tau_ms / 1000.0)
    a_turn = float(np.exp(-dt / turn_tau))
    omega_std = curviness / turn_tau                                  # rad/s angular-velocity std
    b_turn = float(np.sqrt(max(1e-8, 1.0 - a_turn ** 2))) * omega_std

    varying = speed_range is not None
    if varying:
        smin, smax = float(speed_range[0]), float(speed_range[1])
        s_mid = 0.5 * (smin + smax)
        speed_tau = max(dt, speed_tau_ms / 1000.0)
        a_s = float(np.exp(-dt / speed_tau))
        sigma_s = (smax - smin) * 0.35                               # per-sqrt(s) drive
        s = rand(n) * (smax - smin) + smin
    else:
        s = torch.full((n,), float(speed), device=device)

    q = q0.clone()
    phi = rand(n) * (2 * np.pi)                                       # heading
    omega = torch.zeros(n, device=device)
    onset = move_onset.to(device)

    def _smoothstep(x):
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    out = torch.empty(n, T, 2, device=device)
    with torch.no_grad():
        for t in range(T):
            out[:, t] = q
            moving = (t >= onset).float()                            # (n,)

            omega = a_turn * omega + b_turn * randn(n)               # smooth angular velocity

            # soft wall steering: inward push grows within `margin` of a wall; rotate heading
            # toward that inward direction (rotation only -- does not change speed).
            inward = (margin - q).clamp(min=0) - (q - (1.0 - margin)).clamp(min=0)   # (n, 2)
            wall_mag = inward.norm(dim=1)
            phi_in = torch.atan2(inward[:, 1], inward[:, 0])
            steer = wall_gain * wall_mag * _wrap_angle(phi_in - phi)
            phi = phi + (omega + steer) * dt * moving

            if varying:
                s = a_s * s + (1 - a_s) * s_mid + sigma_s * float(np.sqrt(dt)) * randn(n)
                s = s.clamp(smin, smax)

            dir_xy = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)            # (n, 2) unit
            # smooth per-axis wall brake: damp only the velocity component heading into a wall
            # (within `margin` of it), so the target curves along the wall at speed instead of
            # stalling, and never leaves [0,1]. Smoothstep keeps it C1-smooth.
            ahead = torch.where(dir_xy >= 0, 1.0 - q, q)            # (n, 2) dist to approaching wall
            brake = _smoothstep(ahead / margin)                    # (n, 2) per-axis, 1 interior, 0 at wall
            v = s.unsqueeze(1) * dir_xy * brake
            q = q + v * dt * moving.unsqueeze(1)
            q = q.clamp(0.0, 1.0)                                    # ultimate safety (should be inactive)
    return out


class PursuitTask:
    """Continuous random-pursuit tracking with a rolling preview and a go cue (same structure as
    the PacMan task, but for a moving *position* target instead of an isometric force).

    A smooth random trajectory M is generated over the execution window (exec_ms) in normalized
    config space (so every point is reachable), then mapped to Cartesian. The trial then wraps it
    exactly like PacMan:
        - Preview: the upcoming trajectory scrolls through the [x, y, on, go] input from t=0,
          leading the required hand motion by preview_ms (default 1000). During the preview the
          hand holds at the start posture (desired = start).
        - A go-cue pulse fires at the end of the preview; from then the hand tracks M from its
          beginning, i.e. desired(t) = M(t - preview). The input keeps leading by preview_ms, so
          near the trial's end the target preview has run out and the hand must track from memory.
        - Catch trials (prob_catch, default 0.2): no go cue, and the hand holds at the start for
          the whole trial despite the moving preview.

    Speed is fixed (speed) or a smooth random signal (speed_range); heading turns smoothly
    (turn_tau_ms, curviness). All timing/speed knobs are switchable, matching PacMan.
    """
    name = "pursuit"
    input_channels = 4                    # [target_x, target_y, target_visible, go]
    supports_unified = True

    def __init__(self, effector, exec_ms=2000, preview_ms=1000, prob_catch=0.2, go_pulse_ms=150,
                 speed=0.5, speed_range=None, turn_tau_ms=400.0, speed_tau_ms=800.0,
                 curviness=1.0, center_frac=0.5, unified_input=False,
                 desired_profile='step', **kwargs):
        self.effector = effector
        self.unified_input = bool(unified_input)
        self.center_frac = float(center_frac)
        self.speed = float(speed)
        self.speed_range = None if speed_range is None else (float(speed_range[0]), float(speed_range[1]))
        self.turn_tau_ms = float(turn_tau_ms)
        self.speed_tau_ms = float(speed_tau_ms)
        self.curviness = float(curviness)
        self.prob_catch = float(prob_catch)
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.exec_steps = ms2steps(exec_ms)
        self.preview_steps = max(0, round(preview_ms / 1000 / effector.dt))
        self.steps = self.preview_steps + self.exec_steps
        self.go_pulse = (None if (go_pulse_ms is None or go_pulse_ms <= 0)
                         else max(1, round(go_pulse_ms / 1000 / effector.dt)))

    def _targets_from_norm(self, q_norm):
        """Map normalized (n, T, 2) config-box coords to cartesian target_xy (n, T, 2)."""
        lo, hi = self.effector.config_range
        lo = lo.view(1, 1, 2); hi = hi.view(1, 1, 2)
        q_cfg = lo + q_norm * (hi - lo)
        n, T, _ = q_cfg.shape
        return self.effector.joint_to_cart(q_cfg.reshape(n * T, 2)).reshape(n, T, 2)

    def _gen_motion(self, n, spec):
        """Generate the raw moving trajectory M (n, exec_steps, 2, cartesian) that plays from the
        start posture at step 0 (the preview/hold/go wrapping is added by make_batch). Returns
        (theta0, M, n)."""
        eff, dev = self.effector, self.effector.device
        lo, hi = eff.config_range
        lo2, hi2 = lo.view(1, 2), hi.view(1, 2)
        L = self.exec_steps

        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = eff.sample_center_joint(n, self.center_frac)
            speed, speed_range = self.speed, self.speed_range
            curviness, turn_tau = self.curviness, self.turn_tau_ms
        else:
            start = spec.get("start")
            if start is None:
                theta0 = eff.sample_center_joint(int(spec.get("n", 1)), self.center_frac)
            else:
                start = torch.as_tensor(np.asarray(start, dtype=np.float32), device=dev)
                if start.ndim == 1:
                    start = start.unsqueeze(0)
                theta0 = (eff.cartesian_to_joint(start)
                          if spec.get("start_space", "joint") == "cartesian" else start)
            n = theta0.shape[0]
            speed = float(spec.get("speed", self.speed))
            speed_range = spec.get("speed_range", self.speed_range)
            curviness = float(spec.get("curviness", self.curviness))
            turn_tau = float(spec.get("turn_tau_ms", self.turn_tau_ms))

        q0 = ((theta0 - lo2) / (hi2 - lo2)).clamp(0.0, 1.0)
        onset = torch.zeros(n, dtype=torch.long, device=dev)         # motion from step 0
        q_norm = smooth_random_pursuit(n, L, eff.dt, dev, q0, onset,
                                       speed=speed, speed_range=speed_range,
                                       turn_tau_ms=turn_tau, speed_tau_ms=self.speed_tau_ms,
                                       curviness=curviness, generator=None)
        return theta0, self._targets_from_norm(q_norm), n            # M: (n, L, 2)

    def make_batch(self, n=None, spec=None):
        """Random batch (spec None) or an explicit batch from a spec dict.

        spec keys (all optional):
            start / start_space  : start posture; default = random central sample
            speed / speed_range  : fixed or smooth time-varying target speed
            curviness / turn_tau_ms : path winding controls
            catch / no_go        : withhold the go cue (bool or length-n); hand holds at start
            go_pulse_steps       : go-cue pulse length (default: task's go_pulse_ms)
        """
        eff, dev = self.effector, self.effector.device
        L, P, T = self.exec_steps, self.preview_steps, self.steps
        theta0, M, n = self._gen_motion(n, spec)                     # M: (n, L, 2), M[:,0] = start
        start_xy = M[:, 0, :]                                        # (n, 2) start position

        if spec is None:
            catch = torch.rand(n, device=dev) < self.prob_catch
            pulse = self.go_pulse
        else:
            catch = _as_col(spec.get("catch", spec.get("no_go", False)), n, dev, torch.bool).squeeze(1)
            pulse = self.go_pulse
            if spec.get("go_pulse_steps") is not None:
                p = int(spec["go_pulse_steps"]); pulse = p if p > 0 else None

        # rolling preview: input shows M over [0, L); desired hand motion is M delayed by P.
        inp = torch.zeros(n, T, 4, device=dev)
        inp[:, :L, 0:2] = M                                          # target preview (leads by P)
        inp[:, :L, 2]   = 1.0                                        # 'on' while preview streams
        tg = torch.arange(T, device=dev).unsqueeze(0)
        if pulse is None:
            go = (tg >= P).float().expand(n, T).clone()             # sustained fallback
        else:
            go = ((tg >= P) & (tg < P + pulse)).float().expand(n, T).clone()
        go = go * (~catch).float().unsqueeze(1)                      # no go on catch trials
        inp[:, :, 3] = go
        if self.unified_input:
            inp = _pack_unified(inp)

        # desired: hold at start during the preview, then track M from its beginning; hold at start
        # for the whole trial on catch trials.
        desired = start_xy.unsqueeze(1).expand(n, T, 2).contiguous()
        desired[:, P:P + L] = M
        desired = torch.where(catch.view(n, 1, 1), start_xy.unsqueeze(1), desired)

        timestamps = {
            'move_start':  torch.full((n,), P, dtype=torch.long, device=dev),   # go onset
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'is_catch':    catch,
        }
        return theta0, inp, desired, None, timestamps


class PacManTask:
    """Isometric force-production task, after Amematsro et al. (the 'Pac-Man' task).

    The plant is held isometric (posture fixed at the start; the effector's `isometric` mode does
    this and records the actuator's Cartesian endpoint force in states.force). The network is shown
    a target force profile as a rolling preview and must reproduce it as generated force -- so the
    loss compares FORCE, not position (train.py switches to states.force for this task).

    Trial structure (rolling 1 s preview, like the paper's scrolling dot-path):
        - The target force profile P (a static / ramp / sine / chirp, see CONDITIONS) is shown in
          the [x, y, on, go] instruction stream from t=0, scaled by a per-trial direction. It leads
          the required output by `preview_ms` (default 1000 ms): input(t) shows P(t) while the
          desired output is P(t - preview).
        - Output should stay at zero during the preview. A go-cue pulse fires at the end of the
          preview; from then the network reproduces P from its beginning.
        - Catch trials (prob_catch, default 0.2): no go cue, and the desired output stays at zero
          for the whole trial even though the force preview still scrolls by -- the network must
          learn to withhold force until cued.

    Force is a 2-D Cartesian vector = scalar_profile x direction. By default the direction is fixed
    (force_angle_deg, paper-like 1-D forward force); random_dir=True randomizes it per trial. Trial
    variety otherwise comes from the random start posture. Profile magnitudes peak at peak_force_n.
    """
    name = "pacman"
    input_channels = 4                    # [force_x, force_y, on, go]
    supports_unified = True
    is_force_task = True                  # train.py compares states.force (not pos) to `desired`

    CONDITIONS = ['static_low', 'static_med', 'static_high',
                  'ramp_up', 'ramp_down', 'ramp_up_fast', 'ramp_down_fast',
                  'sine_025', 'sine_1', 'sine_2', 'sine_3', 'chirp']

    def __init__(self, effector, exec_ms=2000, preview_ms=1000,
                 peak_force_n=8.0, force_angle_deg=90.0, random_dir=False,
                 prob_catch=0.2, go_pulse_ms=150,
                 conditions=None, condition_probs=None,
                 chirp_f0=0.25, chirp_f1=3.0, center_frac=0.5,
                 unified_input=False, desired_profile='step', **kwargs):
        self.effector = effector
        effector.isometric = True                     # this task holds the plant isometric
        self.unified_input = bool(unified_input)
        self.center_frac = float(center_frac)
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.exec_steps = ms2steps(exec_ms)
        self.preview_steps = ms2steps(preview_ms)
        self.steps = self.preview_steps + self.exec_steps
        self.peak = float(peak_force_n)
        self.angle = math.radians(force_angle_deg)
        self.random_dir = bool(random_dir)
        self.prob_catch = float(prob_catch)
        self.go_pulse = (None if (go_pulse_ms is None or go_pulse_ms <= 0)
                         else max(1, round(go_pulse_ms / 1000 / effector.dt)))
        self.conditions = list(conditions) if conditions else list(self.CONDITIONS)
        self.cond_index = {c: i for i, c in enumerate(self.conditions)}
        if condition_probs is None:
            self.condition_probs = torch.ones(len(self.conditions)) / len(self.conditions)
        else:
            cp = torch.as_tensor(condition_probs, dtype=torch.float32)
            self.condition_probs = cp / cp.sum()
        self.chirp_f0, self.chirp_f1 = float(chirp_f0), float(chirp_f1)
        self.templates = self._build_templates()      # (n_cond, exec_steps) scalar profiles

    # -- force-profile templates -----------------------------------------------------------
    def _build_templates(self):
        L, dt, A = self.exec_steps, self.effector.dt, self.peak
        t = torch.arange(L, dtype=torch.float32) * dt            # seconds, 0..(L-1)dt
        D = (L - 1) * dt if L > 1 else dt                        # profile duration
        two_pi = 2 * math.pi
        def sine(f):                                             # 0..A, starts at 0
            return 0.5 * A * (1.0 - torch.cos(two_pi * f * t))
        def ramp(up):
            r = t / D
            return A * (r if up else (1.0 - r))
        def fast_ramp(up, frac=0.3):                             # reach peak/zero in frac*D, hold
            r = (t / (frac * D)).clamp(max=1.0)
            return A * (r if up else (1.0 - r))
        # chirp: instantaneous frequency sweeps chirp_f0 -> chirp_f1 (phase = 2pi * integral f dt)
        f_of_t = self.chirp_f0 + (self.chirp_f1 - self.chirp_f0) * (t / D)
        phase = two_pi * torch.cumsum(f_of_t, dim=0) * dt
        chirp = 0.5 * A * (1.0 - torch.cos(phase))
        table = {
            'static_low':  torch.full((L,), 0.25 * A), 'static_med': torch.full((L,), 0.5 * A),
            'static_high': torch.full((L,), 0.75 * A),
            'ramp_up':   ramp(True),        'ramp_down':      ramp(False),
            'ramp_up_fast': fast_ramp(True), 'ramp_down_fast': fast_ramp(False),
            'sine_025': sine(0.25), 'sine_1': sine(1.0), 'sine_2': sine(2.0), 'sine_3': sine(3.0),
            'chirp': chirp,
        }
        return torch.stack([table[c] for c in self.conditions], dim=0)   # (n_cond, L)

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional):
            start / start_space  : start posture; default = random central sample
            condition            : condition name(s) from CONDITIONS (scalar or length-n list)
            angle_deg            : force direction(s) in degrees (default: task's fixed angle)
            no_go / catch        : withhold the go cue (bool or length-n); desired output stays 0
            go_pulse_steps       : go-cue pulse length (default: task's go_pulse_ms)
        """
        eff, dev = self.effector, self.effector.device
        L, P, T = self.exec_steps, self.preview_steps, self.steps
        templates = self.templates.to(dev)

        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = eff.sample_center_joint(n, self.center_frac)
            cond = torch.multinomial(self.condition_probs.to(dev), n, replacement=True)
            ang = (torch.rand(n, device=dev) * (2 * math.pi) if self.random_dir
                   else torch.full((n,), self.angle, device=dev))
            catch = torch.rand(n, device=dev) < self.prob_catch
            pulse = self.go_pulse
        else:
            start = spec.get("start")
            if start is None:
                theta0 = eff.sample_center_joint(int(spec.get("n", 1)), self.center_frac)
            else:
                start = torch.as_tensor(np.asarray(start, dtype=np.float32), device=dev)
                if start.ndim == 1:
                    start = start.unsqueeze(0)
                theta0 = (eff.cartesian_to_joint(start)
                          if spec.get("start_space", "joint") == "cartesian" else start)
            n = theta0.shape[0]
            c = spec.get("condition", self.conditions[0])
            clist = [c] * n if isinstance(c, str) else list(c)
            cond = torch.tensor([self.cond_index[cc] for cc in clist], device=dev)
            ang = _as_col(spec.get("angle_deg", math.degrees(self.angle)), n, dev,
                          torch.float32).squeeze(1) * math.pi / 180.0
            catch_in = spec.get("catch", spec.get("no_go", False))
            catch = _as_col(catch_in, n, dev, torch.bool).squeeze(1)
            pulse = self.go_pulse
            if spec.get("go_pulse_steps") is not None:
                p = int(spec["go_pulse_steps"]); pulse = p if p > 0 else None

        scalar = templates[cond]                                 # (n, L)
        dir_xy = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)   # (n, 2)

        # rolling preview: input shows P over [0, L); desired output is P delayed by the preview
        prof_in = torch.zeros(n, T, device=dev); prof_in[:, :L] = scalar          # input stream
        des_sc  = torch.zeros(n, T, device=dev); des_sc[:, P:P + L] = scalar       # delayed output
        des_sc  = des_sc * (~catch).float().unsqueeze(1)                           # catch -> 0

        inp = torch.zeros(n, T, 4, device=dev)
        inp[:, :, 0:2] = prof_in.unsqueeze(-1) * dir_xy.unsqueeze(1)   # force preview vector
        inp[:, :L, 2]  = 1.0                                           # 'on' while preview streams
        tg = torch.arange(T, device=dev).unsqueeze(0)
        if pulse is None:
            go = (tg >= P).float().expand(n, T).clone()               # sustained fallback
        else:
            go = ((tg >= P) & (tg < P + pulse)).float().expand(n, T).clone()
        go = go * (~catch).float().unsqueeze(1)                        # no go on catch trials
        inp[:, :, 3] = go
        if self.unified_input:
            inp = _pack_unified(inp)

        desired = des_sc.unsqueeze(-1) * dir_xy.unsqueeze(1)          # (n, T, 2) target FORCE
        perturbation = None
        timestamps = {
            'move_start':  torch.full((n,), P, dtype=torch.long, device=dev),   # go onset
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'condition':   cond.long(),
            'is_catch':    catch,
        }
        return theta0, inp, desired, perturbation, timestamps


TASKS = {'delayed_reach': DelayedReaching, 'delayed_reach_posture': DelayedReachPosture,
         'horizon_sequence': HorizonSequence,
         'hold_posture_pulse': HoldPosturePulse, 'hold_posture_ramp': HoldPostureRamp,
         'pursuit': PursuitTask,
         'pacman': PacManTask}


def task_input_channels(name, unified_input=False):
    """Instruction-stream width the effector should be built with for a given task. Single-target
    tasks widen to UNIFIED_INPUT_CHANNELS when unified_input=True; the sequence task is always
    its native width."""
    cls = TASKS[name]
    if unified_input and getattr(cls, 'supports_unified', False):
        return UNIFIED_INPUT_CHANNELS
    return cls.input_channels


def make_task(name, effector, **kwargs):
    return TASKS[name](effector, **kwargs)