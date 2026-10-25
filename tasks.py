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

from effectors import CartesianForce


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


def _random_force_pulses(n, T, device, prob_per_step, max_force, dur_steps):
    """Random training disturbance at the HAND: brief force pulses (N) of random direction and
    magnitude uniform in [0, max_force], each lasting dur_steps. A pulse starts with probability
    prob_per_step on every step that is not already inside one, so pulses never overlap and a
    trial gets about T / (1/prob_per_step + dur_steps) of them, anywhere in the trial -- during
    holds as much as during movement. Never shown to the controller: it has to feel the push
    through its delayed feedback and hold or recover. Returns a CartesianForce ((n, T, 2)), which
    the rollout turns into joint torque at the arm's posture of the moment, or None when off.

    Built without a loop over time: the number of free steps before each pulse is geometric, so
    the start times are a cumulative sum of geometric gaps plus the pulses already spent."""
    if prob_per_step <= 0.0 or max_force <= 0.0:
        return None
    d = max(1, int(dur_steps))
    K = int(3 * T * prob_per_step) + 10                       # far more pulses than can ever fit
    u = torch.rand(n, K, device=device).clamp(min=1e-12)
    gap = torch.floor(torch.log(u) / math.log1p(-prob_per_step)).long()     # free steps before pulse k
    start = gap.cumsum(1) + d * torch.arange(K, device=device).unsqueeze(0)  # (n, K)
    ang = 2 * math.pi * torch.rand(n, K, device=device)
    mag = max_force * torch.rand(n, K, device=device)
    vec = torch.stack([ang.cos(), ang.sin()], dim=-1) * mag.unsqueeze(-1)   # (n, K, 2)
    tg = torch.arange(T, device=device).view(1, 1, T)
    on = ((tg >= start.unsqueeze(-1)) & (tg < (start + d).unsqueeze(-1))).float()   # (n, K, T)
    return CartesianForce(torch.einsum('nkt,nkc->ntc', on, vec))


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
                 capture_mode='scripted', capture_radius_cm=1.0, capture_hold_ms=300,
                 reach_timeout_ms=800, reach_timeout_range_ms=None, force_advance=True,
                 target_noise_cm=0.0, reach_len_range_cm=None,
                 reach_len_mix_frac=0.0, reach_len_long_min_cm=12.0,
                 perturb_prob=0.0, perturb_mag=0.0, perturb_dur_ms=100,
                 pulse_prob=0.0, pulse_force_n=0.0, pulse_dur_ms=100, **kwargs):
        self.effector = effector
        # How a target gets captured (and the next reach cued):
        #   'scripted'   : at a pre-sampled time (dwell_range_ms) regardless of where the hand
        #                  is. The whole instruction stream is built ahead of the rollout.
        #   'contingent' : only once the hand has stayed within capture_radius_cm of the target
        #                  for capture_hold_ms, checked against the true hand position every
        #                  step of the rollout (ContingentStepper); force-advanced after
        #                  reach_timeout_ms. Makes "sweep near the target without committing"
        #                  impossible instead of merely costly.
        assert capture_mode in ('scripted', 'contingent')
        self.capture_mode = capture_mode
        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.capture_radius = capture_radius_cm / 100.0
        self.capture_hold_steps = ms2steps(capture_hold_ms)
        self.reach_timeout_steps = ms2steps(reach_timeout_ms)
        # Force-advance deadline. A *range* samples it independently per (trial, reach), which
        # is what keeps the schedule unpredictable while the net is still too poor to capture
        # anything: with a single fixed deadline every reach lasts exactly reach_timeout_ms, so
        # reach onsets are pinned to the clock and the task loses the timing jitter the scripted
        # mode gets from its random per-reach dwell. Once capture is reliable the deadline stops
        # binding and the jitter disappears on its own.
        if reach_timeout_range_ms is None:
            self.reach_timeout_lo = self.reach_timeout_hi = self.reach_timeout_steps
        else:
            lo, hi = reach_timeout_range_ms
            self.reach_timeout_lo, self.reach_timeout_hi = ms2steps(lo), ms2steps(hi)
            assert self.reach_timeout_lo <= self.reach_timeout_hi, \
                "reach_timeout_range_ms must be (lo, hi)"
            self.reach_timeout_steps = self.reach_timeout_hi          # worst case for budgeting
        self.reach_timeout_range_ms = reach_timeout_range_ms
        # Whether missing the deadline still advances the sequence. True (default) = the old
        # behaviour: a reach the hand never dwelled on is force-advanced, so flying through a
        # target costs only (deadline - a normal reach). That is a loophole -- it lets the network
        # treat pairs of targets as one movement, sweeping through the first at full speed and
        # collecting it by timeout. An animal cannot do that: no dwell, no progression. False
        # keeps the target live until it is actually acquired, so the hand must come back, and a
        # trial that runs out of time simply ends unfinished.
        self.force_advance = bool(force_advance)
        # Sensory uncertainty about WHERE the targets are: fresh Gaussian noise (std
        # target_noise_cm, independent per step, per slot, per axis) on the coordinates of every
        # lit slot. The on flags and the go cue stay exact. Without it the network reads a
        # target's position perfectly the instant it appears, so it has no preparation cost and a
        # preview buys it almost nothing -- under honest acquisition it re-initiates ~100 ms after
        # every capture at every horizon, and reach 1 is no faster than mid-sequence. With noise, a
        # precise estimate takes integration time, so a target seen in advance (target 0 over the
        # delay, or a previewed next target) can be acted on sooner than one that just appeared.
        self.target_noise = float(target_noise_cm) / 100.0
        # Length of each reach in RANDOM (training) batches. None = every target drawn independently
        # over the whole workspace, which makes the median reach ~27 cm -- nearly six times the
        # 4.7 cm neighbour steps of the grid the models are tested on. That mismatch matters once
        # targets are noisy: what makes launching on a first look costly is noise relative to reach
        # length (2 cm of noise is a 4 deg aiming error on a 27 cm reach but 23 deg on a 4.7 cm
        # one). (lo, hi) in cm makes each trial a random walk: every target, the first included, lies
        # between lo and hi from the previous point. Explicit specs are untouched.
        self.reach_len_range = (None if reach_len_range_cm is None else
                                (float(reach_len_range_cm[0]) / 100.0,
                                 float(reach_len_range_cm[1]) / 100.0))
        # Mix of reach lengths within every batch. reach_len_mix_frac of the trials are LONG: their
        # targets are spread over the whole workspace with successive ones at least
        # reach_len_long_min_cm apart. The rest keep reach_len_range (the short, grid-like walk).
        # The two regimes fail in opposite ways when trained alone -- short reaches alone make
        # premature movement cheap (the first reach drifts off before its cue, holds are left
        # early), long reaches alone give discipline and speed but no reason to wait for a noisy
        # target -- so each batch carries both. Needs reach_len_range_cm for the short share.
        self.reach_len_mix_frac = float(reach_len_mix_frac)
        self.reach_len_long_min = float(reach_len_long_min_cm) / 100.0
        assert 0.0 <= self.reach_len_mix_frac <= 1.0
        assert self.reach_len_mix_frac == 0.0 or self.reach_len_range is not None, \
            "reach_len_mix_frac needs reach_len_range_cm (the range of the short trials)"
        # How to handle the window after an *unpreviewed* go cue, during which the target has not
        # reached the controller yet (sensory delay):
        #   'delay' : the desired holds at the previous target across the window, so waiting is
        #             optimal and pre-moving is penalised  (recommended)
        #   'mask'  : the window is dropped from the loss -- removes the cost of guessing but not
        #             its benefit, so the network may still pre-move
        #   'none'  : original behaviour, desired jumps at the capture
        assert blind_mode in ('delay', 'mask', 'none')
        self.blind_mode = blind_mode
        self.blind_steps = (effector.task_d if blind_ms is None
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
        # force pulses at the hand, several per trial (see _random_force_pulses). Random batches
        # only; an explicit spec carries its own perturbation or none. Takes over from
        # perturb_prob/perturb_mag (one joint-torque pulse on a fraction of trials) when set.
        self.pulse_prob = float(pulse_prob)
        self.pulse_force = float(pulse_force_n)
        self.pulse_dur = max(1, round(pulse_dur_ms / 1000 / effector.dt))
        if self.contingent:
            # Trial-level no-go IS supported: the stepper guards every capture check and every
            # go-cue write with ~nogo, so such a trial never goes live, never captures, and
            # _outputs (cap = 0) holds `desired` at the start posture for the whole episode.
            # Verified end-to-end: 0 go-cue steps, desired == start exactly, 0 captures, 0
            # timeout flags, finite gradients.
            # The *reach-level* skip is the one with no contingent definition: nothing can end a
            # skipped reach, because the hand is already on the previous target and no new target
            # is ever cued, so there is no capture event to advance the sequence. It would need a
            # duration rule of its own (e.g. reuse that reach's sampled deadline, recorded as a
            # skip rather than a timeout).
            assert self.prob_no_go_reach == 0.0, \
                "capture_mode='contingent' does not support reach-level no-go (prob_no_go_reach) " \
                "yet -- a skipped reach has no capture event to end it. Trial-level no-go " \
                "(prob_no_go) is supported."
            per_reach = self.reach_timeout_steps          # worst case: every reach times out
        else:
            per_reach = self.dwell_hi
        self.steps = (self.init_hi + self.delay_hi
                      + self.n_reaches * per_reach + self.final_hi)

    @property
    def contingent(self):
        return self.capture_mode == 'contingent'

    def _sample_walk_targets(self, theta0, n, R, n_cand=768):
        """(n, R, 2) targets forming a random walk whose every step length lies in that trial's
        range: reach_len_range, or [reach_len_long_min, anywhere] for the reach_len_mix_frac of
        trials that are long. Candidates are drawn exactly as independent targets are -- uniformly
        in joint space, so each is reachable by construction -- and one whose distance from the
        previous point is in range is kept. If a trial has no candidate in range (rare: a point
        hemmed into a corner), it takes the one that misses the range by least."""
        eff, dev = self.effector, self.effector.device
        lo = torch.full((n, 1), self.reach_len_range[0], device=dev)
        hi = torch.full((n, 1), self.reach_len_range[1], device=dev)
        n_long = int(round(n * self.reach_len_mix_frac))                # exact share, like a split batch
        if n_long > 0:
            lo[:n_long] = self.reach_len_long_min
            hi[:n_long] = float('inf')
        prev = eff.joint_to_cart(theta0)                                # (n, 2)
        rows = torch.arange(n, device=dev)
        out = []
        for _ in range(R):
            cand = eff.joint_to_cart(eff.sample_joint(n * n_cand)).reshape(n, n_cand, 2)
            dist = (cand - prev.unsqueeze(1)).norm(dim=-1)              # (n, n_cand)
            ok = (dist >= lo) & (dist <= hi)
            miss = torch.where(dist < lo, lo - dist, dist - hi).clamp(min=0)
            # in-range candidates get a random score in [1, 2); the rest are ranked by how little
            # they miss, always below any in-range one
            score = torch.where(ok, 1.0 + torch.rand(n, n_cand, device=dev), -miss)
            prev = cand[rows, score.argmax(1)]
            out.append(prev)
        return torch.stack(out, 1)

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
            if self.reach_len_range is None:
                targets = eff.joint_to_cart(eff.sample_joint(n * R)).reshape(n, R, 2)
            else:
                targets = self._sample_walk_targets(theta0, n, R)
            T = self.steps
            d1   = torch.randint(self.init_lo,  self.init_hi + 1,  (n, 1), device=dev)
            d2   = torch.randint(self.delay_lo, self.delay_hi + 1, (n, 1), device=dev)
            dseg = torch.randint(self.dwell_lo, self.dwell_hi + 1, (n, R), device=dev)
            d4   = torch.randint(self.final_lo, self.final_hi + 1, (n, 1), device=dev)
            if self.contingent:
                # reach durations aren't known ahead of time, so the trial can't be
                # right-aligned: slack goes to the end (final hold at the last target)
                t_delay = d1
            else:
                pre  = T - (d1 + d2 + dseg.sum(1, keepdim=True) + d4)   # slack -> initial hold
                t_delay = pre + d1
            t_go    = t_delay + d2
            horizon = torch.multinomial(self.horizon_probs.to(dev), n, replacement=True) + 1
            nogo    = torch.rand(n, device=dev) < self.prob_no_go                     # (n,)
            hit_r   = (torch.rand(n, device=dev) < self.prob_no_go_reach) & ~nogo & (R > 1)
            ridx    = torch.randint(1, max(2, R), (n,), device=dev)
            no_go_reach = torch.where(hit_r, ridx, torch.full_like(ridx, -1))         # (n,)
            pulse = self.go_pulse
            if self.pulse_prob > 0.0 and self.pulse_force > 0.0:
                perturbation = _random_force_pulses(n, T, dev, self.pulse_prob, self.pulse_force,
                                                    self.pulse_dur)
            else:
                perturbation = _random_perturbation(eff, n, T, dev, self.perturb_prob,
                                                    self.perturb_mag, self.perturb_dur)
        else:
            theta0, targets, n = self._resolve_spec_geometry(spec, dev)
            d1 = _as_col(spec.get("init_steps",  mid(self.init_lo,  self.init_hi)),  n, dev)
            d2 = _as_col(spec.get("delay_steps", mid(self.delay_lo, self.delay_hi)), n, dev)
            dw = np.broadcast_to(np.asarray(spec.get("dwell_steps",
                                                     mid(self.dwell_lo, self.dwell_hi))), (n, R))
            dseg = torch.as_tensor(np.array(dw), device=dev).long()                  # (n, R)
            d4 = _as_col(spec.get("final_steps", mid(self.final_lo, self.final_hi)), n, dev)
            if self.contingent:
                T = int((d1 + d2 + d4).max().item()) + R * self.reach_timeout_steps
            else:
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
        if self.contingent:
            assert not (no_go_reach >= 0).any(), \
                "capture_mode='contingent' does not support no_go_reach yet"
            if self.reach_timeout_lo == self.reach_timeout_hi:
                deadline = torch.full((n, R), self.reach_timeout_hi, dtype=torch.long, device=dev)
            elif spec is None:
                deadline = torch.randint(self.reach_timeout_lo, self.reach_timeout_hi + 1,
                                         (n, R), device=dev)
            else:                                    # specs stay deterministic: use the midpoint
                deadline = torch.full((n, R), (self.reach_timeout_lo + self.reach_timeout_hi) // 2,
                                      dtype=torch.long, device=dev)
            stepper = ContingentStepper(self, theta0, start, targets, t_delay, t_go, horizon,
                                        nogo, pulse, T, deadline)
            return theta0, stepper.inp, None, perturbation, stepper

        bounds = t_go + dseg.cumsum(1)                                  # (n, R) capture times
        pulse_times = torch.cat([t_go, bounds[:, :-1]], dim=1)          # (n, R) go for reach k
        cap, skip, skip_col, has_skip = self._nogo_masks(nogo, no_go_reach, R)

        tg = torch.arange(T, device=dev).unsqueeze(0)                   # (1, T)
        cur = (tg.unsqueeze(-1) >= bounds.unsqueeze(1)).sum(-1)         # (n, T) captures so far
        cur_eff = torch.minimum(cur, cap.unsqueeze(1))                  # frozen on no-go variants

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
        inp = self._instruction(cur_eff, tg >= t_delay, go, targets, horizon)

        desired, timestamps = self._outputs(start, targets, t_delay, t_go, bounds, pulse_times,
                                            horizon, nogo, no_go_reach, T)
        return theta0, inp, desired, perturbation, timestamps

    # -- shared by the scripted path above and ContingentStepper -------------------------
    @staticmethod
    def _nogo_masks(nogo, no_go_reach, R):
        """Two independent no-go variants:
          trial no-go (cap = 0): no pulse ever fires and the hand holds at the start posture for
              the whole episode; the target display freezes at the beginning of the sequence.
          reach no-go (skip = r): reach r alone receives no pulse, so the hand simply stays where
              it is for that segment -- reach r-1's dwell is effectively extended -- and the
              sequence then RESUMES normally at reach r+1 (target r is skipped, never reached).
        """
        cap = torch.full_like(nogo, R, dtype=torch.long)
        cap = torch.where(nogo, torch.zeros_like(cap), cap)
        skip = no_go_reach.clone()                                      # (n,) -1 = no skipped reach
        skip = torch.where(nogo, torch.full_like(skip, -1), skip)       # variants are exclusive
        skip_col = skip.unsqueeze(1)                                    # (n, 1)
        has_skip = (skip >= 0).unsqueeze(1)
        return cap, skip, skip_col, has_skip

    def _instruction(self, cur_eff, shown, go, targets, horizon):
        """Instruction stream rows, 3 slots x [x, y, on] + go. cur_eff / shown / go are (n, K)
        -- K = T for a whole scripted trial, K = 1 for one step of a contingent one -- giving
        (n, K, input_channels). Slot j shows the (cur_eff + j)-th target if j < horizon, that
        target exists, and the targets are shown at all."""
        n, K = cur_eff.shape
        R = targets.shape[1]
        tx, ty = targets[..., 0], targets[..., 1]                       # (n, R)
        inp = torch.zeros(n, K, self.input_channels, device=targets.device)
        for j in range(self.n_slots):
            idx = cur_eff + j                                           # target index in slot j
            on = (idx < R) & (j < horizon.view(n, 1)) & shown
            idxc = idx.clamp(max=R - 1)
            x, y = tx.gather(1, idxc), ty.gather(1, idxc)
            if self.target_noise > 0:
                x = x + self.target_noise * torch.randn_like(x)
                y = y + self.target_noise * torch.randn_like(y)
            inp[:, :, 3 * j]     = x * on.float()
            inp[:, :, 3 * j + 1] = y * on.float()
            inp[:, :, 3 * j + 2] = on.float()
        inp[:, :, 3 * self.n_slots] = go.float()
        return inp

    def _outputs(self, start, targets, t_delay, t_go, bounds, pulse_times, horizon, nogo,
                 no_go_reach, T):
        """`desired` and the timestamps dict from per-reach capture times `bounds` (n, R) and
        go-pulse times `pulse_times` (n, R). One code path whether those are scripted
        (make_batch) or realized (ContingentStepper.finalize), so the blind_mode and no-go
        semantics cannot drift apart between the two."""
        n, R = targets.shape[:2]
        dev = targets.device
        tx, ty = targets[..., 0], targets[..., 1]                       # (n, R)
        cap, skip, skip_col, has_skip = self._nogo_masks(nogo, no_go_reach, R)
        tg = torch.arange(T, device=dev).unsqueeze(0)                   # (1, T)
        final_start = bounds[:, -1]                                     # (n,)

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
        return desired, timestamps


class ContingentStepper:
    """Closed-loop instruction stream for HorizonSequence(capture_mode='contingent').

    make_batch returns one of these in place of a finished (inp, desired, timestamps): the
    capture times aren't known until the hand has actually gone somewhere. Effector.rollout
    calls `step(s, pos)` at the top of every timestep with the hand's true position at the end
    of the previous one; it advances the capture bookkeeping and writes that step's instruction
    row into `inp` in place (rollout then reads it through the usual visual delay). After the
    rollout, `finalize()` turns the realized capture times into `desired` + timestamps through
    the same HorizonSequence._outputs the scripted mode uses. Call `reset()` before reusing
    the same batch for another rollout (rollout_batch does).

    Capture rule, per trial: reach k is live from its go-pulse (t_go for k = 0, the previous
    capture otherwise). It is captured at the first step at which the hand has been within
    `capture_radius` of target k for `capture_hold_steps` consecutive steps -- or, failing
    that, `reach_timeout_steps` after its pulse (force-advance, recorded in timestamps
    'timeouts'). Capture fires the next reach's pulse and shifts the slots, exactly as a
    scripted capture does.
    """

    def __init__(self, task, theta0, start, targets, t_delay, t_go, horizon, nogo, pulse, T,
                 deadline=None):
        self.task = task
        self.theta0, self.start, self.targets = theta0, start, targets      # targets (n, R, 2)
        self.t_delay, self.t_go = t_delay, t_go                             # (n, 1)
        self.horizon, self.nogo, self.pulse, self.T = horizon, nogo, pulse, T
        self.n, self.R = targets.shape[:2]
        # per-(trial, reach) force-advance deadline in steps; constant unless the task was given
        # a reach_timeout_range_ms
        self.deadline = (torch.full((self.n, self.R), task.reach_timeout_steps,
                                    dtype=torch.long, device=targets.device)
                         if deadline is None else deadline)
        self.inp = torch.zeros(self.n, T, task.input_channels, device=targets.device)
        self.reset()

    def reset(self):
        n, R, dev = self.n, self.R, self.targets.device
        self.inp.zero_()
        self.cur = torch.zeros(n, dtype=torch.long, device=dev)         # captures so far
        self.in_tol = torch.zeros(n, dtype=torch.long, device=dev)      # consecutive steps in radius
        self.last_pulse = self.t_go.view(-1).clone()                    # go time of the live reach
        self.bounds = torch.full((n, R), self.T, dtype=torch.long, device=dev)  # T = never captured
        self.timeouts = torch.zeros(n, R, dtype=torch.bool, device=dev)

    @torch.no_grad()
    def step(self, s, pos):
        task, n, R = self.task, self.n, self.R
        rows = torch.arange(n, device=pos.device)
        t_delay, t_go = self.t_delay.view(-1), self.t_go.view(-1)

        # -- capture check against the hand's true position at the end of step s-1 --------
        live = (s >= t_go) & (self.cur < R) & ~self.nogo
        tgt = self.targets[rows, self.cur.clamp(max=R - 1)]             # (n, 2)
        inside = ((pos.detach() - tgt) ** 2).sum(-1) < task.capture_radius ** 2
        self.in_tol = torch.where(inside & live, self.in_tol + 1, torch.zeros_like(self.in_tol))
        held = self.in_tol >= task.capture_hold_steps
        timed_out = (s - self.last_pulse) >= self.deadline[rows, self.cur.clamp(max=R - 1)]
        captured = live & (held | (timed_out & self.task.force_advance))
        if bool(captured.any()):
            k = self.cur[captured]
            self.bounds[rows[captured], k] = s
            self.timeouts[rows[captured], k] = (timed_out & ~held)[captured]
            self.cur = self.cur + captured.long()
            self.last_pulse = torch.where(captured, torch.full_like(self.last_pulse, s),
                                          self.last_pulse)
            self.in_tol = torch.where(captured, torch.zeros_like(self.in_tol), self.in_tol)

        # -- instruction row for step s (after any capture, like the scripted `cur`) -------
        live = (s >= t_go) & (self.cur < R) & ~self.nogo
        cur_eff = torch.where(self.nogo, torch.zeros_like(self.cur), self.cur)
        go = live if self.pulse is None else live & ((s - self.last_pulse) < self.pulse)
        self.inp[:, s, :] = task._instruction(cur_eff.unsqueeze(1), (s >= t_delay).unsqueeze(1),
                                              go.unsqueeze(1), self.targets, self.horizon)[:, 0]

    def finalize(self):
        """(desired, timestamps) from the capture times realized during the rollout."""
        pulse_times = torch.cat([self.t_go, self.bounds[:, :-1]], dim=1)     # (n, R)
        no_go_reach = torch.full((self.n,), -1, dtype=torch.long, device=self.targets.device)
        desired, timestamps = self.task._outputs(self.start, self.targets, self.t_delay,
                                                 self.t_go, self.bounds, pulse_times,
                                                 self.horizon, self.nogo, no_go_reach, self.T)
        timestamps['timeouts'] = self.timeouts       # (n, R) reach was force-advanced, not captured
        return desired, timestamps


def reach_time_weights(ts, T, dt, grace_ms=50.0, ramp_start_ms=100.0, ramp_per_step=0.1,
                       cap=20.0, hold_weight=1.0):
    """(n, T) weight on the position error as a function of how long the current target has been
    the current target. From HorizonSequence timestamps `ts` (move_start (n,), capture_times
    (n, R) with T = never captured, optional is_no_go (n,)).

    Reach k is current from its advance (the go cue for k = 0, the capture of target k-1
    otherwise) until target k is captured. With tau = steps since that advance:
        tau <  grace       -> 0      nobody reacts instantly
        tau <  ramp_start  -> 1
        otherwise          -> 1 + ramp_per_step * (tau - ramp_start), at most `cap`
    so being late costs more the later it gets, and the clock restarts at every capture. Where
    the hand is simply meant to stay put -- before the first go cue, on a no-go trial, after the
    last capture -- the weight is the constant `hold_weight`."""
    dev = ts['capture_times'].device
    ms = ts['move_start'].to(dev)                                       # (n,)
    capt = ts['capture_times']                                          # (n, R)
    adv = torch.cat([ms.unsqueeze(1), capt[:, :-1]], dim=1)             # (n, R) reach k begins
    tt = torch.arange(T, device=dev).unsqueeze(0)                       # (1, T)
    k = (tt.unsqueeze(-1) >= adv.unsqueeze(1)).sum(-1) - 1              # (n, T) current reach; -1 = none yet
    tau = tt - adv.gather(1, k.clamp(min=0))                            # steps since it began
    g = round(grace_ms / 1000.0 / dt)
    r0 = round(ramp_start_ms / 1000.0 / dt)
    w = (1.0 + ramp_per_step * (tau - r0).clamp(min=0).float()).clamp(max=cap)
    w = torch.where(tau < g, torch.zeros_like(w), w)
    hold = (k < 0) | (tt >= capt[:, -1:])                               # before go / after the last capture
    if 'is_no_go' in ts:
        hold = hold | ts['is_no_go'].to(dev).unsqueeze(1)
    return torch.where(hold, torch.full_like(w, float(hold_weight)), w)


def rollout_batch(effector, controller, batch, **rollout_kwargs):
    """Roll out a make_batch() result on `controller`, hiding the scripted/contingent split.

    Returns (states, desired, timestamps). For a contingent HorizonSequence batch the
    instruction stream is generated inside the rollout and `desired` / timestamps only exist
    afterwards, so callers must use the returned values rather than the batch's own (which are
    None / the stepper). The batch can be rolled out repeatedly (e.g. a fixed eval set).
    """
    theta0, inp, desired, pert, ts = batch
    if isinstance(ts, ContingentStepper):
        ts.reset()
        states = effector.rollout(controller, theta0, inp, pert, task_step=ts.step, **rollout_kwargs)
        desired, ts = ts.finalize()
    else:
        states = effector.rollout(controller, theta0, inp, pert, **rollout_kwargs)
    return states, desired, ts


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


# ----------------------------------------------------------------------------- NHP multitask
# The three tasks of the NHP multitask sessions (KINARM; blocks of center-out reaches, torque
# pulses and torque ramps, all from one home posture). Numbers below were measured from the
# recordings (monkey Pert, session 111524):
#   home posture   shoulder 32 deg, elbow 72 deg (KINARM L2Ang is the absolute forearm angle, 104)
#   center-out     8 targets 5 cm from home; the target appears after a 500 ms home hold, the go
#                  cue 270-780 ms later; the hand then stays on the target (~2.5 s)
#   pulse          rectangular load for 300 ms after a 1.2-4 s hold
#   ramp           trapezoid: ~600 ms linear rise, ~3.7 s plateau, ~600 ms fall. The trial's
#                  alignment event ('Pert Onset') is the END of the rise.
#   loads          8 directions 45 deg apart in joint-torque space, all of the same torque norm
#                  (ramp plateau ~2/3 of the pulse), plus a null condition with no load. At the
#                  hand they are 0.6-2.4 N (pulse) and 0.4-1.6 N (ramp), depending on direction.
#                  The default test torques (0.4 and 0.27 N.m) give this arm the same range of
#                  hand forces (0.9-2.4 N and 0.6-1.6 N).
# All three share one instruction stream (the 10-wide unified layout: 3 slots x [x, y, on] + go)
# and carry NO task cue: slot 0 shows the home location, so the tasks are indistinguishable to the
# controller until the reach target appears (in slot 1) or a load is felt.
NHP_HOME_DEG = (32.0, 72.0)                         # shoulder, elbow (relative) at home
NHP_REACH_DIRS = (0, 45, 90, 135, 180, 225, 270, 315)
NHP_REACH_CM = 5.0
# Condition label in the recordings (direction of the commanded hand force, deg) -> direction of
# that load in joint-torque space (deg; 0 = +shoulder, 90 = +elbow). Found by matching
# F = J^-T tau at the home posture to the recorded force directions: the labels 30/210 are the
# two-joint loads of equal sign, 90/270 those of opposite sign, the rest are single-joint.
NHP_TORQUE_DIRS = {30: 225, 70: 270, 90: 315, 110: 0, 210: 45, 260: 90, 270: 135, 285: 180}
NHP_NULL = -1                                       # label of the no-load condition


class _NHPBase:
    """Shared by the three NHP tasks: the home posture and the instruction-stream layout."""
    supports_unified = True
    input_channels = 4                    # [x, y, on, go]; 10 when unified_input=True

    def __init__(self, effector, home_deg=NHP_HOME_DEG, home_jitter_deg=8.0,
                 unified_input=False, **kwargs):
        self.effector = effector
        self.unified_input = bool(unified_input)
        self.home = tuple(math.radians(a) for a in home_deg)
        # random batches start from the home posture +- home_jitter_deg on each joint, so the
        # controller does not learn one posture by heart; explicit specs use the exact home
        self.home_jitter = math.radians(home_jitter_deg)

    def _ms(self, ms):
        return max(1, round(ms / 1000 / self.effector.dt))

    def _sample_home(self, n):
        dev = self.effector.device
        home = torch.tensor(self.home, device=dev).unsqueeze(0)
        return home + (2 * torch.rand(n, 2, device=dev) - 1) * self.home_jitter

    def _spec_home(self, spec, n):
        """Start posture of an explicit batch: spec 'start' (see start_space) or the home."""
        eff, dev = self.effector, self.effector.device
        if spec.get("start") is None:
            return torch.tensor(self.home, device=dev).unsqueeze(0).expand(n, -1).contiguous()
        start = torch.as_tensor(np.asarray(spec["start"], dtype=np.float32), device=dev)
        if start.ndim == 1:
            start = start.unsqueeze(0)
        if spec.get("start_space", "joint") == "cartesian":
            start = eff.cartesian_to_joint(start)
        return start.expand(n, -1).contiguous() if start.shape[0] == 1 else start

    @staticmethod
    def _spec_n(spec, keys):
        """Trials in an explicit batch: the longest per-trial entry among `keys`."""
        lead = lambda k: (np.asarray(spec[k]).shape[0]
                          if (spec.get(k) is not None and np.asarray(spec[k]).ndim >= 1) else 1)
        n = max(lead(k) for k in keys)
        if spec.get("start") is not None and np.asarray(spec["start"]).ndim == 2:
            n = max(n, np.asarray(spec["start"]).shape[0])
        return n

    def _pack(self, inp):
        return _pack_unified(inp) if self.unified_input else inp


class NHPCenterOut(_NHPBase):
    """Delayed center-out reach from the home posture.

        1. home hold : slot 0 shows the home location, go = 0.       duration ~ init_range_ms
        2. delay     : the reach target appears in slot 1 while slot 0 keeps showing home;
                       go = 0; keep holding at home.                 duration ~ delay_range_ms
        3. reach     : go cue = home (slot 0) turns off, with a go pulse (go_pulse_ms); reach to
                       the target, which stays in slot 1, and hold there to the end of the
                       episode (at least post_ms after the go cue).

    This is the display the monkeys saw (home lit until the go cue) and needs the unified
    10-wide stream. target_slot=0 is the earlier layout, kept for models trained with it: the
    target replaces home in slot 0 when it appears and nothing but the go pulse marks the cue.
    With that layout "keep the hand on slot 0" (the whole of the pulse and ramp tasks) moves the
    hand at target onset, and every model crept part-way to the target before the cue.

    Random batches draw the reach direction uniformly and its length from reach_range_cm; a
    prob_no_go fraction of trials never get the go cue and must hold at home (the monkeys had no
    such trials; they keep the controller from moving on the target alone -- with 0.1 it still
    reached as soon as the target appeared). condition_spec()
    gives the eight recorded conditions. The desired trajectory jumps to the target at the go cue
    ('step') or follows a minimum-jerk line over mj_move_ms ('min_jerk')."""
    name = "nhp_center_out"

    def __init__(self, effector, init_range_ms=(300, 700), delay_range_ms=(300, 800),
                 post_ms=1500, reach_range_cm=(2.0, 8.0), prob_no_go=0.3, go_pulse_ms=150,
                 desired_profile='step', mj_move_ms=400, target_slot=1, **kwargs):
        super().__init__(effector, **kwargs)
        assert target_slot in (0, 1)
        assert target_slot == 0 or self.unified_input, "target_slot=1 needs unified_input=True"
        self.target_slot = target_slot
        self.init_lo, self.init_hi = self._ms(init_range_ms[0]), self._ms(init_range_ms[1])
        self.delay_lo, self.delay_hi = self._ms(delay_range_ms[0]), self._ms(delay_range_ms[1])
        self.post = self._ms(post_ms)
        self.reach_lo, self.reach_hi = reach_range_cm[0] / 100.0, reach_range_cm[1] / 100.0
        self.prob_no_go = float(prob_no_go)
        self.go_pulse = None if (go_pulse_ms is None or go_pulse_ms <= 0) else self._ms(go_pulse_ms)
        assert desired_profile in ('step', 'min_jerk')
        self.desired_profile = desired_profile
        self.mj_move_steps = self._ms(mj_move_ms)
        self.steps = self.init_hi + self.delay_hi + self.post

    def condition_spec(self):
        """(spec, labels): the eight recorded reaches (5 cm, 45 deg apart) and their direction
        labels in deg."""
        return ({"angle_deg": list(NHP_REACH_DIRS), "reach_cm": NHP_REACH_CM},
                np.array(NHP_REACH_DIRS))

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional; scalars broadcast, or give length-n lists):
            start / start_space      : start posture (default: the home posture)
            angle_deg                : reach direction(s) (default: the 8 recorded directions)
            reach_cm                 : reach length (default: 5)
            init_steps / delay_steps : segment durations in steps (default: range midpoints)
            no_go        (False)     : hold-at-home trials
            go_pulse_steps           : go-cue pulse length (0 or negative -> sustained cue)
            steps                    : episode length (default: go cue + post_ms)
        """
        eff, dev = self.effector, self.effector.device
        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = self._sample_home(n)
            ang = torch.rand(n, device=dev) * (2 * math.pi)
            dist = torch.rand(n, device=dev) * (self.reach_hi - self.reach_lo) + self.reach_lo
            t1 = torch.randint(self.init_lo, self.init_hi + 1, (n, 1), device=dev)
            t2 = t1 + torch.randint(self.delay_lo, self.delay_hi + 1, (n, 1), device=dev)
            nogo = torch.rand(n, 1, device=dev) < self.prob_no_go
            T, pulse = self.steps, self.go_pulse
        else:
            spec = dict(spec)
            spec.setdefault("angle_deg", list(NHP_REACH_DIRS))
            n = self._spec_n(spec, ("angle_deg", "reach_cm", "init_steps", "delay_steps", "no_go"))
            theta0 = self._spec_home(spec, n)
            ang = _as_col(spec["angle_deg"], n, dev, torch.float32).squeeze(1) * math.pi / 180.0
            dist = _as_col(spec.get("reach_cm", NHP_REACH_CM), n, dev, torch.float32).squeeze(1) / 100.0
            t1 = _as_col(spec.get("init_steps", (self.init_lo + self.init_hi) // 2), n, dev)
            t2 = t1 + _as_col(spec.get("delay_steps", (self.delay_lo + self.delay_hi) // 2), n, dev)
            nogo = _as_col(spec.get("no_go", False), n, dev, torch.bool)
            T = int(spec.get("steps", int(t2.max().item()) + self.post))
            pulse = self.go_pulse
            if spec.get("go_pulse_steps") is not None:
                p = int(spec["go_pulse_steps"])
                pulse = p if p > 0 else None

        home = eff.joint_to_cart(theta0)                                   # (n, 2)
        target = home + dist.unsqueeze(1) * torch.stack([ang.cos(), ang.sin()], dim=1)
        tg = torch.arange(T, device=dev).unsqueeze(0)                      # (1, T)
        shown = (tg >= t1).unsqueeze(-1)                                   # reach target on display
        after_go = (tg >= t2) & ~nogo
        go = after_go if pulse is None else (after_go & (tg < t2 + pulse))

        if self.target_slot == 0:
            inp = torch.zeros(n, T, 4, device=dev)
            inp[:, :, 0:2] = torch.where(shown, target.unsqueeze(1), home.unsqueeze(1))
            inp[:, :, 2] = 1.0
            inp[:, :, 3] = go.float()
            inp = self._pack(inp)
        else:
            home_on = (~after_go).float().unsqueeze(-1)                   # home lit until the go cue
            inp = torch.zeros(n, T, UNIFIED_INPUT_CHANNELS, device=dev)
            inp[:, :, 0:2] = home.unsqueeze(1) * home_on
            inp[:, :, 2:3] = home_on
            inp[:, :, 3:5] = target.unsqueeze(1) * shown.float()
            inp[:, :, 5:6] = shown.float()
            inp[:, :, -1] = go.float()

        if self.desired_profile == 'step':
            desired = torch.where(after_go.unsqueeze(-1), target.unsqueeze(1), home.unsqueeze(1))
        else:
            s = _min_jerk_s((tg - t2).float() / self.mj_move_steps) * (~nogo).float()
            desired = home.unsqueeze(1) + (target - home).unsqueeze(1) * s.unsqueeze(-1)

        timestamps = {
            'delay_start': t1.squeeze(-1).long(),                          # reach target appears
            'move_start':  t2.squeeze(-1).long(),                          # go cue
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'is_no_go':    nogo.squeeze(-1),
            'hold_mask':   ~after_go,          # (n, T) steps the hand should be still at home
            'angle_deg':   ang * 180.0 / math.pi,
            'reach_cm':    dist * 100.0,
        }
        return theta0, inp, desired, None, timestamps


class _NHPLoad(_NHPBase):
    """Hold the home posture against an external load defined in joint-torque space (shoulder,
    elbow; N.m), as the KINARM applies it. The load is never shown in the instruction stream:
    slot 0 shows the home location for the whole episode and there is no go cue, so the desired
    position is the home throughout.

    Random batches draw the direction of the load uniformly in joint-torque space and its norm
    from torque_range_nm; a catch_prob fraction of trials carry no load (the recorded null
    condition is 1 of 9). condition_spec() gives the nine recorded conditions at test_torque_nm.
    Subclasses shape the load over time."""

    def __init__(self, effector, onset_range_ms, torque_range_nm, test_torque_nm, settle_ms,
                 catch_prob=1 / 9, **kwargs):
        super().__init__(effector, **kwargs)
        self.onset_lo, self.onset_hi = self._ms(onset_range_ms[0]), self._ms(onset_range_ms[1])
        self.torque_lo, self.torque_hi = float(torque_range_nm[0]), float(torque_range_nm[1])
        self.test_torque = float(test_torque_nm)
        self.settle = self._ms(settle_ms)
        self.catch_prob = float(catch_prob)
        self.steps = self.onset_hi + self._max_load_steps() + self.settle

    # -- subclass hooks --------------------------------------------------------------------
    _shape_keys = ()                      # spec keys of the load's timing, in steps

    def _max_load_steps(self):
        raise NotImplementedError

    def _sample_shape(self, n, dev):
        """dict of (n, 1) step counts describing the load's time course."""
        raise NotImplementedError

    def _spec_shape(self, spec, n, dev):
        raise NotImplementedError

    def _envelope(self, tg, start, shape):
        """(n, T) load as a fraction of its norm, plus the (n, 1) step it ends and the (n, 1)
        step trials are aligned on."""
        raise NotImplementedError

    # -- shared ----------------------------------------------------------------------------
    def condition_spec(self):
        """(spec, labels): the eight recorded load directions and the null condition, at
        test_torque_nm. Labels are the recorded direction labels (deg), -1 for the null."""
        labels = sorted(NHP_TORQUE_DIRS) + [NHP_NULL]
        spec = {"torque_angle_deg": [NHP_TORQUE_DIRS.get(l, 0) for l in labels],
                "torque_nm": [0.0 if l == NHP_NULL else self.test_torque for l in labels]}
        return spec, np.array(labels)

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional; scalars broadcast, or give length-n lists):
            start / start_space      : start posture (default: the home posture)
            torque_angle_deg         : load direction(s) in joint-torque space (0 = +shoulder,
                                       90 = +elbow; default: 8 directions 45 deg apart)
            torque_nm                : load norm(s) in N.m (default: test_torque_nm)
            onset_steps              : step the load starts (default: onset range midpoint)
            + the load's timing in steps (dur_steps for the pulse; rise_steps / plateau_steps
              for the ramp)
            steps                    : episode length (default: load end + settle_ms)
        """
        eff, dev = self.effector, self.effector.device
        if spec is None:
            if n is None:
                raise ValueError("make_batch needs either n (random) or spec (explicit)")
            theta0 = self._sample_home(n)
            start = torch.randint(self.onset_lo, self.onset_hi + 1, (n, 1), device=dev)
            ang = torch.rand(n, device=dev) * (2 * math.pi)
            amp = torch.rand(n, device=dev) * (self.torque_hi - self.torque_lo) + self.torque_lo
            amp = amp * (torch.rand(n, device=dev) >= self.catch_prob).float()
            shape = self._sample_shape(n, dev)
            T = self.steps
        else:
            spec = dict(spec)
            spec.setdefault("torque_angle_deg", list(range(0, 360, 45)))
            n = self._spec_n(spec, ("torque_angle_deg", "torque_nm", "onset_steps") + self._shape_keys)
            theta0 = self._spec_home(spec, n)
            start = _as_col(spec.get("onset_steps", (self.onset_lo + self.onset_hi) // 2), n, dev)
            ang = _as_col(spec["torque_angle_deg"], n, dev, torch.float32).squeeze(1) * math.pi / 180.0
            amp = _as_col(spec.get("torque_nm", self.test_torque), n, dev, torch.float32).squeeze(1)
            shape = self._spec_shape(spec, n, dev)
            T = None

        if T is None:                                    # explicit batch: load end + settle
            _, end, _ = self._envelope(torch.zeros(1, 1, dtype=torch.long, device=dev), start, shape)
            T = int(spec.get("steps", int(end.max().item()) + self.settle))
        tg = torch.arange(T, device=dev).unsqueeze(0)                      # (1, T)
        env, end, onset = self._envelope(tg, start, shape)                 # (n, T), (n, 1), (n, 1)
        tau = amp.unsqueeze(1) * torch.stack([ang.cos(), ang.sin()], dim=1)   # (n, 2) N.m
        perturbation = env.unsqueeze(-1) * tau.unsqueeze(1)                # (n, T, 2)

        home = eff.joint_to_cart(theta0)                                   # (n, 2)
        inp = torch.zeros(n, T, 4, device=dev)
        inp[:, :, 0:2] = home.unsqueeze(1)
        inp[:, :, 2] = 1.0
        desired = home.unsqueeze(1).expand(n, T, 2).contiguous()

        timestamps = {
            'load_start':  start.squeeze(-1).long(),                       # first step with load
            'load_end':    end.squeeze(-1).long(),                         # first step without
            'onset':       onset.squeeze(-1).long(),                       # alignment event
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'torque_nm':   amp,
            'torque_angle_deg': ang * 180.0 / math.pi,
            'is_catch':    amp == 0,
            'hold_mask':   tg < start,         # (n, T) steps before the load: still at home
        }
        return theta0, self._pack(inp), desired, perturbation, timestamps


class NHPPulse(_NHPLoad):
    """Hold the home posture against a rectangular torque pulse (300 ms in the recordings).
    Trials are aligned on the start of the pulse."""
    name = "nhp_pulse"
    _shape_keys = ("dur_steps",)

    def __init__(self, effector, onset_range_ms=(500, 1500), dur_range_ms=(200, 400),
                 torque_range_nm=(0.05, 0.8), test_torque_nm=0.4, settle_ms=1500, **kwargs):
        self.dur_lo, self.dur_hi = (max(1, round(d / 1000 / effector.dt)) for d in dur_range_ms)
        super().__init__(effector, onset_range_ms, torque_range_nm, test_torque_nm, settle_ms,
                         **kwargs)

    def _max_load_steps(self):
        return self.dur_hi

    def _sample_shape(self, n, dev):
        return {'dur': torch.randint(self.dur_lo, self.dur_hi + 1, (n, 1), device=dev)}

    def _spec_shape(self, spec, n, dev):
        return {'dur': _as_col(spec.get("dur_steps", self._ms(300)), n, dev)}

    def _envelope(self, tg, start, shape):
        end = start + shape['dur']
        return ((tg >= start) & (tg < end)).float(), end, start


class NHPRamp(_NHPLoad):
    """Hold the home posture against a torque trapezoid: a linear rise, a plateau, and a linear
    fall as long as the rise. The plateau is shorter than the recorded ~3.7 s by default to keep
    episodes affordable. Trials are aligned on the END of the rise, as in the recordings."""
    name = "nhp_ramp"
    _shape_keys = ("rise_steps", "plateau_steps")

    def __init__(self, effector, onset_range_ms=(300, 1000), rise_range_ms=(400, 800),
                 plateau_range_ms=(1000, 2000), torque_range_nm=(0.05, 0.6), test_torque_nm=0.27,
                 settle_ms=500, **kwargs):
        steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.rise_lo, self.rise_hi = steps(rise_range_ms[0]), steps(rise_range_ms[1])
        self.plat_lo, self.plat_hi = steps(plateau_range_ms[0]), steps(plateau_range_ms[1])
        super().__init__(effector, onset_range_ms, torque_range_nm, test_torque_nm, settle_ms,
                         **kwargs)

    def _max_load_steps(self):
        return 2 * self.rise_hi + self.plat_hi

    def _sample_shape(self, n, dev):
        return {'rise': torch.randint(self.rise_lo, self.rise_hi + 1, (n, 1), device=dev),
                'plateau': torch.randint(self.plat_lo, self.plat_hi + 1, (n, 1), device=dev)}

    def _spec_shape(self, spec, n, dev):
        return {'rise': _as_col(spec.get("rise_steps", self._ms(600)), n, dev),
                'plateau': _as_col(spec.get("plateau_steps", self.plat_hi), n, dev)}

    def _envelope(self, tg, start, shape):
        rise, plateau = shape['rise'], shape['plateau']
        end = start + 2 * rise + plateau
        up = (tg - start).float() / rise.float()
        down = (end - tg).float() / rise.float()
        return torch.minimum(up, down).clamp(0.0, 1.0), end, start + rise


NHP_TASKS = ['nhp_center_out', 'nhp_pulse', 'nhp_ramp']

TASKS = {'delayed_reach': DelayedReaching, 'delayed_reach_posture': DelayedReachPosture,
         'horizon_sequence': HorizonSequence,
         'hold_posture_pulse': HoldPosturePulse, 'hold_posture_ramp': HoldPostureRamp,
         'pursuit': PursuitTask,
         'pacman': PacManTask,
         'nhp_center_out': NHPCenterOut, 'nhp_pulse': NHPPulse, 'nhp_ramp': NHPRamp}


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