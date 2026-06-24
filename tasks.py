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
"""

import numpy as np
import torch


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

class DelayedReaching:
    """Reach to a target after a go cue; some trials are no-go (hold at start)."""
    name = "delayed_reaching"

    def __init__(self, effector, steps=100, go_range=(20, 50), prob_no_go=0.3, **kwargs):
        self.effector = effector
        self.steps = steps
        self.go_range = tuple(go_range)
        self.prob_no_go = prob_no_go

    def make_batch(self, n=None, spec=None):
        """Random batch when spec is None (training), or an explicit batch from a spec dict.

        spec keys (all optional except start/target; scalars broadcast, or give length-n lists):
            start / target           : initial config and target (see start_space/target_space)
            start_space  ('joint')   : 'joint' or 'cartesian'
            target_space ('cartesian'): 'cartesian' or 'joint'
            go_time                  : go-cue onset step (default: midpoint of go_range)
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
            perturbation = None
        else:
            theta0, target, n = _resolve_start_target(eff, spec, dev)
            steps   = int(spec.get("steps", self.steps))
            go_def  = (self.go_range[0] + self.go_range[1]) // 2
            go_time = _as_col(spec.get("go_time", go_def), n, dev, torch.long)
            nogo    = _as_col(spec.get("no_go", False), n, dev, torch.bool)
            perturbation = _constant_perturbation(eff, spec.get("perturbation"), n, steps, dev)

        start = eff.joint_to_cart(theta0)
        tgrid = torch.arange(steps, device=dev).unsqueeze(0).expand(n, steps)
        go_mask = (tgrid >= go_time) & ~nogo

        inp = torch.zeros(n, steps, 4, device=dev)
        inp[:, :, 0:2] = target.unsqueeze(1)
        inp[:, :, 2]   = 1.0              # target always visible in this task
        inp[:, :, 3]   = go_mask.float()

        desired = torch.where(go_mask.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))

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
        3. movement      : target shown, go = 1; reach to it.  duration = move_ms (fixed)
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
                 final_input='null', prob_no_go=0.4, **kwargs):
        self.effector = effector
        self.prob_no_go = prob_no_go
        assert final_input in ('null', 'target')
        self.final_input = final_input

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
            perturbation = None
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

        go = in_move.float() * (~nogo).float()                            # no go cue on no-go trials
        show_target = in_delay | in_move
        if final_input == 'target':
            show_target = show_target | in_final

        vis = show_target.float().unsqueeze(-1)                # (n, T, 1)
        xy  = target.unsqueeze(1) * vis                        # zero when target hidden
        inp = torch.cat([xy, vis, go.unsqueeze(-1)], dim=-1)   # (n, T, 4)

        reach = (in_move | in_final) & ~nogo                              # no-go trials hold at start
        desired = torch.where(reach.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))

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