import torch

class DelayedReaching:
    """Reach to a target after a go cue; some trials are no-go (hold at start)."""
    name = "delayed_reaching"

    def __init__(self, effector, steps=100, go_range=(20, 50), prob_no_go=0.3, **kwargs):
        self.effector = effector
        self.steps = steps
        self.go_range = tuple(go_range)
        self.prob_no_go = prob_no_go

    def make_batch(self, n):
        eff, dev, steps = self.effector, self.effector.device, self.steps

        theta0 = eff.sample_joint(n)
        theta_t = eff.sample_joint(n)
        start = eff.joint_to_cart(theta0)
        target = eff.joint_to_cart(theta_t)

        go_time = torch.randint(self.go_range[0], self.go_range[1], (n, 1), device=dev)
        tgrid = torch.arange(steps, device=dev).unsqueeze(0).expand(n, steps)
        go_mask = tgrid >= go_time
        nogo = torch.rand(n, 1, device=dev) < self.prob_no_go
        go_mask = go_mask & ~nogo

        inp = torch.zeros(n, steps, 3, device=dev)
        inp[:, :, 0:2] = target.unsqueeze(1)
        inp[:, :, 2] = go_mask.float()

        desired = torch.where(go_mask.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))

        # per-trial epoch timestamps (step indices); not used in training, handy for analysis
        timestamps = {
            'go_start':    go_time.squeeze(-1),                                       # movement onset
            'episode_end': torch.full((n,), steps, dtype=torch.long, device=dev),
            'is_no_go':    nogo.squeeze(-1),
        }
        return theta0, inp, desired, timestamps


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

    def __init__(self, effector, init_range_ms=(300, 500), delay_range_ms=(300, 700),
                 move_ms=1200, final_range_ms=(300, 500), null_value=-2.0,
                 final_input='null', prob_no_go=0.4, **kwargs):
        self.effector = effector
        self.null_value = null_value
        self.prob_no_go = prob_no_go
        assert final_input in ('null', 'target')
        self.final_input = final_input

        ms2steps = lambda ms: max(1, round(ms / 1000 / effector.dt))
        self.init_lo,  self.init_hi  = ms2steps(init_range_ms[0]),  ms2steps(init_range_ms[1])
        self.delay_lo, self.delay_hi = ms2steps(delay_range_ms[0]), ms2steps(delay_range_ms[1])
        self.move = ms2steps(move_ms)
        self.final_lo, self.final_hi = ms2steps(final_range_ms[0]), ms2steps(final_range_ms[1])
        self.steps = self.init_hi + self.delay_hi + self.move + self.final_hi

    def make_batch(self, n):
        eff, dev, T = self.effector, self.effector.device, self.steps

        theta0 = eff.sample_joint(n)
        theta_t = eff.sample_joint(n)
        start = eff.joint_to_cart(theta0)                                   # (n, 2)
        target = eff.joint_to_cart(theta_t)                                 # (n, 2)

        d1 = torch.randint(self.init_lo,  self.init_hi + 1,  (n, 1), device=dev)
        d2 = torch.randint(self.delay_lo, self.delay_hi + 1, (n, 1), device=dev)
        d4 = torch.randint(self.final_lo, self.final_hi + 1, (n, 1), device=dev)
        pre = T - (d1 + d2 + self.move + d4)                                # slack -> initial hold (>=0)

        t1 = pre + d1                                                       # init -> delay
        t2 = t1 + d2                                                        # delay -> move
        t3 = t2 + self.move                                                 # move -> final
        tg = torch.arange(T, device=dev).unsqueeze(0)                      # (1, T)

        in_delay = (tg >= t1) & (tg < t2)
        in_move = (tg >= t2) & (tg < t3)
        in_final = tg >= t3

        nogo = torch.rand(n, 1, device=dev) < self.prob_no_go              # (n, 1)
        go = in_move.float() * (~nogo).float()                            # no go cue on no-go trials
        show_target = in_delay | in_move
        if self.final_input == 'target':
            show_target = show_target | in_final

        null = torch.full((n, T, 2), self.null_value, device=dev)
        xy = torch.where(show_target.unsqueeze(-1), target.unsqueeze(1), null)
        inp = torch.cat([xy, go.unsqueeze(-1)], dim=-1)                    # (n, T, 3)

        reach = (in_move | in_final) & ~nogo                              # no-go trials hold at start
        desired = torch.where(reach.unsqueeze(-1), target.unsqueeze(1), start.unsqueeze(1))

        # per-trial epoch boundaries (step indices); not used in training, handy for analysis
        timestamps = {
            'init_start':  torch.zeros(n, dtype=torch.long, device=dev),    # initial hold begins
            'delay_start': t1.squeeze(-1),                                  # target appears
            'move_start':  t2.squeeze(-1),                                  # go onset (movement)
            'final_start': t3.squeeze(-1),                                  # movement end / final hold
            'episode_end': torch.full((n,), T, dtype=torch.long, device=dev),
            'is_no_go':    nogo.squeeze(-1),
        }
        return theta0, inp, desired, timestamps


TASKS = {'delayed_reach': DelayedReaching, 'delayed_reach_posture': DelayedReachPosture}

def make_task(name, effector, **kwargs):
    return TASKS[name](effector, **kwargs)