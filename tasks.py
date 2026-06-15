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
        return theta0, inp, desired


TASKS = {'delayed_reaching': DelayedReaching}

def make_task(name, effector, **kwargs):
    return TASKS[name](effector, **kwargs)