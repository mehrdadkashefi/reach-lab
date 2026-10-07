"""check_task_delay.py -- the instruction stream is delayed by task_d, the hand's visual feedback
by vis_d, and the default (task_delay_ms=None) is the old single visual delay."""
import torch
from effectors import make_effector
from tasks import make_task


class Rec:
    hidden_dim = 4
    def __init__(self, out_dim): self.out_dim = out_dim; self.seen = []
    def start_sequence(self): self.seen = []
    def __call__(self, x, h):
        self.seen.append(x.clone())
        return 0.3 * torch.rand(x.shape[0], self.out_dim), h


def build(task_delay_ms):
    eff = make_effector('arm26', dt=0.01, vis_delay_ms=70, pro_delay_ms=25,
                        task_delay_ms=task_delay_ms, task_dim=10)
    task = make_task('horizon_sequence', eff, capture_mode='contingent', force_advance=False)
    return eff, task


eff, task = build(None)
assert eff.task_d == eff.vis_d == 7 and task.blind_steps == 7, "default must be the old behaviour"
for ms, td in ((120, 12), (200, 20)):
    eff, task = build(ms)
    assert eff.task_d == td and eff.vis_d == 7 and task.blind_steps == td
    torch.manual_seed(0)
    b, T = 4, 80
    inp = torch.randn(b, T, 10)
    ctl = Rec(eff.output_dim)
    st = eff.rollout(ctl, eff.sample_joint(b), inp)
    seen = torch.stack(ctl.seen, 1)                       # (b, T, 10 + 2 + proprio)
    assert torch.equal(seen[:, td:, :10], inp[:, :T - td]), "instruction not delayed by task_d"
    assert torch.equal(seen[:, :td, :10], inp[:, :1].expand(b, td, 10))
    assert torch.allclose(seen[:, 7:, 10:12], st.pos[:, :T - 7]), "fingertip not delayed by vis_d"
    assert (st.pos[:, -1] - st.pos[:, 0]).norm(dim=1).min() > 1e-3, "hand did not move; test is vacuous"
    print("task delay %d ms: instruction delayed %d steps, fingertip 7 steps  OK" % (ms, td))
print("check_task_delay: all passed")
