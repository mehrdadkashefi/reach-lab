"""check_dwell_speed.py -- the dwell-speed cost (--w-loss-dwell-speed), checked on a trained model.

usage: python check_dwell_speed.py [experiments/_eval_W3] [free speed cm/s, default 2.5] [grace ms ...]
Rolls out the 3 cm grid test (seq3cm.json) and checks that
  1. the cost is charged only on wall steps (first entry into the live target until its capture),
     and none in the grace period after the first entry;
  2. a step under the free speed costs nothing, and a wall step over it costs speed - free speed;
  3. the cost has a gradient with respect to the network.
Then prints the cost by horizon for each grace period, next to the wall's, to set the weight.
"""
import json
import sys

import numpy as np
import torch

from tasks import rollout_batch
from test_model import load_experiment

folder = sys.argv[1] if len(sys.argv) > 1 else "experiments/_eval_W3"
free = (float(sys.argv[2]) if len(sys.argv) > 2 else 2.5) / 100.0
graces = [float(x) for x in sys.argv[3:]] or [0.0]
eff, controller, task, cfg = load_experiment(folder, "cpu")
spec = json.load(open("seq3cm.json"))
keep = list(range(0, 300, 10))                                   # 10 trials per horizon
for k in ("start", "targets", "horizon"):
    spec[k] = [spec[k][i] for i in keep]
torch.manual_seed(0)
states, desired, ts = rollout_batch(eff, controller, task.make_batch(spec=spec))
wall_all = ts["wall"].float()
assert wall_all.any(), "no wall steps at all"
speed = states.vel.pow(2).sum(-1).add(1e-12).sqrt()
r = task.capture_radius
dist = (states.pos - desired).pow(2).sum(-1).add(1e-12).sqrt()
hinge = ((dist - r).clamp(min=0) * wall_all).detach()
hor = np.array(spec["horizon"])
print("free speed %.1f cm/s; per-trial mean over all steps (the training loss averages this over trials)" % (free * 100))
for grace_ms in graces:
    # the same expressions as train.py
    _c = wall_all.cumsum(1)
    since = _c - (_c * (1 - wall_all)).cummax(1).values
    wall = wall_all * (since > grace_ms / 1000.0 / cfg["dt"]).float()
    cost = (speed - free).clamp(min=0) * wall
    assert float(cost[wall_all == 0].abs().sum()) == 0, "charged outside the wall steps"
    assert float(cost[(since > 0) & (wall == 0)].abs().sum()) == 0, "charged inside the grace period"
    slow = (speed < free) & (wall > 0)
    fast = (speed > free) & (wall > 0)
    assert float(cost[slow].abs().sum()) == 0, "a wall step under the free speed was charged"
    assert torch.allclose(cost[fast], speed[fast] - free), "a fast wall step was not charged speed - free"
    controller.zero_grad()
    cost.mean().backward(retain_graph=True)
    g = sum(float(p.grad.abs().sum()) for p in controller.parameters() if p.grad is not None)
    assert np.isfinite(g) and g > 0, "no gradient from the dwell-speed cost"
    cost = cost.detach()
    print("grace %3.0f ms: charged steps over the free speed %.0f%%" % (
        grace_ms, 100 * float(fast.sum()) / max(float((wall > 0).sum()), 1)))
    for h in (1, 2, 3):
        m = torch.tensor(hor == h)
        print("   H%d   dwell speed %.2e   wall %.2e   mean speed on charged steps %.1f cm/s" % (
            h, float(cost[m].mean()), float(hinge[m].mean()),
            100 * float((speed.detach()[m] * wall[m]).sum() / wall[m].sum().clamp(min=1))))
print("check_dwell_speed: all passed")
