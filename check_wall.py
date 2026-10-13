"""check_wall.py -- the dwell-wall mask from ContingentStepper, checked on a trained model.

usage: python check_wall.py [experiments/_eval_G200]
Rolls out the 3 cm grid test (seq3cm.json) and checks that
  1. wall steps only occur between a target's go and its capture, never at or after the capture;
  2. on every wall step `desired` is the live target;
  3. the first wall step of a reach is the hand's first entry into that target's window;
  4. the hinge is zero on dwells that were kept and positive on dwells that were broken;
  5. the hinge has a gradient with respect to the network.
"""
import json
import sys

import numpy as np
import torch

from tasks import rollout_batch
from test_model import load_experiment

folder = sys.argv[1] if len(sys.argv) > 1 else "experiments/_eval_G200"
eff, controller, task, cfg = load_experiment(folder, "cpu")
spec = json.load(open("seq3cm.json"))
keep = list(range(0, 300, 10))                                   # 10 trials per horizon
for k in ("start", "targets", "horizon"):
    spec[k] = [spec[k][i] for i in keep]
torch.manual_seed(0)
states, desired, ts = rollout_batch(eff, controller, task.make_batch(spec=spec))
wall, cap, go = ts["wall"], ts["capture_times"], ts["move_start"]
tg = torch.tensor(spec["targets"], dtype=torch.float32)
n, T = wall.shape
R = cap.shape[1]
assert wall.any(), "no wall steps at all"
r = task.capture_radius
dist = (states.pos - desired).pow(2).sum(-1).add(1e-12).sqrt()
hinge = (dist - r).clamp(min=0) * wall.float()

n_kept = n_broken = 0
for i in range(n):
    for k in range(R):
        s0 = int(go[i]) if k == 0 else int(cap[i, k - 1])
        e = int(cap[i, k])
        if s0 >= T:
            break
        seg = slice(s0, min(e, T))
        w = wall[i, seg]
        if e < T:
            assert not wall[i, e], "wall still on at the capture step"
        if not w.any():
            continue
        assert torch.allclose(desired[i, seg][w], tg[i, k].expand(int(w.sum()), 2), atol=1e-6), "desired is not the live target on a wall step"
        # the stepper sees the position at the end of the previous step
        inside_prev = ((states.pos[i, s0 - 1:min(e, T) - 1] - tg[i, k]) ** 2).sum(-1) < r ** 2
        first = int(torch.nonzero(w)[0])
        assert bool(inside_prev[first]) and not bool(inside_prev[:first].any()), "wall does not start at the first entry"
        assert bool(w[first:].all()), "wall switched off before the capture"
        if e < T:
            broken = (e - (s0 + first)) > task.capture_hold_steps
            h = float(hinge[i, seg].sum())
            if broken:
                n_broken += 1
                assert h > 0, "a broken dwell was not charged"
            else:
                n_kept += 1
                assert h == 0, "a kept dwell was charged (%.6f)" % h
assert n_kept > 0 and n_broken > 0, "need both kept and broken dwells to test (%d, %d)" % (n_kept, n_broken)
controller.zero_grad()
hinge.mean().backward()
g = sum(float(p.grad.abs().sum()) for p in controller.parameters() if p.grad is not None)
assert np.isfinite(g) and g > 0, "no gradient from the wall"
hor = np.array(spec["horizon"])
per = hinge.detach().sum(1).numpy() * 100
print("kept dwells %d (charged 0), broken dwells %d (all charged); gradient OK" % (n_kept, n_broken))
print("wall charge per trial (cm x steps), mean by horizon:", {int(h): round(float(per[hor == h].mean()), 1) for h in (1, 2, 3)})
print("check_wall: all passed")
