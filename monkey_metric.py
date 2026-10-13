"""monkey_metric.py -- score a contingent-capture test exactly as the monkey data are scored.

usage: python monkey_metric.py TAG [TAG ...] [--fig out.png] [--spec-name seq3cm]
       (reads experiments/_eval_TAG/test/<spec-name>, default "sequence")

Inter-target interval (ITI), the monkey definition:
    reach 1      : go cue -> hand-in of target 1
    reach k > 1  : hand-in of target k-1 -> hand-in of target k
where hand-in is the first moment the hand is inside the target. The 300 ms dwell is therefore
part of every middle ITI and of none of the first.

In the monkey task a broken dwell ends the trial, so the headline numbers use only trials in which
every target was held on its first entry ('clean' trials); the all-trials numbers (hand-in = the
entry that led to the capture) are printed next to them.

The second block aligns speed the way the monkey figure does (reach 1 to the go cue, reach k to the
capture of target k-1) and reports movement start (last time before the peak that speed is under
30% of that peak), peak time and peak speed.
"""
import argparse
import json

import numpy as np

np.seterr(all="ignore")


SPEC = "sequence"          # test set to read (folder under test/); set by --spec-name


def load(tag):
    D = "experiments/_eval_%s/test/%s/" % (tag, SPEC)
    d = dict(np.load(D + "states.npz", allow_pickle=True))
    tg = np.array(json.load(open(D + "spec_used.json"))["targets"])
    cfg = json.load(open("experiments/_eval_%s/config.json" % tag))
    return d, tg, cfg


def med(x):
    return np.median(x) if len(x) else np.nan


def entries(d, tg, radius, hold):
    """first[i,k]: first step inside target k after its own cue; last[i,k]: the entry that led to
    the capture (capture - hold); clean[i,k]: the two coincide, i.e. the dwell was never broken."""
    pos, cap, go = d["pos"], d["ts_capture_times"], d["ts_move_start"]
    T = pos.shape[1]
    n, R = cap.shape
    first = np.full((n, R), -1)
    last = np.full((n, R), -1)
    for i in range(n):
        for k in range(R):
            s0 = go[i] if k == 0 else cap[i, k - 1]
            if s0 >= T or cap[i, k] >= T:
                break
            inside = np.linalg.norm(pos[i, s0:cap[i, k]] - tg[i, k], axis=-1) < radius
            if inside.any():
                first[i, k] = s0 + int(np.argmax(inside))
            last[i, k] = cap[i, k] - hold
    # one step of slack: the capture counter and this threshold test can differ by a step
    clean = (first >= 0) & (np.abs(first - last) <= 1)
    return first, last, clean


def report(tag, ax_row=None):
    d, tg, cfg = load(tag)
    dt_ms = cfg["dt"] * 1000
    radius = cfg["capture_radius_cm"] / 100
    hold = int(round(cfg["capture_hold_ms"] / dt_ms))
    pos, cap, go, hor = d["pos"], d["ts_capture_times"], d["ts_move_start"], d["ts_horizon"]
    sp = np.linalg.norm(d["vel"], axis=-1)
    T = pos.shape[1]
    n, R = cap.shape
    first, last, clean = entries(d, tg, radius, hold)
    done = (cap < T).all(1)
    survive = clean.all(1)

    print("=" * 96)
    print("%s   n=%d trials, %d reaches, window %.1f cm, dwell %d ms" % (tag, n, R, radius * 100, hold * dt_ms))
    print("trials: all targets captured %s | every dwell kept first time (monkey rule) %s" % (
        " ".join("H%d %3.0f%%" % (h, 100 * done[hor == h].mean()) for h in (1, 2, 3)),
        " ".join("H%d %3.0f%%" % (h, 100 * survive[hor == h].mean()) for h in (1, 2, 3))))

    def iti(entry, trial_ok):
        out = {}
        for h in (1, 2, 3):
            rows = np.where((hor == h) & trial_ok)[0]
            v = np.full((len(rows), R), np.nan)
            for a, i in enumerate(rows):
                v[a, 0] = (entry[i, 0] - go[i]) * dt_ms
                v[a, 1:] = (entry[i, 1:] - entry[i, :-1]) * dt_ms
            out[h] = v
        return out

    for name, entry, ok in (("CLEAN trials (monkey rule)", first, survive), ("ALL completed trials", last, done)):
        res = iti(entry, ok)
        print("ITI, hand-in to hand-in (ms), median by reach -- %s" % name)
        print("        n   " + " ".join("%6s" % ("r%d" % (k + 1)) for k in range(R)) + "   | mid 2-%d: median / mean" % R)
        for h in (1, 2, 3):
            v = res[h]
            mid = v[:, 1:].ravel()
            print("   H%d %4d   " % (h, len(v)) + " ".join("%6.0f" % x for x in np.nanmedian(v, 0))
                  + "   | %6.0f / %6.0f" % (np.nanmedian(mid), np.nanmean(mid)) if len(v) else "   H%d    0" % h)

    # ---- speed, aligned as in the monkey figure
    print("speed aligned to the go cue (reach 1) or to the previous capture (middle reaches), completed trials")
    print("        start(30%% of peak) / peak time (ms)   peak speed / speed at the alignment event (cm/s)")
    for name, ks in (("reach 1 ", [0]), ("mid 2-%d" % R, range(1, R))):
        for h in (1, 2, 3):
            st, pt, pk, s0v = [], [], [], []
            for i in np.where((hor == h) & done)[0]:
                for k in ks:
                    s0 = go[i] if k == 0 else cap[i, k - 1]
                    seg = sp[i, s0:cap[i, k]]
                    j = int(np.argmax(seg))
                    b = np.where(seg[:j] < 0.3 * seg[j])[0]
                    st.append((b[-1] if len(b) else 0) * dt_ms)
                    pt.append(j * dt_ms); pk.append(seg[j] * 100); s0v.append(sp[i, s0] * 100)
            print("   %s H%d   start %4.0f   peak %4.0f      peak speed %5.1f   at event %4.1f" % (
                name, h, med(st), med(pt), med(pk), med(s0v)))

    if ax_row is not None:
        pre, post = int(200 / dt_ms), int(600 / dt_ms)
        t = np.arange(-pre, post) * dt_ms
        cols = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}
        for k in range(R):
            ax = ax_row[k]
            for h in (1, 2, 3):
                tr = []
                for i in np.where((hor == h) & done)[0]:
                    s0 = go[i] if k == 0 else cap[i, k - 1]
                    if s0 - pre < 0 or s0 + post > T:
                        continue
                    tr.append(sp[i, s0 - pre:s0 + post] * 100)
                if tr:
                    ax.plot(t, np.mean(tr, 0), color=cols[h], lw=1.8, label="H%d" % h)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_title("Reach %d" % (k + 1), fontsize=10)
            ax.set_xticks([-200, 0, 200, 400, 600])
            ax.set_xticklabels(["", "go" if k == 0 else "C%d" % k, "200", "", ""], fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        ax_row[0].set_ylabel("%s\nspeed (cm/s)" % tag)
        ax_row[-1].legend(frameon=False, fontsize=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--fig", default=None)
    ap.add_argument("--spec-name", default="sequence", help="test set folder under test/")
    a = ap.parse_args()
    SPEC = a.spec_name
    axes = None
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(a.tags), 7, figsize=(15, 2.6 * len(a.tags)), sharey="row", squeeze=False)
    for r, tag in enumerate(a.tags):
        report(tag, None if axes is None else axes[r])
    if a.fig:
        fig.supxlabel("Time (ms)")
        fig.tight_layout()
        fig.savefig(a.fig, dpi=150)
        print("figure ->", a.fig)
