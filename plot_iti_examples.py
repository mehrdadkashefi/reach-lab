"""plot_iti_examples.py -- ITI plot (monkey definition) and one example sequence per horizon.

usage: python plot_iti_examples.py TAG [--spec-name seq3cm] [--out prefix]
Reads experiments/_eval_TAG/test/<spec-name>; writes <prefix>_iti.png and <prefix>_examples.png.

ITI: reach 1 = go cue -> hand-in; reach k = hand-in of target k-1 -> hand-in of target k, where
hand-in is the entry that led to the capture. Boxes show ALL completed trials; the right panel
compares the median over all trials, the mean over all trials, and the median over trials in which
no dwell was broken (the only trials a monkey would have completed).
The example for each horizon is the completed trial whose total duration is the median one.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import monkey_metric as mm

COL = {1: "#0072B2", 2: "#E69F00", 3: "#009E73"}       # the monkey figure's horizon colours

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--spec-name", default="sequence")
ap.add_argument("--out", default=None)
a = ap.parse_args()
mm.SPEC = a.spec_name
out = a.out or "fig_%s" % a.tag

d, tg, cfg = mm.load(a.tag)
start = np.array(json.load(open("experiments/_eval_%s/test/%s/spec_used.json" % (a.tag, a.spec_name)))["start"])
dt = cfg["dt"] * 1000
radius = cfg["capture_radius_cm"] / 100
hold = int(round(cfg["capture_hold_ms"] / dt))
pos, cap, go, hor = d["pos"], d["ts_capture_times"], d["ts_move_start"], d["ts_horizon"]
sp = np.linalg.norm(d["vel"], axis=-1) * 100
T = pos.shape[1]
n, R = cap.shape
first, last, clean = mm.entries(d, tg, radius, hold)
done = (cap < T).all(1)
survive = clean.all(1)


def iti(entry, rows):
    v = np.full((len(rows), R), np.nan)
    for j, i in enumerate(rows):
        v[j, 0] = (entry[i, 0] - go[i]) * dt
        v[j, 1:] = (entry[i, 1:] - entry[i, :-1]) * dt
    return v


# ------------------------------------------------------------------ ITI figure
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3.2, 1.5]})
for h in (1, 2, 3):
    v = iti(last, np.where((hor == h) & done)[0])
    x = np.arange(1, R + 1) + (h - 2) * 0.24
    bp = ax.boxplot([v[:, k] for k in range(R)], positions=x, widths=0.18, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=1.5),
                    whiskerprops=dict(color=COL[h]), capprops=dict(visible=False))
    for b in bp["boxes"]:
        b.set(facecolor=COL[h], edgecolor=COL[h])
    ax.plot([], [], color=COL[h], lw=6, label="Horizon %d  (%d of %d trials completed)" % (h, len(v), (hor == h).sum()))
    allv = v[:, 1:].ravel()
    cl = iti(first, np.where((hor == h) & survive)[0])[:, 1:].ravel()
    ax2.plot(h - 0.2, np.median(allv), "o", color=COL[h], ms=9)
    ax2.plot(h, np.mean(allv), "s", color=COL[h], ms=8)
    if len(cl):
        ax2.plot(h + 0.2, np.median(cl), "o", mfc="white", mec=COL[h], mew=2, ms=9)
        ax2.annotate("n=%d" % ((hor == h) & survive).sum(), (h + 0.2, np.median(cl)), xytext=(0, -14),
                     textcoords="offset points", ha="center", fontsize=7, color="0.35")
ax.set_xticks(range(1, R + 1)); ax.set_xticklabels(range(1, R + 1))
ax.set_xlabel("Reach #"); ax.set_ylabel("ITI (ms)")
ax.set_title("%s: inter-target interval, all completed trials" % a.tag, loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax2.plot([], [], "o", color="0.3", ms=8, label="median, all trials")
ax2.plot([], [], "s", color="0.3", ms=7, label="mean, all trials")
ax2.plot([], [], "o", mfc="white", mec="0.3", mew=2, ms=8, label="median, no dwell broken")
ax2.set_xticks([1, 2, 3]); ax2.set_xlim(0.5, 3.5); ax2.set_xlabel("Horizon")
ax2.set_title("Reaches 2-%d" % R, loc="left", fontsize=11)
ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
lo = min(ax.get_ylim()[0], ax2.get_ylim()[0]); hi = max(ax.get_ylim()[1], ax2.get_ylim()[1])
for x_ in (ax, ax2):
    x_.set_ylim(lo, hi)
    x_.grid(axis="y", color="0.9", lw=0.6); x_.set_axisbelow(True)
    for s in ("top", "right"):
        x_.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out + "_iti.png", dpi=150)

# ------------------------------------------------------------------ example sequences
G = np.unique(np.round(np.concatenate([tg.reshape(-1, 2), start]), 4), axis=0)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), gridspec_kw={"height_ratios": [2.4, 1]})
for c, h in enumerate((1, 2, 3)):
    rows = np.where((hor == h) & done)[0]
    dur = cap[rows, -1] - go[rows]
    i = rows[np.argsort(dur)[len(rows) // 2]]
    t0, t1 = go[i], min(T, cap[i, -1] + 20)
    ax = axes[0, c]
    for g in G:
        ax.add_patch(plt.Circle(g * 100, radius * 100, fc="none", ec="0.85", lw=0.8))
    seq = np.vstack([start[i], tg[i]]) * 100
    for k in range(R):
        ax.add_patch(plt.Circle(seq[k + 1], radius * 100, fc="none", ec="0.25", lw=1.2))
        ax.annotate(str(k + 1), seq[k + 1], xytext=(7, 7), textcoords="offset points", fontsize=9, color="0.25")
    ax.plot(*(pos[i, t0:t1] * 100).T, color=COL[h], lw=1.8)
    ax.plot(*seq[0], "o", color="k", ms=6); ax.annotate("start", seq[0], xytext=(7, -12), textcoords="offset points", fontsize=8)
    ax.plot([seq[:, 0].min() - 2.5, seq[:, 0].min() - 0.5], [seq[:, 1].min() - 2.2] * 2, color="k", lw=2)
    ax.annotate("2 cm", (seq[:, 0].min() - 1.5, seq[:, 1].min() - 2.2), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlim(seq[:, 0].min() - 3, seq[:, 0].max() + 3); ax.set_ylim(seq[:, 1].min() - 3, seq[:, 1].max() + 3)
    ax.set_aspect("equal"); ax.axis("off")
    nb = int((~clean[i]).sum())
    ax.set_title("Horizon %d   (go to last capture %.1f s; %d dwell%s broken)" % (h, (cap[i, -1] - go[i]) * dt / 1000, nb, "" if nb == 1 else "s"),
                 fontsize=10, color=COL[h])
    ax = axes[1, c]
    t = (np.arange(t0 - 30, t1) - t0) * dt / 1000
    ax.plot(t, sp[i, t0 - 30:t1], color=COL[h], lw=1.6)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    for k in range(R):
        ax.axvline((cap[i, k] - t0) * dt / 1000, color="0.6", lw=0.8)
        ax.annotate("C%d" % (k + 1), ((cap[i, k] - t0) * dt / 1000, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel("Time from go cue (s)")
    if c == 0:
        ax.set_ylabel("Speed (cm/s)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
ym = max(x_.get_ylim()[1] for x_ in axes[1]); [x_.set_ylim(0, ym) for x_ in axes[1]]
fig.suptitle("%s: the median-duration completed trial of each horizon (circles = %.1f cm capture windows; grey lines = captures)" % (a.tag, radius * 100), fontsize=10, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out + "_examples.png", dpi=150)
print("wrote", out + "_iti.png", out + "_examples.png")
