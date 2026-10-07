"""analyze_iri.py -- target arrival-time analysis for horizon_sequence runs.

Reads the test/<spec>/states.npz already written by test_model.py for every experiment folder
and reports, per horizon, arrival time in ms -- how long after a reach's own go-pulse the hand
first comes within `--tol` metres of that reach's target (see metrics.py). Pure post-hoc
analysis: no model loading, no torch, no retraining.

    python analyze_iri.py --experiments-dir experiments/sequence --spec sequence
    python analyze_iri.py --only 2026-08-28_16-34-21 --spec sequence_segment

For each run folder this prints a horizon x arrival-time table and writes
<folder>/test/<spec>/iri_by_horizon.png (boxplot) next to the existing reaches.png /
diagnostics.png. With more than one run folder it also writes one cross-run trend figure,
<experiments-dir>/iri_trend_<spec>.png, comparing mean arrival time for unpreviewed (horizon-1)
vs. previewed (horizon 2/3) reaches across runs -- the debugging-history-at-a-glance plot.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from metrics import tidy_frame, summarize


def _load(folder, spec):
    npz_path = os.path.join(folder, "test", spec, "states.npz")
    cfg_path = os.path.join(folder, "config.json")
    if not os.path.exists(npz_path) or not os.path.exists(cfg_path):
        return None, None
    d = dict(np.load(npz_path, allow_pickle=True))
    with open(cfg_path) as f:
        cfg = json.load(f)
    return d, cfg


def analyze_folder(folder, spec, tol):
    """Load one run's states.npz for `spec` and return its tidy per-(trial, reach) frame, or
    None if that run has no such test dump."""
    d, cfg = _load(folder, spec)
    if d is None:
        return None
    dt = float(cfg.get("dt", 0.01) or 0.01)
    df = tidy_frame(d, dt, tol=tol)
    df["run"] = os.path.basename(folder)
    return df


def plot_by_horizon(df, out_path, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    dd = df[(~df["excluded"]) & (df["reach"] > 0)]
    sns.boxplot(data=dd, x="horizon", y="iri_ms", ax=ax, showfliers=False)
    sns.stripplot(data=dd, x="horizon", y="iri_ms", ax=ax, color="k", size=2, alpha=.3)
    ax.set_xlabel("horizon"); ax.set_ylabel("arrival time (ms)")
    fig.suptitle(title); fig.tight_layout()
    sns.despine(fig=fig, trim=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_trend(big_df, out_path, spec):
    """Mean arrival time for unpreviewed (H1, reach>0) vs. previewed (H2/H3) reaches, across
    runs -- lets the debugging history be read off at a glance."""
    d = big_df[(~big_df["excluded"]) & (big_df["reach"] > 0)]
    trend = (d.groupby(["run", "unpreviewed"])["iri_ms"].mean()
             .unstack("unpreviewed")
             .rename(columns={True: "H1 unpreviewed", False: "H2/H3 previewed"}))
    trend = trend.reindex(sorted(trend.index))
    fig, ax = plt.subplots(figsize=(8, 4))
    trend.plot(marker="o", ax=ax)
    ax.set_ylabel("mean arrival time (ms)"); ax.set_xlabel("run")
    ax.set_title(f"arrival-time trend across runs  ({spec})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    sns.despine(fig=fig, trim=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-dir", default="experiments/sequence")
    ap.add_argument("--spec", default="sequence",
                    help="test/<spec>/states.npz subfolder to analyze (e.g. sequence, sequence_segment)")
    ap.add_argument("--only", default=None, help="substring filter on the run folder name")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="metres; how close to the target counts as 'arrived' (default 1cm)")
    args = ap.parse_args()

    folders = sorted(f for f in glob.glob(os.path.join(args.experiments_dir, "*"))
                     if os.path.isdir(f))
    if args.only:
        folders = [f for f in folders if args.only in os.path.basename(f)]
    if not folders:
        print(f"no run folders under {args.experiments_dir}")
        return

    all_dfs = []
    header = f"{'run':32s} {'horizon':>7s} {'n':>4s} {'arrival mean':>12s} {'arrival med':>11s}"
    print(header)
    for folder in folders:
        df = analyze_folder(folder, args.spec, args.tol)
        if df is None:
            print(f"{os.path.basename(folder):32s}   (no test/{args.spec}/states.npz -- skipped)")
            continue
        all_dfs.append(df)
        name = os.path.basename(folder)
        by_h = summarize(df, metric="iri_ms").set_index("horizon")
        for h in sorted(df["horizon"].unique()):
            r = by_h.loc[h] if h in by_h.index else None
            print(f"{name:32s} {h:7d} {int(r['count']) if r is not None else 0:4d} "
                  f"{(r['mean'] if r is not None else float('nan')):12.1f} "
                  f"{(r['median'] if r is not None else float('nan')):11.1f}")
        out_dir = os.path.join(folder, "test", args.spec)
        plot_by_horizon(df, os.path.join(out_dir, "iri_by_horizon.png"), f"{name}  ({args.spec})")

    if len(all_dfs) > 1:
        big = pd.concat(all_dfs, ignore_index=True)
        trend_path = os.path.join(args.experiments_dir, f"iri_trend_{args.spec}.png")
        plot_trend(big, trend_path, args.spec)
        print(f"\ncross-run trend figure -> {trend_path}")


if __name__ == "__main__":
    main()
