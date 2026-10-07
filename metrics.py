"""metrics.py -- kinematic timing metrics for the horizon_sequence task.

Pure numpy/pandas, no torch: everything here operates on a dict of arrays shaped like a
test_model.py `states.npz` dump (states + inp + desired + the ts_* timestamp arrays), so it
works identically on a fresh rollout or a checkpoint's states.npz from months ago.

The central quantity is per-reach target arrival time (arrival_times): the time from a reach's
own go-pulse until the hand first comes within `tol` metres of that reach's target (read from
`desired` just before its own capture). This is a position-based measure, not a velocity-onset
one -- an earlier onset-detector approach (peak-speed-relative threshold crossing) was scrapped
because the arm's motion is continuous enough between reaches that it floors at ~0 for nearly
every reach regardless of horizon, making it blind to the real effect. Arrival time instead
directly asks "how long until the hand is actually near the target", which stays meaningful
whether or not the hand ever comes to a full stop.
"""
import numpy as np
import pandas as pd


def pulse_times(d):
    """(n, R) go-pulse step index for each reach: ts_move_start for reach 0, ts_capture_times[:,
    k-1] for reach k > 0 -- the time HorizonSequence actually cues that reach (tasks.py:580)."""
    move_start = np.asarray(d["ts_move_start"])
    capture = np.asarray(d["ts_capture_times"])
    n, R = capture.shape
    pt = np.empty((n, R), dtype=capture.dtype)
    pt[:, 0] = move_start
    if R > 1:
        pt[:, 1:] = capture[:, :-1]
    return pt


def unpreviewed_mask(d):
    """(n, R) bool: True iff this reach's target was not visible before its own go-pulse.

    Matches tasks.py HorizonSequence's own `unpreviewed` definition exactly: solving
    clamp(reach - horizon + 1, min=0) == reach for reach > 0 shows this is true iff horizon == 1
    (reach 0's target is always shown during the initial delay regardless of horizon, and any
    horizon >= 2 previews at least the next reach ahead)."""
    horizon = np.asarray(d["ts_horizon"] if "ts_horizon" in d else d["horizon"])
    R = np.asarray(d["ts_capture_times"]).shape[1]
    ks = np.arange(R)[None, :]
    return (horizon[:, None] == 1) & (ks > 0)


def arrival_times(d, dt, tol=0.01):
    """(n, R) time in ms from each reach's own go-pulse until the hand first comes within `tol`
    metres of that reach's target. NaN where the hand never gets within tolerance during the
    reach's own window [pulse_time, capture_time)."""
    pos = np.asarray(d["pos"])
    desired = np.asarray(d["desired"])
    bounds = np.asarray(d["ts_capture_times"])
    pt = pulse_times(d)
    n, R = pt.shape
    out = np.full((n, R), np.nan)
    for i in range(n):
        for r in range(R):
            s, e = int(pt[i, r]), int(bounds[i, r])
            if e <= s:
                continue
            target = desired[i, max(e - 1, 0), :]
            dist = np.linalg.norm(pos[i, s:e, :] - target[None, :], axis=1)
            idx = np.where(dist < tol)[0]
            if len(idx):
                out[i, r] = idx[0] * dt * 1000.0
    return out


def tidy_frame(d, dt, tol=0.01):
    """One row per (trial, reach): horizon, unpreviewed, pulse_time_ms, iri_ms (arrival time --
    see module docstring; NaN if the hand never arrived within tolerance), dwell_ms (the
    environment-scripted epoch duration), is_no_go, no_go_reach, excluded (no-go trial, a
    skipped (no_go_reach) reach, or arrival never detected)."""
    pt = pulse_times(d)
    bounds = np.asarray(d["ts_capture_times"])
    arr = arrival_times(d, dt, tol=tol)
    horizon = np.asarray(d["ts_horizon"] if "ts_horizon" in d else d["horizon"])
    unpv = unpreviewed_mask(d)
    n, R = pt.shape
    dwell_ms = (bounds - pt) * dt * 1000.0
    is_nogo = np.asarray(d["ts_is_no_go"]) if "ts_is_no_go" in d else np.zeros(n, dtype=bool)
    no_go_reach = (np.asarray(d["ts_no_go_reach"]) if "ts_no_go_reach" in d
                   else np.full(n, -1, dtype=int))

    rows = []
    for i in range(n):
        for k in range(R):
            never_arrived = bool(np.isnan(arr[i, k]))
            excluded = bool(is_nogo[i]) or (no_go_reach[i] == k) or never_arrived
            rows.append({
                "trial": i, "reach": k, "horizon": int(horizon[i]),
                "unpreviewed": bool(unpv[i, k]),
                "pulse_time_ms": float(pt[i, k]) * dt * 1000.0,
                "iri_ms": float(arr[i, k]) if not never_arrived else np.nan,
                "dwell_ms": float(dwell_ms[i, k]),
                "is_no_go": bool(is_nogo[i]),
                "no_go_reach": int(no_go_reach[i]),
                "excluded": excluded,
            })
    return pd.DataFrame(rows)


def summarize(df, by="horizon", metric="iri_ms", exclude_flagged=True):
    """Grouped mean/median/std/sem/count of `metric`, ready for a table or plot."""
    d = df
    if exclude_flagged:
        d = d[~d["excluded"]]
    g = d.groupby(by)[metric]
    out = g.agg(["mean", "median", "std", "count"]).reset_index()
    out["sem"] = out["std"] / out["count"].clip(lower=1) ** 0.5
    return out
