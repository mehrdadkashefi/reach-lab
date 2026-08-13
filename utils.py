import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


# ----------------------------------------------------------------------------- plot helpers
def fig_reaches(pos, desired, title):
    P, D = pos.detach().cpu().numpy(), desired.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    for j in range(pos.shape[0]):
        ax.plot(P[j, :, 0], P[j, :, 1], lw=.9)
        ax.plot(D[j, 0, 0], D[j, 0, 1], 'ko', ms=3); ax.plot(D[j, -1, 0], D[j, -1, 1], 'r*', ms=8)
    ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.tight_layout(); sns.despine(trim = True); 
    return fig


def fig_diagnostics(effector, states, inp, desired, title, num_trial=5, n_units=25):
    """Per-trial diagnostics. Rows: inputs, position, speed, force, action, neural. For an isometric
    force task (states has `force` and the task's target is a force), `desired` is a force profile and
    is drawn on the force row (dashed) instead of the position row; position/speed then just confirm
    the plant stayed put. The neural row shows a subset of hidden-unit activity."""
    t_axis = np.arange(desired.shape[1]) * effector.dt
    show_idx = np.random.choice(states.pos.shape[0], size=num_trial, replace=False)
    has_force = hasattr(states, 'force')
    has_hidden = hasattr(states, 'hidden')
    # heuristic: the target is a force (not a position) when the plant never moves -- i.e. the
    # isometric task. Detected from the states themselves so callers don't need to pass a flag.
    force_target = has_force and (states.pos - states.pos[:, :1]).abs().max().item() < 1e-4
    # the endpoint-force readout is only meaningful when isometric; for moving tasks the static
    # J^-T tau equivalent is huge/uninformative, so only show the force row for the force task.
    show_force = force_target
    # pick a fixed subset of hidden units to show (same across trials, for comparability)
    if has_hidden:
        H = states.hidden.shape[-1]
        unit_idx = np.linspace(0, H - 1, min(n_units, H)).round().astype(int)

    # rows are assembled dynamically; the force row is included only for the isometric task
    rows = ['inputs', 'position', 'speed']
    if show_force: rows.append('force')
    rows.append('action')
    if has_hidden: rows.append('neural')
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, len(show_idx), figsize=(2.7 * len(show_idx), 1.45 * nrows),
                             squeeze=False, sharex=True, sharey='row')
    for r, idx in enumerate(show_idx):
        pp = states.pos[idx].detach().cpu().numpy()
        ins = inp[idx].detach().cpu().numpy()
        v = states.vel[idx].detach().cpu().numpy()
        a = states.action[idx].detach().cpu().numpy()
        tgt = desired[idx].detach().cpu().numpy()
        ff = states.force[idx].detach().cpu().numpy() if has_force else None
        row = {name: axes[k, r] for k, name in enumerate(rows)}     # name -> axis for this column

        ax = row['inputs']
        ax.plot(t_axis, ins)
        if r == 0: ax.set_ylabel("Inputs")

        ax = row['position']
        ax.plot(t_axis, pp[:, 0], 'C0', label='x'); ax.plot(t_axis, pp[:, 1], 'C1', label='y')
        if not force_target:                                       # position target -> dashed on pos row
            ax.plot(t_axis, tgt[:, 0], ls='--', c='C0', lw=.7); ax.plot(t_axis, tgt[:, 1], ls='--', c='C1', lw=.7)
        if r == 0: ax.set_ylabel("position (m)"); ax.legend(fontsize=7, loc='upper right')

        ax = row['speed']
        ax.plot(t_axis, np.linalg.norm(v, axis=1), 'C3')
        if r == 0: ax.set_ylabel("speed (m/s)")

        if show_force:
            ax = row['force']
            ax.plot(t_axis, ff[:, 0], 'C0', label='Fx'); ax.plot(t_axis, ff[:, 1], 'C1', label='Fy')
            ax.plot(t_axis, tgt[:, 0], ls='--', c='C0', lw=.8); ax.plot(t_axis, tgt[:, 1], ls='--', c='C1', lw=.8)
            if r == 0: ax.set_ylabel("force (N)"); ax.legend(fontsize=7, loc='upper right')

        ax = row['action']
        for m, mname in enumerate(effector.action_names):
            ax.plot(t_axis, a[:, m], lw=1, label=mname)
        if r == 0: ax.set_ylabel("action a.u."); ax.legend(fontsize=6, ncol=2, loc='upper right')

        if has_hidden:
            hh = states.hidden[idx][:, unit_idx].detach().cpu().numpy()      # (T, n_units)
            row['neural'].plot(t_axis, hh, lw=.6, alpha=.7)
            if r == 0: row['neural'].set_ylabel(f"neural ({len(unit_idx)}/{H})")

        axes[nrows - 1, r].set_xlabel("time (s)")

    fig.suptitle(title, y=1.001); fig.tight_layout()
    sns.despine(trim=True)
    return fig


def fig_learning_curve(loss_hist, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_hist, lw=.7, alpha=.4, color='C0', label='loss')
    k = 20
    if len(loss_hist) >= k:
        ma = np.convolve(loss_hist, np.ones(k) / k, mode='valid')
        ax.plot(np.arange(k - 1, len(loss_hist)), ma, color='C1', lw=1.6, label=f'{k}-batch avg')
    ax.set_yscale('log'); ax.set_xlabel('training batch'); ax.set_ylabel('loss (log)')
    ax.legend(); ax.set_title(title); fig.tight_layout()
    return fig