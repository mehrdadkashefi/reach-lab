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

# ----------------------------------------------------------------------------- video
def _visible_targets(ins_t, n_ch):
    """Targets visible at one timestep, decoded from the instruction stream.

    10-wide (unified / sequence / horizon) layout is 3 slots x [x, y, on] + go, so slot 0 is the
    current target and slots 1-2 are the look-ahead ones. 4-wide native layout is [x, y, on, go].
    Returns a list of (x, y, slot_index) for the slots that are on.
    """
    out = []
    n_slots = 3 if n_ch >= 10 else 1
    for j in range(n_slots):
        x, y, on = ins_t[3 * j], ins_t[3 * j + 1], ins_t[3 * j + 2]
        if on > 0.5:
            out.append((float(x), float(y), j))
    return out


def video_trials(effector, states, inp, desired, path, num_trial=5, task_name="",
                 fps=25, n_units=20, per_trial=False, dpi=110, tail=40,
                 speed=1.0, dark=True):
    """Animate rollouts: the workspace with the effector and targets, plus live neural / actuator
    traces underneath.

    Top panel is the workspace: the two-link arm is drawn as a linkage (or a dot for the point
    mass), with the hand path traced behind it. Visible targets come straight from the instruction
    stream, so the sequence task's look-ahead targets appear automatically -- drawn as circles with
    the most immediate target brightest and each look-ahead step fainter. For the isometric force
    task the arm is drawn too -- it visibly does not move -- and the produced/target force is shown
    both as an arrow at the fingertip and as a trace panel.

    Lower panels show the full trace with a vertical cursor sweeping in time: hidden units, and
    either force (isometric task) or actuator commands.

    Playback is real time by default: simulation steps are decimated by 1/(fps*dt) so one second of
    video equals one second of simulated behaviour (`speed` scales this, e.g. 0.5 = half speed).

    Args:
        states/inp/desired: as returned by effector.rollout / task.make_batch.
        path:       output file. With per_trial=False (default) all trials are concatenated into
                    this one file; with per_trial=True one file per trial is written next to it,
                    named <stem>_trial00<ext>, ... . '.mp4' needs ffmpeg; '.gif' always works.
        num_trial:  how many trials to animate.
        tail:       length (steps) of the fading hand-path tail; None = full path.
        speed:      playback rate relative to real time (1.0 = real time, 0.5 = slow motion).
        dark:       black background with light foreground (default).
    """
    import os
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    P = states.pos.detach().cpu().numpy()
    D = desired.detach().cpu().numpy()
    I = inp.detach().cpu().numpy()
    n, T, _ = P.shape
    num_trial = min(num_trial, n)
    idx_list = list(range(num_trial))
    dt = effector.dt
    n_ch = I.shape[-1]

    is_arm = hasattr(states, 'joints')
    JT = states.joints.detach().cpu().numpy() if is_arm else None
    l1 = getattr(effector, 'l1', 0.309); l2 = getattr(effector, 'l2', 0.333)

    has_force = hasattr(states, 'force')
    F = states.force.detach().cpu().numpy() if has_force else None
    # isometric force task: the plant never moves, so the target is a force, not a position
    force_task = has_force and float(np.abs(P - P[:, :1]).max()) < 1e-4

    A = states.action.detach().cpu().numpy()
    H = states.hidden.detach().cpu().numpy() if hasattr(states, 'hidden') else None
    if H is not None:
        unit_idx = np.linspace(0, H.shape[2] - 1, min(n_units, H.shape[2])).astype(int)

    # real-time playback: show every `stride`-th simulation step so fps * stride * dt = speed
    stride = max(1, int(round(1.0 / max(1e-6, fps * dt * speed))))
    frame_steps = list(range(0, T, stride))

    # fixed y-ranges over the animated trials, so the vertical scale never jumps between trials
    def _range(arr, pad=0.06):
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-9:
            lo, hi = lo - 1e-3, hi + 1e-3
        m = pad * (hi - lo)
        return lo - m, hi + m
    V = states.vel.detach().cpu().numpy()
    act_lim = _range(A[idx_list])
    speed_lim = _range(np.linalg.norm(V[idx_list], axis=-1))
    neural_lim = _range(H[idx_list][:, :, unit_idx]) if H is not None else (0, 1)
    force_lim = _range(np.concatenate([F[idx_list], D[idx_list]], axis=1)) if force_task else (0, 1)
    # one fixed colour per actuator, reused across trials
    act_colors = [plt.get_cmap('tab10')(m % 10) for m in range(len(effector.action_names))]
    unit_colors = ([plt.get_cmap('turbo')(v) for v in np.linspace(0.05, 0.95, len(unit_idx))]
                   if H is not None else [])

    # theme
    if dark:
        BG, FG = 'black', '0.88'
        ARM_C, HAND_C, PATH_C, TGT_C = '0.75', 'deepskyblue', 'deepskyblue', 'tomato'
    else:
        BG, FG = 'white', '0.15'
        ARM_C, HAND_C, PATH_C, TGT_C = '0.35', 'C0', 'C0', 'C3'

    def _style(ax, despine=True):
        ax.set_facecolor(BG)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if not despine:
            for s in ('left', 'bottom'):
                ax.spines[s].set_visible(True)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(FG)
        ax.tick_params(colors=FG, labelsize=7)
        ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)

    def _legend(ax, **kw):
        lg = ax.legend(framealpha=0.0, labelcolor=FG, **kw)
        return lg

    # workspace limits: everything the hand, the targets, and the arm can reach in these trials.
    # On the isometric task `desired` / the instruction stream hold FORCES (newtons), not
    # positions, so they must not stretch the spatial axes -- use the arm geometry only.
    xs = [P[idx_list, :, 0]]
    ys = [P[idx_list, :, 1]]
    if not force_task:
        xs += [D[idx_list, :, 0], I[idx_list, :, 0]]
        ys += [D[idx_list, :, 1], I[idx_list, :, 1]]
    if is_arm:
        xs.append(np.array([0.0])); ys.append(np.array([0.0]))          # the shoulder
        # keep the whole linkage in view even when the hand barely moves (isometric)
        reach = l1 + l2
        xs.append(np.array([-0.15 * reach, 0.15 * reach]))
    elif force_task:
        # isometric point mass: the hand never moves, so give the panel a little extent around it
        cx, cy = float(P[idx_list, 0, 0].mean()), float(P[idx_list, 0, 1].mean())
        xs.append(np.array([cx - 0.1, cx + 0.1])); ys.append(np.array([cy - 0.1, cy + 0.1]))
    xlo, xhi = min(a.min() for a in xs), max(a.max() for a in xs)
    ylo, yhi = min(a.min() for a in ys), max(a.max() for a in ys)
    pad = 0.12 * max(xhi - xlo, yhi - ylo, 1e-3)
    xlim, ylim = (xlo - pad, xhi + pad), (ylo - pad, yhi + pad)
    # force arrows are drawn in metres: scale peak force to a readable fraction of the workspace
    fscale = (0.30 * max(xhi - xlo, yhi - ylo) / max(1e-6, float(np.abs(D[idx_list]).max()))
              if force_task else 0.0)

    def render(indices, outfile):
        nrow = 2 + (1 if H is not None else 0)
        fig = plt.figure(figsize=(6.4, 3.2 + 1.6 * nrow), facecolor=BG)
        gs = fig.add_gridspec(nrow + 1, 1, height_ratios=[3.2] + [1.25] * nrow, hspace=.5)
        ax_ws = fig.add_subplot(gs[0])
        ax_low = [fig.add_subplot(gs[k + 1]) for k in range(nrow)]
        for ax in [ax_ws] + ax_low:
            _style(ax)

        ax_ws.set_xlim(*xlim); ax_ws.set_ylim(*ylim); ax_ws.set_aspect('equal')
        ax_ws.set_xlabel('x (m)')
        # no y axis in the workspace: the vertical scale is set by the equal aspect ratio
        ax_ws.yaxis.set_visible(False)
        ax_ws.spines['left'].set_visible(False)
        ax_ws.xaxis.set_major_locator(plt.MaxNLocator(4))          # few ticks: the panel is narrow
        # workspace artists
        (path_ln,) = ax_ws.plot([], [], '-', color=PATH_C, lw=1.1, alpha=.6)
        (arm_ln,) = ax_ws.plot([], [], '-o', color=ARM_C, lw=3, ms=4, mfc=ARM_C)
        (hand_pt,) = ax_ws.plot([], [], 'o', color=HAND_C, ms=4.5, zorder=5)   # small circle cursor
        # targets: filled circles, larger than the cursor; the immediate target is brightest and
        # each look-ahead step is fainter
        slot_alpha = [1.0, 0.5, 0.25]
        tgt_pts = [ax_ws.plot([], [], 'o', ms=12, mfc=TGT_C, mec=TGT_C,
                              alpha=slot_alpha[j], zorder=4)[0] for j in range(3)]
        force_arrow = ax_ws.annotate("", xy=(0, 0), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle='-|>', color=TGT_C, lw=2))
        tgt_arrow = ax_ws.annotate("", xy=(0, 0), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle='-|>', color=FG, lw=1.5, ls='--'))
        title = ax_ws.set_title("", fontsize=10, color=FG)

        # lower panels: full traces (drawn per trial) + a sweeping time cursor
        t_axis = np.arange(T) * dt
        cursors = [ax.axvline(0, color=FG, lw=1) for ax in ax_low]
        for ax in ax_low:
            ax.set_xlim(0, (T - 1) * dt)
        ax_low[-1].set_xlabel('time (s)')
        lower_lines = {'act': [], 'force': [], 'hidden': []}

        # panel labels, fixed y-limits and legends are set once (not per trial), so the vertical
        # scale and the actuator colours stay identical across trials
        k = 0
        if force_task:
            ax_low[k].set_ylabel('force (N)'); ax_low[k].set_ylim(*force_lim)
        else:
            ax_low[k].set_ylabel('action'); ax_low[k].set_ylim(*act_lim)
        k += 1
        if force_task:
            ax_low[k].set_ylabel('action'); ax_low[k].set_ylim(*act_lim)
        else:
            ax_low[k].set_ylabel('speed (m/s)'); ax_low[k].set_ylim(*speed_lim)
        k += 1
        if H is not None:
            ax_low[k].set_ylabel(f'neural ({len(unit_idx)})'); ax_low[k].set_ylim(*neural_lim)
        # limits are final, so spines can be trimmed to the ticks
        for ax in ax_low:
            sns.despine(ax=ax, trim=True)
        sns.despine(ax=ax_ws, left=True, trim=True)

        def setup_trial(ti):
            """Draw the static full traces for trial ti; the cursor animates over them."""
            for key in lower_lines:
                for ln in lower_lines[key]:
                    ln.remove()
                lower_lines[key] = []
            k = 0
            if force_task:
                ax = ax_low[k]
                for c, lab in [(0, 'Fx'), (1, 'Fy')]:
                    col = act_colors[c % len(act_colors)]
                    lower_lines['force'] += ax.plot(t_axis, F[ti, :, c], color=col, lw=1.1, label=lab)
                    lower_lines['force'] += ax.plot(t_axis, D[ti, :, c], color=col, lw=.9, ls='--')
                if ti == indices[0]: _legend(ax, fontsize=6, loc='upper right', ncol=2)
            else:
                ax = ax_low[k]
                for m, mname in enumerate(effector.action_names):
                    lower_lines['act'] += ax.plot(t_axis, A[ti, :, m], color=act_colors[m],
                                                  lw=.9, label=mname)
                if ti == indices[0]: _legend(ax, fontsize=5, ncol=2, loc='upper right')
            k += 1
            # second panel: speed (moving tasks) or actuator commands (isometric task)
            ax = ax_low[k]
            if force_task:
                for m, mname in enumerate(effector.action_names):
                    lower_lines['act'] += ax.plot(t_axis, A[ti, :, m], color=act_colors[m],
                                                  lw=.9, label=mname)
                if ti == indices[0]: _legend(ax, fontsize=5, ncol=2, loc='upper right')
            else:
                lower_lines['act'] += ax.plot(t_axis, np.linalg.norm(V[ti], axis=1),
                                              color=TGT_C, lw=1.1)
            k += 1
            if H is not None:
                ax = ax_low[k]
                for u, ui in enumerate(unit_idx):
                    lower_lines['hidden'] += ax.plot(t_axis, H[ti][:, ui],
                                                     color=unit_colors[u], lw=.6, alpha=.75)
                ax.relim(); ax.autoscale_view(scaley=True)

        frames = [(ti, t) for ti in indices for t in frame_steps]

        def init():
            return []

        def update(fr):
            ti, t = fr
            if t == 0:
                setup_trial(ti)
            lo = 0 if tail is None else max(0, t - tail)
            path_ln.set_data(P[ti, lo:t + 1, 0], P[ti, lo:t + 1, 1])
            hand_pt.set_data([P[ti, t, 0]], [P[ti, t, 1]])
            if is_arm:
                sho, elb = JT[ti, t, 0], JT[ti, t, 1]
                ex, ey = l1 * np.cos(sho), l1 * np.sin(sho)
                hx, hy = ex + l2 * np.cos(sho + elb), ey + l2 * np.sin(sho + elb)
                arm_ln.set_data([0, ex, hx], [0, ey, hy])
            # targets straight from the instruction stream (slot 0 solid, look-ahead hollow).
            # On the isometric task the stream holds forces, which are drawn as arrows instead.
            vis = [] if force_task else _visible_targets(I[ti, t], n_ch)
            for j, pt in enumerate(tgt_pts):
                if j < len(vis):
                    pt.set_data([vis[j][0]], [vis[j][1]])
                    pt.set_alpha(slot_alpha[min(vis[j][2], len(slot_alpha) - 1)])
                else:
                    pt.set_data([], [])
            if force_task:                       # force arrows at the (static) fingertip
                bx, by = P[ti, t, 0], P[ti, t, 1]
                force_arrow.set_position((bx, by))
                force_arrow.xy = (bx + fscale * F[ti, t, 0], by + fscale * F[ti, t, 1])
                tgt_arrow.set_position((bx, by))
                tgt_arrow.xy = (bx + fscale * D[ti, t, 0], by + fscale * D[ti, t, 1])
            for c in cursors:
                c.set_xdata([t * dt, t * dt])
            title.set_text(f"{task_name}  trial {ti}   t = {t * dt:.2f} s")
            return []

        anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False,
                             interval=1000 / fps)
        ext = os.path.splitext(outfile)[1].lower()
        if ext == ".gif":
            writer = PillowWriter(fps=fps)
        else:
            try:
                writer = FFMpegWriter(fps=fps, bitrate=2400)
            except Exception:
                outfile = os.path.splitext(outfile)[0] + ".gif"
                writer = PillowWriter(fps=fps)
        anim.save(outfile, writer=writer, dpi=dpi,
                  savefig_kwargs={'facecolor': BG})
        plt.close(fig)
        return outfile

    written = []
    if per_trial:
        stem, ext = os.path.splitext(path)
        for ti in idx_list:
            written.append(render([ti], f"{stem}_trial{ti:02d}{ext}"))
    else:
        written.append(render(idx_list, path))          # all trials concatenated into one file
    return written