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


def fig_diagnostics(effector, states, inp, desired, title, num_trial=5):
    t_axis = np.arange(desired.shape[1]) * effector.dt
    show_idx = np.random.choice(states.pos.shape[0], size=num_trial, replace=False)

    fig, axes = plt.subplots(4, len(show_idx), figsize=(2.7 * len(show_idx), 6), squeeze=False, sharex=True, sharey = 'row')
    for r, idx in enumerate(show_idx):
        pp = states.pos[idx].detach().cpu().numpy()
        ins = inp[idx].detach().cpu().numpy()
        v = states.vel[idx].detach().cpu().numpy()
        a = states.action[idx].detach().cpu().numpy()
        tgt = desired[idx].detach().cpu().numpy()

        ax = axes[0, r]
        ax.plot(t_axis, ins)
        if r == 0: ax.set_ylabel(f"Inputs")

        ax = axes[1, r]
        ax.plot(t_axis, pp[:, 0], 'C0', label='x'); ax.plot(t_axis, pp[:, 1], 'C1', label='y')
        ax.plot(t_axis, tgt[:, 0], ls='--', c='C0', lw=.7); ax.plot(t_axis, tgt[:, 1], ls='--', c='C1', lw=.7)
        if r == 0: ax.set_ylabel(f"position (m)"); ax.legend(fontsize=7, loc='upper right')

        ax = axes[2, r]
        ax.plot(t_axis, np.linalg.norm(v, axis=1), 'C3')
        if r == 0:  ax.set_ylabel(f"speed (m/s)")

        ax = axes[3, r]
        for m, mname in enumerate(effector.action_names):
            ax.plot(t_axis, a[:, m], lw=1, label=mname)
        #if effector.name == "arm26": ax.set_ylim(-0.02, 1.0)
        if r == 0: ax.set_ylabel(f"action a.u."); ax.legend(fontsize=6, ncol=2, loc='upper right')

        if r == 3:
            for c in range(num_trial): axes[r, c].set_xlabel("time (s)")
    
    fig.suptitle(title, y=1.001); fig.tight_layout()
    sns.despine(trim = True)
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