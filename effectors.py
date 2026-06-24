"""
effectors.py -- effector models for the reaching task.

Three effectors, each with its own parameters and a common interface:
    PointMass    : 2D point mass actuated by an xy force.
    TwoJointArm  : two-joint planar arm actuated directly by joint torques.
    Arm26        : two-joint arm actuated by six muscles (arm26).

Common interface:
    eff.input_dim / eff.output_dim / eff.input_layout / eff.out_bias   # to build the controller
    eff.perturbation_dim       -> dimensionality of an external perturbation (xy force / joint torque)
    eff.sample_joint(n)        -> initial configuration (joint angles, or xy for the point mass)
    eff.joint_to_cart(theta)   -> fingertip xy (identity for the point mass)
    eff.rollout(controller, theta0, inp, perturbation=None) -> states  (SimpleNamespace)

Effector-level parameters (overridable via kwargs from the training script):
    dt, vis_delay_ms, pro_delay_ms,
    sho_range/elb_range (arm sampling range), joint_limits (arm range of motion sent to physics),
    pos_range (point mass), muscle_names (arm26), torque_scale / force_scale / mass.

An external perturbation (created by the task) is an xy force for the point mass and a signed
joint torque for the arms; it is added into the physics each step. None means no perturbation.
"""

import math
from types import SimpleNamespace
import torch

from physics import (point_mass_step, arm_step, arm_fingertip,
                     arm26_to, arm26_muscle_step, arm26_muscle_length,
                     muscle_activation_step)

# default arm range of motion (rad): shoulder 0-135 deg, elbow 0-155 deg
DEFAULT_JOINT_LIMITS = ((0.0, math.radians(135)), (0.0, math.radians(155)))


class Effector:
    """Base class: holds shared params and the generic delayed-feedback rollout."""
    name = "base"
    dof = 2
    output_dim = None
    out_bias = 0.0
    proprio_dim = None
    perturbation_dim = None
    action_names = []
    state_specs = {}

    def __init__(self, dt=0.01, vis_delay_ms=70, pro_delay_ms=25, **kwargs):
        self.dt = dt
        self.vis_delay_ms = vis_delay_ms
        self.pro_delay_ms = pro_delay_ms
        self.device = torch.device("cpu")

    # --- derived ---
    @property
    def vis_d(self): return max(1, round(self.vis_delay_ms / 1000 / self.dt))
    @property
    def pro_d(self): return max(1, round(self.pro_delay_ms / 1000 / self.dt))
    @property
    def input_layout(self):
        # instruction = [target_x*vis, target_y*vis, target_visible, go]
        return [('task', 4), ('vision', 2), ('proprio', self.proprio_dim)]
    @property
    def input_dim(self): return 4 + 2 + self.proprio_dim

    def to(self, device):
        self.device = torch.device(device)
        return self

    # --- to be provided by subclasses ---
    def sample_joint(self, n): raise NotImplementedError
    def joint_to_cart(self, theta): raise NotImplementedError
    def reset(self, theta0): raise NotImplementedError
    def act_from_output(self, out): raise NotImplementedError
    def step(self, st, action, pert=None): raise NotImplementedError
    def feedback(self, st): raise NotImplementedError
    def collect(self, st, fb): raise NotImplementedError

    # --- generic simulation with delayed visual + proprioceptive feedback ---
    def rollout(self, controller, theta0, inp, perturbation=None,
                obs_noise=0.0, neural_noise=0.0):
        """perturbation: optional (batch, steps, perturbation_dim) external force/torque, or None.

        obs_noise:    std of i.i.d. Gaussian noise added to the *observed* inputs each step:
                      the instruction / visual target (target xy + visibility & go cues), the
                      visual fingertip, and the proprioceptive feedback. Noise is in the native
                      units of each channel and corrupts only what the controller sees, not the
                      recorded ground-truth trajectory or the underlying instruction tensor.
        neural_noise: std of i.i.d. Gaussian noise injected into the hidden state each step,
                      after the controller update. It enters the recurrent dynamics (it
                      persists to the next step) and is what gets recorded in `hidden`.
        Both default to 0.0 (deterministic, identical to before)."""
        b, steps = theta0.shape[0], inp.shape[1]
        dev = theta0.device
        vis_d, pro_d = self.vis_d, self.pro_d

        controller.start_sequence()
        st = self.reset(theta0)
        h = torch.zeros(b, controller.hidden_dim, device=dev)
        fb0 = self.feedback(st)                                   # initial sensory values

        hist = {k: torch.zeros(b, steps, d, device=dev) for k, d in self.state_specs.items()}
        pro_h = torch.zeros(b, steps, self.proprio_dim, device=dev)
        h_hist = torch.zeros(b, steps, controller.hidden_dim, device=dev)
        for s in range(steps):
            # visual channel (70 ms): instruction + fingertip
            if s >= vis_d:
                inp_v = inp[:, s - vis_d, :]; ft_v = hist['pos'][:, s - vis_d, :]
            else:
                inp_v = inp[:, 0, :];         ft_v = fb0['fingertip']
            # proprioceptive channel (25 ms)
            pro_p = pro_h[:, s - pro_d, :] if s >= pro_d else fb0['proprio']

            # sensory observation noise: corrupt only the copy the controller sees.
            # the instruction (target xy + visibility/go cues) is treated as a visual input
            # and is noised too, alongside the fingertip and proprioception.
            if obs_noise > 0.0:
                inp_v = inp_v + obs_noise * torch.randn_like(inp_v)
                ft_v = ft_v + obs_noise * torch.randn_like(ft_v)
                pro_p = pro_p + obs_noise * torch.randn_like(pro_p)

            out, h = controller(torch.cat([inp_v, ft_v, pro_p], dim=1), h)
            # neural noise: enters the recurrent state (propagates) and is what we record
            if neural_noise > 0.0:
                h = h + neural_noise * torch.randn_like(h)
            h_hist[:, s, :] = h
            pert_s = perturbation[:, s, :] if perturbation is not None else None
            st = self.step(st, self.act_from_output(out), pert_s)
            fb = self.feedback(st)
            pro_h[:, s, :] = fb['proprio']
            for k, v in self.collect(st, fb).items():
                hist[k][:, s, :] = v
        return SimpleNamespace(hidden=h_hist, **hist) 


# ----------------------------------------------------------------------------- point mass
class PointMass(Effector):
    name = "point_mass"
    output_dim = 2
    out_bias = 0.0
    proprio_dim = 4                                   # position(2) + velocity(2)
    perturbation_dim = 2                              # xy force

    def __init__(self, dt=0.01, vis_delay_ms=70, pro_delay_ms=25,
                 pos_range=(-0.5, 0.5), mass=1.0, force_scale=10.0, **kwargs):
        super().__init__(dt, vis_delay_ms, pro_delay_ms, **kwargs)
        self.pos_range = pos_range
        self.mass = mass
        self.force_scale = force_scale
        self.action_names = ['Fx', 'Fy']
        self.state_specs = {'pos': 2, 'vel': 2, 'action': 2}

    def sample_joint(self, n):
        lo, hi = self.pos_range
        return torch.rand(n, 2, device=self.device) * (hi - lo) + lo

    def joint_to_cart(self, theta):
        return theta                                  # the "joint" IS the cartesian position

    def reset(self, theta0):
        b = theta0.shape[0]
        return {'pos': theta0.clone(), 'vel': torch.zeros(b, 2, device=theta0.device)}

    def act_from_output(self, out):
        return self.force_scale * (2.0 * out - 1.0)   # [0,1] -> signed force

    def step(self, st, force, pert=None):
        applied = force if pert is None else force + pert        # external perturbation force
        pos, vel = point_mass_step(st['pos'], st['vel'], applied, self.dt, self.mass)
        return {'pos': pos, 'vel': vel, 'force': force}          # store the control force (for effort)

    def feedback(self, st):
        return {'fingertip': st['pos'], 'vel': st['vel'],
                'proprio': torch.cat([st['pos'], st['vel']], dim=1)}

    def collect(self, st, fb):
        return {'pos': st['pos'], 'vel': st['vel'], 'action': st['force']}


# ----------------------------------------------------------------------------- torque arm
class TwoJointArm(Effector):
    name = "arm_torque"
    output_dim = 2
    out_bias = 0.0
    proprio_dim = 4                                   # joint angles(2) + joint velocities(2)
    perturbation_dim = 2                              # joint torque

    def __init__(self, dt=0.01, vis_delay_ms=70, pro_delay_ms=25,
                 sho_range=(0.35, 1.75), elb_range=(0.52, 2.18), torque_scale=10.0,
                 joint_limits=DEFAULT_JOINT_LIMITS, **kwargs):
        super().__init__(dt, vis_delay_ms, pro_delay_ms, **kwargs)
        self.sho_range = sho_range                    # sampling range (where targets are drawn)
        self.elb_range = elb_range
        self.torque_scale = torque_scale
        (slo, shi), (elo, ehi) = joint_limits         # range of motion -> physics
        self.lim = (float(slo), float(shi), float(elo), float(ehi))
        self.action_names = ['shoulder torque', 'elbow torque']
        self.state_specs = {'pos': 2, 'vel': 2, 'joints': 2, 'action': 2}

    def sample_joint(self, n):
        sho = torch.rand(n, 1, device=self.device) * (self.sho_range[1] - self.sho_range[0]) + self.sho_range[0]
        elb = torch.rand(n, 1, device=self.device) * (self.elb_range[1] - self.elb_range[0]) + self.elb_range[0]
        return torch.cat([sho, elb], dim=1)

    def joint_to_cart(self, theta):
        return arm_fingertip(theta, torch.zeros_like(theta))[:, :2]

    def reset(self, theta0):
        b = theta0.shape[0]
        return {'theta': theta0.clone(), 'omega': torch.zeros(b, 2, device=theta0.device),
                'tau': torch.zeros(b, 2, device=theta0.device)}

    def act_from_output(self, out):
        return self.torque_scale * (2.0 * out - 1.0)  # [0,1] -> signed torque

    def step(self, st, tau, pert=None):
        applied = tau if pert is None else tau + pert            # external perturbation torque
        theta, omega = arm_step(st['theta'], st['omega'], applied, self.dt, *self.lim)
        return {'theta': theta, 'omega': omega, 'tau': tau}      # store the control torque (for effort)

    def feedback(self, st):
        ft = arm_fingertip(st['theta'], st['omega'])
        return {'fingertip': ft[:, :2], 'vel': ft[:, 2:4],
                'proprio': torch.cat([st['theta'], st['omega']], dim=1)}

    def collect(self, st, fb):
        return {'pos': fb['fingertip'], 'vel': fb['vel'], 'joints': st['theta'], 'action': st['tau']}


# ----------------------------------------------------------------------------- muscle arm (arm26)
class Arm26(Effector):
    name = "arm26"
    output_dim = 6
    out_bias = -5.0                                   # start with low muscle activity
    proprio_dim = 8                                   # joint angles(2) + muscle lengths(6)
    perturbation_dim = 2                              # external joint torque

    def __init__(self, dt=0.01, vis_delay_ms=70, pro_delay_ms=25,
                 sho_range=(0.35, 1.75), elb_range=(0.52, 2.18), muscle_names=None,
                 joint_limits=DEFAULT_JOINT_LIMITS, **kwargs):
        super().__init__(dt, vis_delay_ms, pro_delay_ms, **kwargs)
        self.sho_range = sho_range                    # sampling range (where targets are drawn)
        self.elb_range = elb_range
        (slo, shi), (elo, ehi) = joint_limits         # range of motion -> physics
        self.lim = (float(slo), float(shi), float(elo), float(ehi))
        self.muscle_names = muscle_names or ['pectoralis', 'deltoid', 'brachioradialis',
                                             'triceps-lat', 'biceps', 'triceps-long']
        self.n_muscles = len(self.muscle_names)
        self.action_names = self.muscle_names
        self.state_specs = {'pos': 2, 'vel': 2, 'joints': 2,
                            'muscle_length': 6, 'activation': 6, 'action': 6}

    def to(self, device):
        super().to(device)
        arm26_to(device)                              # move the muscle constants onto the device
        return self

    def sample_joint(self, n):
        sho = torch.rand(n, 1, device=self.device) * (self.sho_range[1] - self.sho_range[0]) + self.sho_range[0]
        elb = torch.rand(n, 1, device=self.device) * (self.elb_range[1] - self.elb_range[0]) + self.elb_range[0]
        return torch.cat([sho, elb], dim=1)

    def joint_to_cart(self, theta):
        return arm_fingertip(theta, torch.zeros_like(theta))[:, :2]

    def reset(self, theta0):
        b = theta0.shape[0]
        return {'theta': theta0.clone(), 'omega': torch.zeros(b, 2, device=theta0.device),
                'act': torch.zeros(b, self.n_muscles, device=theta0.device)}

    def act_from_output(self, out):
        return out                                    # excitation in [0,1]

    def step(self, st, excitation, pert=None):
        act = muscle_activation_step(st['act'], excitation, self.dt)
        # external perturbation enters as a joint torque added to the muscle torque
        theta, omega = arm26_muscle_step(st['theta'], st['omega'], act, self.dt, pert, *self.lim)
        return {'theta': theta, 'omega': omega, 'act': act}

    def feedback(self, st):
        ft = arm_fingertip(st['theta'], st['omega'])
        ml = arm26_muscle_length(st['theta'])
        return {'fingertip': ft[:, :2], 'vel': ft[:, 2:4],
                'proprio': torch.cat([st['theta'], ml], dim=1), 'ml': ml}

    def collect(self, st, fb):
        return {'pos': fb['fingertip'], 'vel': fb['vel'], 'joints': st['theta'],
                'muscle_length': fb['ml'], 'activation': st['act'], 'action': st['act']}


# ----------------------------------------------------------------------------- factory
EFFECTORS = {'point_mass': PointMass, 'arm_torque': TwoJointArm, 'arm26': Arm26}

def make_effector(name, **kwargs):
    return EFFECTORS[name](**kwargs)