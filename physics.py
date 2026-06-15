import math
import torch

# ----------------------------------------------------------------------------------------
# POINT MASS  (force applied directly in xy, exactly like your update_physics)
# ----------------------------------------------------------------------------------------
@torch.jit.script
def point_mass_step(pos, vel, force, dt: float = 0.01, mass: float = 1.0):
    """pos, vel, force: (batch, 2). Returns (pos, vel)."""
    vel = vel + (force / mass) * dt
    pos = pos + vel * dt          # semi-implicit (uses new vel); MotorNet uses old vel
    return pos, vel


# ----------------------------------------------------------------------------------------
# ARM26  --  two-link planar arm, driven by joint torques
# segment params (RigidTendonArm26 defaults): 1 = upper arm, 2 = forearm
#   m1=1.82, m2=1.43 (kg) | l1=0.309, l2=0.333 (m) | l1g=0.135, l2g=0.165 (m)
#   i1=0.051, i2=0.057 (kg.m^2) | joint limits: shoulder [0,135deg], elbow [0,155deg]
# ----------------------------------------------------------------------------------------
@torch.jit.script
def arm_step(theta, omega, tau, dt: float = 0.01):
    """One Euler step of the two-link arm.
    theta, omega, tau: (batch, 2) = [shoulder, elbow] angle / angular-velocity / torque.
    Returns (theta, omega).
    """
    # segment parameters -> inertia-matrix / Coriolis constants (locals: TorchScript can't
    # close over module globals; these fold to constants at compile time).
    m1 = 1.82;  m2 = 1.43
    l1 = 0.309; l2g = 0.165
    i1 = 0.051; i2 = 0.057
    l1g = 0.135
    ic11 = m1 * l1g * l1g + i1 + m2 * (l2g * l2g + l1 * l1) + i2
    ic12 = m2 * l2g * l2g + i2
    ic22 = m2 * l2g * l2g + i2
    im11 = 2.0 * m2 * l1 * l2g
    im12 = m2 * l1 * l2g
    cor  = m2 * l1 * l2g
    pi = 3.141592653589793
    ub0 = 135.0 * pi / 180.0
    ub1 = 155.0 * pi / 180.0

    elb = theta[:, 1]
    c2 = torch.cos(elb)
    s2 = torch.sin(elb)
    v0 = omega[:, 0]
    v1 = omega[:, 1]

    # configuration-dependent inertia matrix  M(q) = [[m11,m12],[m12,m22]]
    m11 = ic11 + c2 * im11
    m12 = ic12 + c2 * im12
    m22 = ic22

    # Coriolis / centrifugal torques
    cor0 = -cor * s2 * (2.0 * v0 + v1) * v1
    cor1 =  cor * s2 * v0 * v0

    rhs0 = tau[:, 0] - cor0
    rhs1 = tau[:, 1] - cor1

    # acc = M^{-1} @ rhs   (closed-form 2x2 inverse)
    det = m11 * m22 - m12 * m12
    acc0 = ( m22 * rhs0 - m12 * rhs1) / det
    acc1 = (-m12 * rhs0 + m11 * rhs1) / det

    omega = omega + torch.stack([acc0, acc1], dim=1) * dt
    theta = theta + omega * dt    # semi-implicit; MotorNet uses old omega here

    # enforce joint limits (drop these blocks for a free arm)
    lb = torch.zeros(2, device=theta.device)
    ub = torch.tensor([ub0, ub1], device=theta.device)
    omega = torch.where(torch.logical_and(omega < 0, theta <= lb), torch.zeros_like(omega), omega)
    omega = torch.where(torch.logical_and(omega > 0, theta >= ub), torch.zeros_like(omega), omega)
    theta = torch.minimum(torch.maximum(theta, lb), ub)
    return theta, omega


@torch.jit.script
def arm_fingertip(theta, omega):
    """Forward kinematics: joint state -> fingertip (x, y, vx, vy).  theta, omega: (batch, 2)."""
    l1 = 0.309; l2 = 0.333
    sho = theta[:, 0]
    elb = theta[:, 1]
    s = sho + elb
    c1 = torch.cos(sho); s1 = torch.sin(sho)
    c12 = torch.cos(s);  s12 = torch.sin(s)
    x = l1 * c1 + l2 * c12
    y = l1 * s1 + l2 * s12
    vx = -(l1 * s1 + l2 * s12) * omega[:, 0] - l2 * s12 * omega[:, 1]
    vy =  (l1 * c1 + l2 * c12) * omega[:, 0] + l2 * c12 * omega[:, 1]
    return torch.stack([x, y, vx, vy], dim=1)


# ----------------------------------------------------------------------------------------
# OPTIONAL: drive the arm with 6 muscles instead of raw joint torques
# (Kistemaker et al. 2010 moment-arm geometry; ReLU muscle force = activation * Fmax)
# order: pectoralis, deltoid, brachioradialis, triceps-lat, biceps, triceps-long
# These are eager (not scripted), so module-level constant tensors are fine here.
# ----------------------------------------------------------------------------------------
_FMAX = torch.tensor([838., 1207., 1422., 1549., 414., 603.])           # (6,)
_A0 = torch.tensor([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989])     # (6,)
_A1 = torch.tensor([[-.03, .03, 0., 0., -.03, .03],
                    [0., 0., -.014, .025, -.016, .03]])                 # (2,6)
_A2 = torch.tensor([[0., 0., 0., 0., 0., 0.],
                    [0., 0., -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]])        # (2,6)
_A3 = torch.tensor([math.pi / 2, 0.])                                   # (2,)  angle offsets


def arm26_to(device):
    """Move the muscle constants to `device` once before training."""
    global _FMAX, _A0, _A1, _A2, _A3
    _FMAX = _FMAX.to(device); _A0 = _A0.to(device); _A1 = _A1.to(device)
    _A2 = _A2.to(device); _A3 = _A3.to(device)


def arm26_moment_arms(theta, omega):
    """Returns moment arms (batch,2,6), musculotendon length (batch,1,6), velocity (batch,1,6)."""
    q = theta.unsqueeze(-1) - _A3.view(1, 2, 1)          # (batch,2,1)
    dq = omega.unsqueeze(-1)                              # (batch,2,1)
    a1 = _A1.unsqueeze(0); a2 = _A2.unsqueeze(0)          # (1,2,6)
    moment_arm = 2.0 * q * a2 + a1                        # (batch,2,6)
    length = ((a1 + q * a2) * q).sum(1, keepdim=True) + _A0.view(1, 1, 6)
    velocity = (dq * moment_arm).sum(1, keepdim=True)
    return moment_arm, length, velocity


@torch.jit.script
def muscle_activation_step(activation, excitation, dt: float = 0.01,
                           tau_act: float = 0.015, tau_deact: float = 0.05):
    """First-order Thelen-style activation dynamics (MotorNet defaults).
    activation, excitation: (batch, 6) in [0,1].  Returns new activation.
    Activation rises with time-constant tau_act, falls with tau_deact.
    """
    e = torch.clamp(excitation, 0.0, 1.0)
    a = torch.clamp(activation, 0.0, 1.0)
    tau_scaler = 0.5 + 1.5 * a
    tau = torch.where(e > a, tau_act * tau_scaler, tau_deact / tau_scaler)
    a = a + (e - a) / tau * dt
    return torch.clamp(a, 0.0, 1.0)


def arm26_muscle_length(theta):
    """Musculotendon length of the 6 muscles (no moment arms / velocity).  theta: (batch,2) -> (batch,6)."""
    q = theta.unsqueeze(-1) - _A3.view(1, 2, 1)
    a1 = _A1.unsqueeze(0); a2 = _A2.unsqueeze(0)
    return ((a1 + q * a2) * q).sum(1) + _A0.view(1, 6)


def arm26_muscle_step(theta, omega, activation, dt: float = 0.01):
    """Advance the arm one step given the current muscle activations.
    activation: (batch, 6) in [0,1] -- the muscle state AFTER activation dynamics.
    ReLU muscle:  force = activation * Fmax.  For a Hill muscle, replace the `force`
    line with a length/velocity-dependent force using `length` / `velocity` below.
    """
    moment_arm, length, velocity = arm26_moment_arms(theta, omega)
    force = activation.unsqueeze(1) * _FMAX.view(1, 1, 6)         # (batch,1,6)
    tau = -(force * moment_arm).sum(-1)                           # (batch,2) joint torques
    return arm_step(theta, omega, tau, dt)