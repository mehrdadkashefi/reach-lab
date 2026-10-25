import functools
import math
import torch


class _Scripted:
    """A physics function in two forms: TorchScript (used normally) and the plain Python original
    (used while torch.compile is tracing, since it cannot look inside TorchScript functions and
    would fall back to running them one operation at a time)."""

    def __init__(self, fn):
        self.py = fn
        self.script = torch.jit.script(fn)
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.py(*args, **kwargs)
        return self.script(*args, **kwargs)

# ----------------------------------------------------------------------------------------
# POINT MASS  (force applied directly in xy, exactly like your update_physics)
# ----------------------------------------------------------------------------------------
@_Scripted
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
@_Scripted
def arm_step(theta, omega, tau, dt: float = 0.01,
             sho_lo: float = 0.0, sho_hi: float = 2.356194490192345,
             elb_lo: float = 0.0, elb_hi: float = 2.705260340591211):
    """One Euler step of the two-link arm.
    theta, omega, tau: (batch, 2) = [shoulder, elbow] angle / angular-velocity / torque.
    sho_lo/sho_hi, elb_lo/elb_hi: joint range of motion (rad); defaults are 0-135 / 0-155 deg.
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

    # enforce joint limits (range of motion)
    lb = torch.tensor([sho_lo, elb_lo], device=theta.device)
    ub = torch.tensor([sho_hi, elb_hi], device=theta.device)
    omega = torch.where(torch.logical_and(omega < 0, theta <= lb), torch.zeros_like(omega), omega)
    omega = torch.where(torch.logical_and(omega > 0, theta >= ub), torch.zeros_like(omega), omega)
    theta = torch.minimum(torch.maximum(theta, lb), ub)
    return theta, omega


@_Scripted
def arm_jacobian_T_force(theta, force_xy):
    """Map a Cartesian fingertip force to joint torque: tau = J(theta)^T @ F.
    theta, force_xy: (batch, 2).  Returns joint torque (batch, 2) = [shoulder, elbow].
    J is the fingertip-position Jacobian (the linear-velocity block of arm_fingertip)."""
    l1 = 0.309; l2 = 0.333
    sho = theta[:, 0]
    s = sho + theta[:, 1]
    s1 = torch.sin(sho); c1 = torch.cos(sho)
    s12 = torch.sin(s);  c12 = torch.cos(s)
    # J = [[-l1 s1 - l2 s12, -l2 s12],
    #      [ l1 c1 + l2 c12,  l2 c12]]
    j11 = -(l1 * s1 + l2 * s12); j12 = -l2 * s12
    j21 =  (l1 * c1 + l2 * c12); j22 =  l2 * c12
    fx = force_xy[:, 0]; fy = force_xy[:, 1]
    tau_s = j11 * fx + j21 * fy
    tau_e = j12 * fx + j22 * fy
    return torch.stack([tau_s, tau_e], dim=1)


@_Scripted
def arm_endpoint_force(theta, tau):
    """Inverse of arm_jacobian_T_force: the Cartesian fingertip force F that a joint torque tau
    produces at posture theta, F = (J^T)^{-1} tau.  theta, tau: (batch, 2) -> force (batch, 2).
    Used for the isometric force readout (near a singular/extended posture det->0, so callers
    should keep postures away from full extension)."""
    l1 = 0.309; l2 = 0.333
    sho = theta[:, 0]
    s = sho + theta[:, 1]
    s1 = torch.sin(sho); c1 = torch.cos(sho)
    s12 = torch.sin(s);  c12 = torch.cos(s)
    # J^T = [[j11, j21], [j12, j22]] (rows map F->tau, see arm_jacobian_T_force)
    j11 = -(l1 * s1 + l2 * s12); j12 = -l2 * s12
    j21 =  (l1 * c1 + l2 * c12); j22 =  l2 * c12
    det = j11 * j22 - j21 * j12
    det = torch.where(det.abs() < 1e-6, torch.full_like(det, 1e-6), det)
    ts = tau[:, 0]; te = tau[:, 1]
    fx = ( j22 * ts - j21 * te) / det
    fy = (-j12 * ts + j11 * te) / det
    return torch.stack([fx, fy], dim=1)


@_Scripted
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
# (Kistemaker et al. 2010 moment-arm geometry, as MotorNet's RigidTendonArm26)
# order: pectoralis, deltoid, brachioradialis, triceps-lat, biceps, triceps-long
# Two force laws (muscle='hill' or 'relu'):
#   hill : MotorNet's RigidTendonHillMuscle. Force depends on activation, muscle length and
#          muscle velocity, plus a passive elastic force, so an active muscle resists being
#          moved (damping) and co-contraction stiffens the joint.
#   relu : force = activation * Fmax. No length or velocity dependence: the arm has no damping
#          or stiffness of its own beyond what the controller produces.
# These are eager (not scripted), so module-level constant tensors are fine here.
# ----------------------------------------------------------------------------------------
_FMAX = torch.tensor([838., 1207., 1422., 1549., 414., 603.])           # (6,)
_A0 = torch.tensor([0.151, 0.2322, 0.2859, 0.2355, 0.3329, 0.2989])     # (6,)
_A1 = torch.tensor([[-.03, .03, 0., 0., -.03, .03],
                    [0., 0., -.014, .025, -.016, .03]])                 # (2,6)
_A2 = torch.tensor([[0., 0., 0., 0., 0., 0.],
                    [0., 0., -4e-3, -2.2e-3, -5.7e-3, -3.2e-3]])        # (2,6)
_A3 = torch.tensor([math.pi / 2, 0.])                                   # (2,)  angle offsets
# RigidTendonHillMuscle constants (MotorNet RigidTendonArm26 defaults)
_L0_SE = torch.tensor([0.039, 0.066, 0.172, 0.187, 0.204, 0.217])     # tendon slack length (m)
_L0_CE = torch.tensor([0.134, 0.140, 0.092, 0.093, 0.137, 0.127])     # optimal fibre length (m)
_HILL_NS = 1.4                                    # normalised slack length of the passive element
_HILL_KPE = 1.0 / (1.66 - _HILL_NS) ** 2          # passive stiffness
_HILL_MIN_ACT = 0.001                             # MotorNet's minimum activation for this muscle


def arm26_to(device):
    """Move the muscle constants to `device` once before training."""
    global _FMAX, _A0, _A1, _A2, _A3, _L0_SE, _L0_CE
    _FMAX = _FMAX.to(device); _A0 = _A0.to(device); _A1 = _A1.to(device)
    _A2 = _A2.to(device); _A3 = _A3.to(device)
    _L0_SE = _L0_SE.to(device); _L0_CE = _L0_CE.to(device)


@_Scripted
def _hill_force(activation, length, velocity, l0_se, l0_ce, fmax, ns: float, kpe: float,
                min_act: float):
    """RigidTendonHillMuscle force in N (MotorNet), all in one scripted function so it runs as
    few GPU kernels as possible. activation, length, velocity: (batch, 6) activation and
    musculotendon length / velocity (rigid tendon: the fibre moves with the musculotendon)."""
    a = torch.clamp(activation, min=min_act)
    fibre = torch.clamp(length - l0_se, min=0.0)
    l_n = fibre / l0_ce                                 # fibre length / optimal
    v_n = velocity / (10.0 * l0_ce)                     # fibre velocity / vmax (negative = shortening)
    strain = torch.clamp(l_n - ns, min=0.0)
    flpe = kpe * strain * strain                        # passive elastic force
    flce = torch.clamp(1.0 - (l_n - 1.0) * (l_n - 1.0) / 0.4356, min=0.01)
    a_rel = torch.where(l_n > 1.0, 0.41 * flce, torch.full_like(l_n, 0.41))
    c = 1.0 - 0.9 * (a - 0.3) / (0.005 - 0.3)
    b_rel = torch.where(a < 0.3, 5.2 * c * c, torch.full_like(a, 5.2))
    d0 = a * (flce + a_rel) / b_rel
    fxa = flce * a
    s_as = 0.001
    # Guard (not in MotorNet): s_as - 2 d0 passes through zero when activation x force-length is
    # tiny (activation near its minimum on a muscle stretched far past optimal), sending the
    # eccentric force's gradient to ~1e7 and then inf / NaN. Keeping d0 >= s_as changes the force
    # only where the active force is below ~0.1% of Fmax.
    den = torch.clamp(s_as - 2.0 * d0, max=-s_as)
    p1 = -0.5 * fxa / den
    p2 = 0.25 * fxa * fxa / den
    p3 = -1.5 * fxa
    # both branches are evaluated for every element, so each is fed only velocities of its own sign
    # (a NaN or inf in the unused branch would still reach the gradient through torch.where)
    v_c = torch.clamp(v_n, max=0.0)
    concentric = (v_c * a * a_rel + fxa * b_rel) / (b_rel - v_c)
    v_e = torch.clamp(v_n, min=0.0)
    eccentric = (-p1 * p3 + p1 * s_as * v_e + p2 - p3 * v_e + s_as * v_e * v_e) / (p1 + v_e)
    active = torch.clamp(torch.where(v_n < 0.0, concentric, eccentric), min=0.0)
    return (active + flpe) * fmax


def arm26_muscle_force(activation, length, velocity, muscle: str = "hill"):
    """Muscle force (batch, 6) in N. length / velocity: musculotendon length and velocity
    (batch, 1, 6) or (batch, 6); with a rigid tendon the fibre moves with the musculotendon."""
    if muscle == "relu":
        return activation * _FMAX.view(1, 6)
    return _hill_force(activation, length.reshape(activation.shape),
                       velocity.reshape(activation.shape), _L0_SE.view(1, 6), _L0_CE.view(1, 6),
                       _FMAX.view(1, 6), _HILL_NS, _HILL_KPE, _HILL_MIN_ACT)


def arm26_moment_arms(theta, omega):
    """Returns moment arms (batch,2,6), musculotendon length (batch,1,6), velocity (batch,1,6)."""
    q = theta.unsqueeze(-1) - _A3.view(1, 2, 1)          # (batch,2,1)
    dq = omega.unsqueeze(-1)                              # (batch,2,1)
    a1 = _A1.unsqueeze(0); a2 = _A2.unsqueeze(0)          # (1,2,6)
    moment_arm = 2.0 * q * a2 + a1                        # (batch,2,6)
    length = ((a1 + q * a2) * q).sum(1, keepdim=True) + _A0.view(1, 1, 6)
    velocity = (dq * moment_arm).sum(1, keepdim=True)
    return moment_arm, length, velocity


@_Scripted
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


def arm26_muscle_torque(theta, activation, omega=None, muscle: str = "relu"):
    """Joint torque (batch,2) produced by the 6 muscles at posture theta given activation (batch,6),
    the same torque as arm26_muscle_step but without advancing the arm (for the endpoint-force
    readout). omega: joint velocity (only the Hill muscle depends on it; default at rest)."""
    if omega is None:
        omega = torch.zeros_like(theta)
    moment_arm, length, velocity = arm26_moment_arms(theta, omega)         # (batch,2,6)
    force = arm26_muscle_force(activation, length, velocity, muscle).unsqueeze(1)
    return -(force * moment_arm).sum(-1)                                    # (batch,2)


def joint_viscosity_torque(omega, viscosity):
    """Passive joint damping torque -B @ omega (batch, 2); viscosity B is a (2, 2) tensor in
    N.m.s/rad, or None for none."""
    if viscosity is None:
        return None
    return -omega @ viscosity.T


def arm26_muscle_step(theta, omega, activation, dt: float = 0.01, ext_tau=None,
                      sho_lo: float = 0.0, sho_hi: float = 2.356194490192345,
                      elb_lo: float = 0.0, elb_hi: float = 2.705260340591211,
                      muscle: str = "relu", viscosity=None):
    """Advance the arm one step given the current muscle activations.
    activation: (batch, 6) in [0,1] -- the muscle state AFTER activation dynamics.
    ext_tau:    optional (batch, 2) external joint torque added to the muscle torque
                (e.g. an experimental perturbation).
    sho_*/elb_*: joint range of motion (rad), forwarded to arm_step.
    muscle:     'relu' (force = activation * Fmax) or 'hill' (RigidTendonHillMuscle).
    viscosity:  optional (2, 2) passive joint damping matrix (N.m.s/rad).
    Muscle force is computed from the state at the start of the step, as in MotorNet.
    """
    moment_arm, length, velocity = arm26_moment_arms(theta, omega)
    force = arm26_muscle_force(activation, length, velocity, muscle).unsqueeze(1)  # (batch,1,6)
    tau = -(force * moment_arm).sum(-1)                           # (batch,2) joint torques
    if ext_tau is not None:
        tau = tau + ext_tau                                      # add external perturbation torque
    damp = joint_viscosity_torque(omega, viscosity)
    if damp is not None:
        tau = tau + damp
    return arm_step(theta, omega, tau, dt, sho_lo, sho_hi, elb_lo, elb_hi)