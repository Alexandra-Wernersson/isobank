"""A mock data set: for each set of parameters theta = (q, e0, zeta0) one curve phi(t), shaped like the (2,2) phase of an eccentric binary,
so the networks can be tried without any real model or data.

It is NOT a physical model. It only has the same structure as the real training targets: a smooth curve on a fixed time grid ending at t = 0,
  - growing fastest near t = 0,
  - with small oscillations whose size is set by e0 and which fade towards t = 0,
  - zeta0 setting where those oscillations start.
For the machine learning this is all that matters: a smooth map from 3 parameters to a curve of n_times values.

Usage: theta, t, phase = make_dataset(n_samples=3000, n_times=1024, seed=0)
"""
import numpy as np

PARAMETER_NAMES = ("q", "e0", "zeta0")
RANGES = {"q": (1.0, 8.0), "e0": (0.0, 0.5), "zeta0": (0.0, 2 * np.pi)}


def time_grid(n_times: int = 1024, t_start: float = -4000.0) -> np.ndarray:
    """times from t_start to 0, denser near t = 0, where the curve changes fastest"""
    u = np.linspace(0.0, 1.0, n_times)
    return t_start * (1.0 - u) ** 1.6


def phase_curve(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
    """the mock (2,2) phase for one theta = (q, e0, zeta0)"""
    q, e0, zeta0 = theta
    tau = (60.0 - t) / 60.0                        # shifted so it never reaches zero
    phi_orb = -6.0 * (1.0 + 0.08 * np.log(q)) * tau ** (5.0 / 8.0)
    phi_orb -= phi_orb[-1]                         # 0 at t = 0
    omega = np.gradient(phi_orb, t)                # grows towards t = 0
    e = e0 * (omega / omega[0]) ** (-19.0 / 18.0)  # the oscillations fade as omega grows
    return 2.0 * phi_orb + 2.0 * e * np.sin(0.92 * phi_orb + zeta0)


def make_dataset(n_samples: int = 3000, n_times: int = 1024, seed: int = 0):
    """theta (n, 3): q, e0, zeta0 drawn uniformly in RANGES; t (n_times,); phase (n, n_times)"""
    rng = np.random.default_rng(seed)
    lo = np.array([RANGES[k][0] for k in PARAMETER_NAMES])
    hi = np.array([RANGES[k][1] for k in PARAMETER_NAMES])
    theta = lo + rng.random((n_samples, 3)) * (hi - lo)
    t = time_grid(n_times)
    return theta, t, np.stack([phase_curve(th, t) for th in theta])


def network_inputs(theta: np.ndarray) -> np.ndarray:
    """the inputs the networks see: 1/q and e0 in [0, 1], and zeta0 as (sin, cos) so 0 and 2 pi are the same point"""
    q, e0, zeta0 = theta.T
    return np.stack([1.0 / q, e0 / RANGES["e0"][1], np.sin(zeta0), np.cos(zeta0)], axis=1).astype(np.float32)
