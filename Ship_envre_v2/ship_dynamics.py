"""
3-DOF surface-vessel dynamics for the attack-scenario environment.

Ported from Ship_envre/simulator.py (ShipAI vessel constants and force model)
with the following deliberate changes:

* The state is integrated in the body frame (Fossen form):
      x_dot   = u cos(psi) - v sin(psi)
      y_dot   = u sin(psi) + v cos(psi)
      psi_dot = r
      (M_RB + M_A) nu_dot = tau - (C_RB + D) nu
  The original rotated body accelerations straight into global accelerations
  and added the Coriolis matrices instead of subtracting them.
* Propeller advance ratio is the physical J = u / (n D), so throttle actually
  sets the steady speed (the original J did not depend on n).
* A yaw-rate damping moment N_r r + N_rr |r| r is added and the added-mass
  Coriolis matrix is dropped: ShipAI's sway-yaw loop has no yaw damping and
  diverges (the ship spins) after a few hundred seconds.
* The rudder is rate-limited (3 deg/s) so a level change is not instantaneous.
* Derivatives are plain float arrays; fixed-step RK4 with `substeps` per
  decision interval, no scipy dependency and no object-dtype arrays.
"""
import numpy as np


class ShipDynamics:
    def __init__(self, decision_interval=10.0, substeps=10, n_prop_max=1.8):
        self.dt_decision = float(decision_interval)   # seconds per RL step
        self.substeps = int(substeps)
        self.state = None                              # [x, y, psi, u, v, r]
        self.action = np.zeros(2)                      # [rudder_level, throttle]

        # Vessel constants (ShipAI)
        self.M = 115000 * 10**3
        self.Iz = 414000000 * 10**3
        self.M11 = 14840.4 * 10**3
        self.M22 = 174050 * 10**3
        self.M26 = 38369.6 * 10**3
        self.M66 = 364540000 * 10**3
        self.M62 = 36103 * 10**3
        self.D11 = 0.35370 * 10**3
        self.D22 = 1.74129 * 10**3
        self.D26 = 1.95949 * 10**3
        self.D62 = 1.85586 * 10**3
        self.D66 = 3.23266 * 10**3

        self.L = 244.74
        self.Draft = 15.3
        self.x_g = 2.2230
        self.x_prop = -112.0
        self.x_rudder = -115.0
        self.Cy = 0.06
        self.lp = 7.65
        self.Cb = 0.85
        self.B = 42.0
        self.S = 27342.0
        self.pho = 1.025 * 10**3
        self.mi = 10**-3
        self.A_rud = 80.0
        self.delta_x = self.x_prop - self.x_rudder
        self.r_aspect = 2.0
        self.D_prop = 7.2
        self.n_prop_max = float(n_prop_max)   # rps at throttle = 1

        self.rudder_max = np.pi / 6           # 30 degrees
        self.rudder_rate = np.radians(3.0)    # rad/s, typical steering-gear rate 2.3-3 deg/s
        self.rudder_actual = 0.0              # current rudder angle as a level in [-1, 1]
        self.Nr_prime = 0.002                 # linear yaw damping N'_r
        self.Nrr_prime = 0.02                 # quadratic yaw damping N'_rr

        # constant matrices
        self.Mrb = np.array([[self.M, 0, 0],
                             [0, self.M, self.M * self.x_g],
                             [0, self.M * self.x_g, self.Iz]])
        self.Ma = np.array([[self.M11, 0, 0],
                            [0, self.M22, self.M26],
                            [0, self.M62, self.M66]])
        self.Dl = np.array([[self.D11, 0, 0],
                            [0, self.D22, self.D26],
                            [0, self.D62, self.D66]])
        self.MM_inv = np.linalg.inv(self.Mrb + self.Ma)

    # ------------------------------------------------------------------ API
    def reset(self, x, y, psi, speed, r=0.0):
        self.state = np.array([x, y, psi, speed, 0.0, r], dtype=float)
        self.action = np.zeros(2)
        self.rudder_actual = 0.0
        return self.state.copy()

    def step(self, rudder_level, throttle):
        """rudder_level in [-1, 1] (-> +-30 deg), throttle in [0, 1]."""
        self.action = np.array([np.clip(rudder_level, -1, 1), np.clip(throttle, 0, 1)])
        h = self.dt_decision / self.substeps
        s = self.state
        max_move = self.rudder_rate * h / self.rudder_max          # level units per substep
        for _ in range(self.substeps):
            # rate-limited rudder: the commanded level is reached at <= rudder_rate
            delta = np.clip(self.action[0] - self.rudder_actual, -max_move, max_move)
            self.rudder_actual = float(np.clip(self.rudder_actual + delta, -1, 1))
            k1 = self._deriv(s)
            k2 = self._deriv(s + 0.5 * h * k1)
            k3 = self._deriv(s + 0.5 * h * k2)
            k4 = self._deriv(s + h * k3)
            s = s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        s[2] = (s[2] + np.pi) % (2 * np.pi) - np.pi
        self.state = s
        return self.state.copy()

    def global_velocity(self):
        x, y, psi, u, v, r = self.state
        return np.array([u * np.cos(psi) - v * np.sin(psi),
                         u * np.sin(psi) + v * np.cos(psi)])

    # ------------------------------------------------------------ physics
    def _deriv(self, s):
        x, y, psi, u, v, r = s
        beta = self.rudder_actual * self.rudder_max
        alpha = self.action[1]

        vc = np.hypot(u, v)
        gamma = np.pi + np.arctan2(v, u)

        # hull resistance (ShipAI)
        Re = self.pho * vc * self.L / self.mi
        if Re <= 1.0:
            C0 = 0.0
        else:
            C0 = 0.0094 * self.S / (self.Draft * self.L) / (np.log10(Re) - 2) ** 2
        C1 = C0 * np.cos(gamma) + (-np.cos(3 * gamma) + np.cos(gamma)) * np.pi * self.Draft / (8 * self.L)
        F1u = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C1
        C2 = ((self.Cy - 0.5 * np.pi * self.Draft / self.L) * np.sin(gamma) * np.abs(np.sin(gamma))
              + 0.5 * np.pi * self.Draft / self.L * np.sin(gamma) ** 3
              + np.pi * self.Draft / self.L * (1 + 0.4 * self.Cb * self.B / self.Draft)
              * np.sin(gamma) * np.abs(np.cos(gamma)))
        F1v = 0.5 * self.pho * vc ** 2 * self.L * self.Draft * C2
        C6 = -self.lp / self.L * self.Cy * np.sin(gamma) * np.abs(np.sin(gamma))
        C6 -= np.pi * self.Draft / self.L * np.sin(gamma) * np.cos(gamma)
        C6 -= ((0.5 + 0.5 * np.abs(np.cos(gamma))) ** 2 * np.pi * self.Draft / self.L
               * (0.5 - 2.4 * self.Draft / self.L) * np.sin(gamma) * np.abs(np.cos(gamma)))
        F1z = 0.5 * self.pho * vc ** 2 * self.L ** 2 * self.Draft * C6

        # propeller + rudder (ShipAI "complex" model, physical advance ratio)
        n = self.n_prop_max * alpha
        if n > 1e-6:
            J = max(u, 0.0) / (n * self.D_prop)
        else:
            J = 1.0
        kt = max(0.5 - 0.5 * J, 0.0)
        Fpx = kt * self.pho * n ** 2 * self.D_prop ** 4
        kr = 0.5 + 0.5 / (1 + 0.15 * self.delta_x / self.D_prop)
        ur = np.sqrt(u ** 2 + kr * 4 * kt * n ** 2 * self.D_prop ** 2 / np.pi)
        vr = -0.8 * v
        Ur2 = ur ** 2 + vr ** 2
        fa = 6.13 * self.r_aspect / (self.r_aspect + 2.25)
        FN = 0.5 * self.pho * self.A_rud * fa * Ur2 * np.sin(beta)
        Fpy = -FN * np.cos(beta)
        Fpz = -FN * np.cos(beta) * self.x_rudder

        # yaw-rate damping (absent in ShipAI, which makes the model course-unstable)
        U = max(abs(u), 1.0)
        N_r = -self.Nr_prime * 0.5 * self.pho * self.L ** 4 * U * r
        N_rr = -self.Nrr_prime * 0.5 * self.pho * self.L ** 5 * abs(r) * r

        tau = np.array([F1u + Fpx, F1v + Fpy, 0.21 * F1z + 0.5 * Fpz + N_r + N_rr])

        Crb = np.array([[0, 0, -self.M * (self.x_g * r + v)],
                        [0, 0, self.M * u],
                        [self.M * (self.x_g * r + v), -self.M * u, 0]])
        # The added-mass Coriolis matrix C_A (Munk moment) is left out, as in
        # ShipAI: the empirical cross-flow hull model already carries the
        # drift-induced yaw moment and adding C_A makes the sway-yaw loop unstable.
        nu = np.array([u, v, r])
        nu_dot = self.MM_inv @ (tau - (Crb + self.Dl) @ nu)

        return np.array([u * np.cos(psi) - v * np.sin(psi),
                         u * np.sin(psi) + v * np.cos(psi),
                         r,
                         nu_dot[0], nu_dot[1], nu_dot[2]])


if __name__ == "__main__":
    # quick sanity run: straight line, then hard turn
    dyn = ShipDynamics()
    dyn.reset(0, 0, 0, 5.0)
    for k in range(60):
        s = dyn.step(0.0, 1.0)
    print("after 600 s full throttle straight: speed %.2f m/s, x %.0f m" % (s[3], s[0]))
    for k in range(60):
        s = dyn.step(1.0, 1.0)
    print("after 600 s hard rudder: psi %.2f rad, r %.4f rad/s, v %.2f m/s" % (s[2], s[5], s[4]))
