import numpy as np
from math import sqrt, pi, exp, atan, sin, sinh, asinh, floor
from enum import Enum
from scipy.integrate import quad
from scipy.special import factorial



def norm2(v):
    return np.dot(v, v)

def norm(v):
    return sqrt(norm2(v))

def dot(a, b):
    return np.dot(a, b)



class LuscherZeta:
    """
    Python translation of the C++ LuscherZeta class.
    """

    def __init__(self, l, d):
        """
        l : array-like of ints (0 or 1), twists
        d : array-like of floats, P_CM / (2pi/L)
        """
        self.l = np.array(l, dtype=float)
        self.d = np.array(d, dtype=float)

        self.N = 5
        self.epsabs = 1e-6
        self.epsrel = 1e-6


    @staticmethod
    def _zeta_func(t, q2, gn2):
        pi2 = pi * pi
        return exp(t * q2 - pi2 * gn2 / t) * (pi / t) ** 1.5

    def _int_zeta(self, q2, gn2):
        """
        Integral from t = 0 to 1
        """
        result, _ = quad(
            self._zeta_func,
            1e-12,
            1.0,
            args=(q2, gn2),
            epsabs=self.epsabs,
            epsrel=self.epsrel,
            limit=200,
        )
        return result



    def _gamma_op_gen(self, n, gamma, inv=False):
        dnorm = norm(self.d)
        if dnorm == 0.0:
            dunit = np.zeros(3)
        else:
            dunit = self.d / dnorm

        npar = dot(n, dunit) * dunit
        nperp = n - npar

        factor = (1.0 / gamma) if inv else gamma
        return factor * npar + nperp

    def gamma_op(self, n, gamma):
        return self._gamma_op_gen(n, gamma, inv=False)

    def inv_gamma_op(self, n, gamma):
        return self._gamma_op_gen(n, gamma, inv=True)


    def _z3sum(self, q, n, gamma):
        r = n + self.d / 2.0 + self.l / 2.0
        r = self.inv_gamma_op(r, gamma)

        r2 = norm2(r)
        q2 = q * q

        out = exp(q2 - r2) / (r2 - q2)

        # Skip integral for zero vector
        if np.allclose(n, 0.0):
            return out

        h = self.gamma_op(n, gamma)
        h2 = norm2(h)

        int_n = self._int_zeta(q2, h2)

        phase = (-1) ** int(dot(n, self.d + self.l))
        out += gamma * int_n * phase

        return out


    def set_integration_error_bounds(self, eps_abs, eps_rel):
        self.epsabs = eps_abs
        self.epsrel = eps_rel

    def set_maximum_vector_magnitude(self, N):
        self.N = N

    def calc_zeta00(self, q, gamma):
        result = 0.0

        for nx in range(-self.N, self.N + 1):
            Ny = int(floor(sqrt(self.N * self.N - nx * nx)))
            for ny in range(-Ny, Ny + 1):
                Nz = int(
                    floor(
                        sqrt(self.N * self.N - nx * nx - ny * ny) + 0.5
                    )
                )
                for nz in range(-Nz, Nz + 1):
                    n = np.array([nx, ny, nz], dtype=float)
                    result += self._z3sum(q, n, gamma)

        # Constant part
        q2 = q * q
        const_part = 0.0
        warn = True

        for l in range(100):
            c_aux = (q2 ** l) / factorial(l) / (l - 0.5)
            if l > 9 and abs(c_aux) < 1e-8 * abs(const_part):
                warn = False
                break
            const_part += c_aux

        if warn:
            print(
                "LuscherZeta warning: reached maximum loop number in constant part"
            )

        result += gamma * (pi ** 1.5) * const_part
        result /= sqrt(4 * pi)

        return result

    def calc_phi(self, q, gamma=1.0):
        return atan(-gamma * q * (pi ** 1.5) / self.calc_zeta00(q, gamma))

    def calc_phi_deriv(self, q, frac_shift=1e-4):
        dq = frac_shift * q
        return (self.calc_phi(q + dq) - self.calc_phi(q - dq)) / (2 * dq)

    def getd(self):
        return self.d

    def getl(self):
        return self.l


# -----------------------------
# Dispersion relations
# -----------------------------

class DispersionRelation(Enum):
    Continuum = 0
    Sin = 1
    SinhSin = 2
    Fit = 3


def get_sin_p2(p):
    return sum((2 * sin(pi / 2)) ** 2 for pi in p)


def get_pn(p, n):
    return sum(pi ** n for pi in p)


def dispersion_energy(disp, m, p, punit=1.0):
    p = np.array(p) * punit

    if disp == DispersionRelation.Continuum:
        return sqrt(m * m + norm2(p))

    if disp == DispersionRelation.Sin:
        return sqrt(m * m + get_sin_p2(p))

    if disp == DispersionRelation.SinhSin:
        h = get_sin_p2(p) + m * m
        return 2 * asinh(sqrt(h / 4.0))

    if disp == DispersionRelation.Fit:
        return sqrt(m * m + 0.995467 * get_pn(p, 2) - 0.144335 * get_pn(p, 4))

    raise ValueError("Unknown dispersion relation")


def dispersion_mass(disp, E, p, punit=1.0):
    p = np.array(p) * punit

    if disp == DispersionRelation.Continuum:
        return sqrt(E * E - norm2(p))

    if disp == DispersionRelation.Sin:
        return sqrt(E * E - get_sin_p2(p))

    if disp == DispersionRelation.SinhSin:
        m2 = (2 * sinh(E / 2)) ** 2 - get_sin_p2(p)
        return sqrt(m2)

    if disp == DispersionRelation.Fit:
        return sqrt(E * E - 0.995467 * get_pn(p, 2) + 0.144335 * get_pn(p, 4))

    raise ValueError("Unknown dispersion relation")


# -----------------------------
# Phase shift
# -----------------------------

def phase_shift_zeta(
    zeta,
    E,
    m,
    L,
    disp=DispersionRelation.Continuum,
    zero_tol=1e-10,
):
    d = zeta.getd()

    E_CM = dispersion_energy(
        disp, E, np.zeros(3), punit=2 * pi / L
    )

    gamma = E / E_CM

    k2 = (E_CM / 2) ** 2 - m * m
    if abs(k2) < zero_tol:
        k2 = 0.0

    k = sqrt(k2)
    q = k * L / (2 * pi)

    delta = -zeta.calc_phi(q, gamma)

    while delta > pi:
        delta -= pi
    while delta < -pi:
        delta += pi

    return delta / pi * 180.0


def phase_shift(
    E,
    m,
    L,
    twists=(0, 0, 0),
    d=(0.0, 0.0, 0.0),
    disp=DispersionRelation.Continuum,
):
    zeta = LuscherZeta(twists, d)
    return phase_shift_zeta(zeta, E, m, L, disp)

import numpy as np

# Physical inputs (example values)
E = 0.347884         # two-pion energy
m = 0.103246         # pion mass
L = 32.0          # spatial lattice size

# Boundary conditions and CM momentum
twists = (1, 1, 1)          # periodic
d = (0.0, 0.0, 0.0)         # P_CM = 0

# Compute phase shift in degrees
delta = phase_shift(
    E=E,
    m=m,
    L=L,
    twists=twists,
    d=d,
    disp=DispersionRelation.Continuum
)

print(f"Phase shift = {delta:.3f} degrees")
