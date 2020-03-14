import numpy as np
from scipy.integrate import solve_ivp


class TwoBodyProblem:

    def __init__(self, r0, r0d=0, E=0, J=None, G=1, M1=1, M2=1, rM=None):
        """
        Initialise the two-body problem from the given initial condition

        Parameters:
        r0  -- Initial distance between the bodies
        r0d -- Initial velocity between the bodies
        E   -- Total energy of the problem
        J   -- Total angular momentum of the problem
        G   -- Gravitational constant
        M1  -- The mass of the first body
        M2  -- The mass of the second body
        rM  -- The reduced mass of the two-body system

        If J is None, it will be automatically computed from other parameters,
        otherwise, J will be used to infer r0d (with the same sign as the passed r0d).

        If rM is None, it will be automaticaally determined from M1 and M2
        """

        self._r0 = r0
        self._G = G

        if rM is None:
            rM = TwoBodyProblem.calculate_reduced_mass((M1, M2))

        self._reduced_mass = rM
        self._M1 = M1
        self._M2 = M2

        if J is None:
            Pr0 = rM * r0d
            J = np.sqrt((E - self.gravi_potential(r0)) * 2 * rM * r0**2 - Pr0**2 * r0**2)
            if np.isnan(J):
                raise ValueError('Specified parameters give unphysical angular momentum')
        else:
            Pr0 = np.sqrt((E - self.gravi_potential(r0)) * 2 * rM - J**2 / r0**2)
            if np.isnan(Pr0):
                raise ValueError('Specified parameters give unphysical initial radial momentum')
            if r0d < 0:
                Pr0 *= -1

        self._Pr0 = Pr0  # Pr0 is the initial radial momentum
        self._J = J

        self._solution = None

    @property
    def initial_distance(self):
        return self._r0

    @property
    def G(self):
        return self._G

    @property
    def reduced_mass(self):
        return self._reduced_mass

    @property
    def Mass1(self):
        return self._M1

    @property
    def Mass2(self):
        return self._M2

    @property
    def initial_radial_momentum(self):
        return self._Pr0  # Pr0 is the initial radial momentum

    @property
    def angular_momentum(self):
        return self._J

    @staticmethod
    def calculate_reduced_mass(Ms):
        Ms = np.array(Ms)
        return np.sum(Ms**(-1))**(-1)

    def gravi_potential(self, r):
        return (- self._G * self._M1 * self._M2) / r

    def deriv_gravi_potential(self, r):
        """Calculate first derivative of gravitational potential"""
        return (self._G * self._M1 * self._M2) / r**2

    def _hamilton_eqm(self, t, phase):
        """
        Calculate system of equations of motion from Hamilton equations
        from the given vectorised points in the phase space

        where
        phase[0] is r radius
        phase[1] is radial conjugate momentum
        phase[2] is phi angle

        angular conjugate momentum is omitted since it is conserved
        """
        r = phase[..., 0]
        Pr = phase[..., 1]
        # angle = phase[..., 2]

        phase_d = np.zeros(phase.shape)
        phase_d[..., 0] = Pr / self._reduced_mass
        phase_d[..., 1] = (-self._J**2 / self._reduced_mass) * r**(-3) - self.deriv_gravi_potential(r)
        phase_d[..., 2] = (self._J / self._reduced_mass) * r**(-2)

        return phase_d
