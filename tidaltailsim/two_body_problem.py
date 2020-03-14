import numpy as np
from scipy.integrate import solve_ivp


class TwoBodyProblem:
    """
    Solver and plotter for the general two-body problem in gravitational potential
    e.g. binary star, two cores of colliding galaxies
    """

    def __init__(self, r0, r0d=0, E=0, J=None, G=1, M1=1, M2=1, use_reduced_mass=True):
        """
        Initialise the two-body problem from the given initial condition

        Parameters:
        r0  -- Initial distance between the bodies
        r0d -- Initial radial velocity between the bodies (at t=0)
        (set r0d=0 if r0 is the extremum distance)
        E   -- Total energy of the problem
        J   -- Total angular momentum of the problem
        G   -- Gravitational constant
        M1  -- The mass of the first body
        M2  -- The mass of the second body
        use_reduced_mass  -- should the Hamiltonian be written with reduced mass approached

        If J is None, it will be automatically computed from other parameters,
        otherwise, J will be used to infer r0d (with the same sign as the passed r0d).

        If use_reduced_mass is False, it will assume that M1 >> M2, ie, reduced mass = M2
        """

        self._r0 = r0
        self._G = G
        self._use_reduced_mass = use_reduced_mass

        if use_reduced_mass:
            rM = TwoBodyProblem.calculate_reduced_mass((M1, M2))
        else:
            rM = M2

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

        self._integration_result_future = None
        self._integration_result_past = None

    @property
    def initial_distance(self):
        return self._r0

    @property
    def G(self):
        return self._G

    @property
    def use_reduced_mass(self):
        return self._use_reduced_mass

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
        to be used in scipy.integrate.solve_ivp

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

    def _get_initial_phase(self):
        return np.array([
            self._r0,   # initial distance
            self._Pr0,  # initial radial momentum
            0           # initial angle
        ])

    def solve_two_body_problem(self, t_end, pos_events=None, neg_events=None):

        # Solve the initial value problem to the future
        self._integration_result_future = \
            solve_ivp(fun=self._hamilton_eqm,
                      t_span=(0, t_end),  # future
                      y0=self._get_initial_phase(),
                      method='RK45',
                      dense_output=True,
                      events=pos_events,
                      vectorized=True)

        # Solve the initial value problem to the past,
        # only if initial radial momentum is non-zero,
        # i.e., the solution is not symmetric for past and present
        if self._Pr0 != 0:
            self._integration_result_past = \
                solve_ivp(fun=self._hamilton_eqm,
                          t_span=(0, -t_end),  # past
                          y0=self._get_initial_phase(),
                          method='RK45',
                          dense_output=True,
                          events=pos_events,
                          vectorized=True)

        self._process_integration_result()

        return (self._integration_result_future, self._integration_result_past)

    def _process_integration_result(self):
        # join past and future solution
        if self._integration_result_past is not None:
            self.__t = np.concatenate((self._integration_result_past.t[:0:-1], self._integration_result_future.t))
            self.__phase = np.concatenate((self._integration_result_past.y[:, :0:-1], self._integration_result_future.y), axis=1)
        else:
            self.__t = np.concatenate((-1 * self._integration_result_future.t[:0:-1], self._integration_result_future.t))
            past_y = self._integration_result_future.y[:, :0:-1]
            self.__phase = np.concatenate((past_y, self._integration_result_future.y), axis=1)

        # calculate coordinates with respect to the centre of mass of the system
        if self._use_reduced_mass:
            self.__r1 = (-self._reduced_mass / self.M1) * self.__phase[:, 0]
            self.__r2 = (+self._reduced_mass / self.M2) * self.__phase[:, 0]
        else:
            self.__r1 = np.zeros(self.__t.shape)
            self.__r2 = np.copy(self.__phase[:, 0])

        self.__angle = self.__phase[2, :]

        # convert polar coordinates to cartesian coordinates
        self.__x1 = self.__r1 * np.cos(self.__angle)
        self.__y1 = self.__r1 * np.sin(self.__angle)
        self.__x2 = self.__r2 * np.cos(self.__angle)
        self.__y2 = self.__r2 * np.sin(self.__angle)

    def evaluate_dense_phase_at(self, time):
        """
        evaluate dense Hamilton phase in polar coordinates
        time must be 1d array
        """
        # evaluate dense solution for t<0 and t>=0 separately, and join them back together at the end
        is_past = time < 0
        is_future = time >= 0

        time_indices = np.indices(time.shape)

        past_indices = time_indices[is_past]
        future_indices = time_indices[is_future]

        past_time = time[is_past]
        future_time = time[is_future]

        if self._integration_result_past is not None:
            past_phase = self._integration_result_past.sol(past_time)
        else:
            past_phase = self._integration_result_future.sol(-past_time)

        future_phase = self._integration_result_future.sol(future_time)

        time_indices_join = np.concatenate((past_indices, future_indices), axis=0)
        phase_join = np.concatenate((past_phase, future_phase), axis=1)

        rank = np.argsort(time_indices_join)
        phase = phase_join[rank]

        return phase

    def evaluate_dense_polar_coords_at(self, time):
        """
        evaluate dense solution in polar coordinates
        time must be 1d array
        """
        phase = self.evaluate_dense_phase_at(time)
        if self._use_reduced_mass:
            r1 = (-self._reduced_mass / self.M1) * phase[0, :]
            r2 = (+self._reduced_mass / self.M2) * phase[0, :]
        else:
            r1 = np.zeros(self.__t.shape)
            r2 = phase[0, :]

        angle = phase[2, :]

        return (r1, r2, angle)

    def evaluate_dense_cartesian_solution_at(self, time):
        """
        evaluate dense solution in cartesian coordinates
        time must be 1d array
        """
        r1, r2, angle = self.evaluate_dense_polar_coords_at(time)
        x1 = r1 * np.cos(angle)
        y1 = r1 * np.sin(angle)
        x2 = r2 * np.cos(angle)
        y2 = r2 * np.sin(angle)
        return ((x1, y1), (x2, y2))
