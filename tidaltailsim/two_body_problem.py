import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


class TwoBodyProblem:
    """
    Solver and plotter for the general gravitational two-body problem in the centre-of-mass frame
    e.g. binary star system, two cores of colliding galaxies

    Remark:
    This class use the polar coordinate Hamiltonian given by,
    Hamiltonian = (1/2) * Pr^2 / rM
                + (1/2) * J^2 / (rM * r^2)
                + gravi_potential(r),
    where gravi_potential(r) = - G * M1 * M2 / r

    r is the radial coordinate between the two bodies.
    Pr is the radial conjugate momentum.
    J is the angular conjugate momentum.
    rM is the reduced mass parameter.
    G is the Gravitational constant.
    M1 and M2 are the masses of the two bodies.
    """

    def __init__(self, r0, r0d=0., E=0., J=None, G=1., M1=1., M2=1., use_reduced_mass=True):
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
            J = np.sqrt((E - self.gravi_potential_energy(r0)) * 2 * rM * r0**2 - Pr0**2 * r0**2)
            if np.isnan(J):
                raise ValueError('Specified parameters give unphysical angular momentum')
        else:
            Pr0 = np.sqrt((E - self.gravi_potential_energy(r0)) * 2 * rM - J**2 / r0**2)
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
        return 1 / np.sum(1 / Ms)

    def gravi_potential_energy(self, r):
        """Calculate the gravitational potential between the two bodies"""
        return (- self._G * self._M1 * self._M2) / r

    def deriv_gravi_potential_energy(self, r):
        """Calculate first derivative of gravitational potential between the two bodies"""
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

        Remark:
        This class use the Hamiltonian given by,
        Hamiltonian = (1/2) * Pr^2 / rM
                    + (1/2) * J^2 / (rM * r^2)
                    + gravi_potential(r),
        where gravi_potential(r) = - G * M1 * M2 / r
        """
        r = phase[0, ...]
        Pr = phase[1, ...]
        # angle = phase[..., 2]

        phase_d = np.zeros(phase.shape)
        phase_d[0, ...] = Pr / self._reduced_mass
        phase_d[1, ...] = (self._J**2 / self._reduced_mass) * r**(-3) - self.deriv_gravi_potential_energy(r)
        phase_d[2, ...] = (self._J / self._reduced_mass) * r**(-2)

        return phase_d

    def _get_initial_phase(self):
        return np.array([
            self._r0,   # initial distance
            self._Pr0,  # initial radial momentum
            0           # initial angle
        ])

    def solve_problem(self, t_end, sampling_points=1000, future_events=None, past_events=None):
        """
        Solve the problem both to the past and to the future (w.r.t t=0)
        ie, from t=-t_end to t=+t_end

        Return:
        a tuple of (future, past) scipy.integrate.OdeSolution objects
        obtained from executing scipy.integrate.solve_ivp
        past is None if the solution is symmetric

        The solution signature is such that each dimension of y corresponds to
        0: radial coordinate between the two-body
        1: radial conjugate momentum
        2: angular coordinate
        (in polar-coordinate)
        """

        if t_end <= 0:
            raise ValueError('t_end must be positive')
        # Solve the initial value problem to the future
        self._integration_result_future = \
            solve_ivp(fun=self._hamilton_eqm,
                      t_span=(0, t_end),  # future
                      t_eval=np.linspace(0, t_end, sampling_points, endpoint=False),
                      y0=self._get_initial_phase(),
                      method='RK45',
                      dense_output=True,
                      events=future_events,
                      vectorized=True)

        # print(self._integration_result_future.t[[0, -1]])
        # print(self._integration_result_future.y[..., [0, -1]])

        # Solve the initial value problem to the past,
        # only if initial radial momentum is non-zero,
        # i.e., the solution is not symmetric for past and present
        # (This helps decrrease execution time)
        if self._Pr0 != 0:
            self._integration_result_past = \
                solve_ivp(fun=self._hamilton_eqm,
                          t_span=(0, -t_end),  # past
                          t_eval=np.linspace(0, -t_end, sampling_points, endpoint=False),
                          y0=self._get_initial_phase(),
                          method='RK45',
                          dense_output=True,
                          events=past_events,
                          vectorized=True)

        self._process_integration_result()

        self._t_end = t_end
        self._sampling_points = 1000

        return (self._integration_result_future, self._integration_result_past)

    def _process_integration_result(self):
        # join past and future solution
        if self._integration_result_past is not None:
            self.__t = np.concatenate((self._integration_result_past.t[:0:-1], self._integration_result_future.t))
            self.__phase = np.concatenate((self._integration_result_past.y[:, :0:-1], self._integration_result_future.y), axis=1)
        else:
            self.__t = np.concatenate((-1 * self._integration_result_future.t[:0:-1], self._integration_result_future.t))
            past_y = np.copy(self._integration_result_future.y[:, :0:-1])
            past_y[1, :] *= -1  # inverse the momentum
            past_y[2, :] *= -1  # inverse the angle
            self.__phase = np.concatenate((past_y, self._integration_result_future.y), axis=1)

        # print(self.__t, self.__phase)

        self.__angle = self.__phase[2, :]

        # convert the momentum into cartesian component from Pr and J
        self.__Px = self.__phase[1, :] * np.cos(self.__angle) - self._J / self.__phase[0, :] * np.sin(self.__angle)
        self.__Py = self.__phase[1, :] * np.sin(self.__angle) + self._J / self.__phase[0, :] * np.cos(self.__angle)

        # calculate coordinates with respect to the centre of mass of the system
        if self._use_reduced_mass:
            self.__r1 = (-self._reduced_mass / self._M1) * self.__phase[0, :]
            self.__r2 = (+self._reduced_mass / self._M2) * self.__phase[0, :]

            self.__vx1 = -self.__Px / self._M1
            self.__vy1 = -self.__Py / self._M1
        else:
            self.__r1 = np.zeros(self.__t.shape)
            self.__r2 = np.copy(self.__phase[0, :])

            self.__vx1 = np.zeros(self.__t.shape)
            self.__vy1 = np.zeros(self.__t.shape)

        self.__vy2 = +self.__Py / self._M2
        self.__vx2 = +self.__Px / self._M2

        # convert polar coordinates to cartesian coordinates
        self.__x1 = self.__r1 * np.cos(self.__angle)
        self.__y1 = self.__r1 * np.sin(self.__angle)
        self.__x2 = self.__r2 * np.cos(self.__angle)
        self.__y2 = self.__r2 * np.sin(self.__angle)

    def evaluate_dense_phase_at(self, time):
        """
        evaluate dense Hamilton phase in polar coordinates
        time must be 0d or 1d array
        """
        # evaluate dense solution for t<0 and t>=0 separately, and join them back together at the end
        is_past = time < 0
        is_future = time >= 0

        # different behaviour depending on whether time is 0d array or 1d array
        if np.ndim(time) == 0:
            if is_past:
                if self._integration_result_past is not None:
                    phase = self._integration_result_past.sol(time)
                else:
                    phase = self._integration_result_future.sol(-time)
            else:
                phase = self._integration_result_future.sol(time)
        elif np.ndim(time) == 1:
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
        else:
            raise ValueError('time must be either 0d or 1d array')

        return phase

    def evaluate_dense_polar_coords_at(self, time):
        """
        evaluate dense solution in polar coordinates
        time must be 1d array
        """
        phase = self.evaluate_dense_phase_at(time)
        if self._use_reduced_mass:
            r1 = (-self._reduced_mass / self._M1) * phase[0]
            r2 = (+self._reduced_mass / self._M2) * phase[0]
        else:
            r1 = np.zeros(self.__t.shape)
            r2 = phase[0]

        angle = phase[2]

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

    def plot_two_body_paths(self, axes, zdir=None, plot_v0=None):

        # if zdir is supplied assume that axes is an axes3D instance
        if zdir is None:
            axes.plot(self.__x1, self.__y1, color='royalblue')
            axes.plot(self.__x2, self.__y2, color='darkorange')
        else:
            axes.plot(self.__x1, self.__y1, zdir=zdir, color='royalblue')
            axes.plot(self.__x2, self.__y2, zdir=zdir, color='darkorange')

        # plot the arrows showing the velocity at given index
        if len(plot_v0) == 2:
            factor = plot_v0[0]
            indices = plot_v0[1]
            for i in indices:
                axes.arrow(self.__x1[i], self.__y1[i], factor * self.__vx1[i], factor * self.__vy1[i], width=.05, color='lightsteelblue')
                axes.arrow(self.__x2[i], self.__y2[i], factor * self.__vx2[i], factor * self.__vy2[i], width=.05, color='burlywood')

    def _prepare_animating_object(self, axes):
        line2body1, = axes.plot(self.__x1[0], self.__y1[0], '.', color='navy', markersize=5.0)
        line2body2, = axes.plot(self.__x2[0], self.__y2[0], '.', color='maroon', markersize=5.0)
        return [line2body1, line2body2]

    def _animation_func(self, frame_index, *animating_artists):
        line2body1 = animating_artists[0]
        line2body2 = animating_artists[1]

        line2body1.set_data([self.__x1[frame_index], self.__y1[frame_index]])
        line2body2.set_data([self.__x2[frame_index], self.__y2[frame_index]])

        return [line2body1, line2body2]

    def animate(self, figure, axes, rate=1.0, framerate=None):

        # if framrate is not given, all frames get rendered (potentially impacting the performance)
        if framerate:
            framestep = int(round(self._sampling_points * rate / (framerate * self._t_end)))
            frames = range(0, self.__t.shape[0], framestep)
            interval = int(round(framestep * 1000 * self._t_end / self._sampling_points / rate))
        else:
            interval = int(round(1000 * self._t_end / self._sampling_points / rate))
            frames = self.__t.shape[0]

        animating_artists = self._prepare_animating_object(axes)

        animation = \
            FuncAnimation(figure,
                          blit=True,
                          frames=frames,
                          interval=interval,
                          fargs=animating_artists,
                          func=self._animation_func)

        return animation
