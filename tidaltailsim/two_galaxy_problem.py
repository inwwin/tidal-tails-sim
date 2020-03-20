import numpy as np
from scipy.integrate import solve_ivp
from tidaltailsim.two_body_problem import TwoBodyProblem
from matplotlib.colors import hsv_to_rgb


class TwoGalaxyProblem(TwoBodyProblem):
    """
    Solver and Plotter for the simplified two-galaxy collision problem
    with the massive cores' trajectory inherited from TwoBodyProblem
    and stars modelled as 'massless' particles orbiting the massive cores
    (massless in the sense that they don't cause perturbation
    to the gravitational field of the system)
    """

    def __init__(self, r0, r0d=0, E=0, J=None, G=1.0, M1=1.0, M2=1.0, use_reduced_mass=True):
        super().__init__(r0, r0d, E, J, G, M1, M2, use_reduced_mass)

        # defaulting that masses of the massive bodies/cores, that the 'massless'
        # particles feel, to be the original masses in the two-body problem
        self.M1_feel = self.Mass1
        self.M2_feel = self.Mass2

        self._G_feel = self.G

        self._galaxy1_initial_condition = []
        self._galaxy2_initial_condition = []

        self.verbose = True
        self.suppress_error = False

    @property
    def M1_feel(self):
        """
        The mass of body/core 1 that the particles will *feel*
        (set to zero, if want the particles not to feel this body/core)
        """
        return self._M1_feel

    @M1_feel.setter
    def M1_feel(self, value):
        if isinstance(value, float):
            self._M1_feel = value
        else:
            raise TypeError('A mass must be floating number')

    @property
    def M2_feel(self):
        """
        The mass of body/core 2 that the particles will *feel*
        (set to zero, if want the particles not to feel this body/core)
        """
        return self._M2_feel

    @M2_feel.setter
    def M2_feel(self, value):
        if isinstance(value, float):
            self._M2_feel = value
        else:
            raise TypeError('A mass must be floating number')

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        if isinstance(value, bool):
            self._verbose = value
        else:
            raise TypeError('verbose must be a boolean')

    @property
    def suppress_error(self):
        return self._suppress_error

    @suppress_error.setter
    def suppress_error(self, value):
        if isinstance(value, bool):
            self._suppress_error = value
        else:
            raise TypeError('suppress_error must be a boolean')

    def configure_galaxy(self, galaxy_index, orbital_radius, orbital_particles, theta=0., phi=0., m=None):
        """
        Set up the initial conditions of the collection of particles
        orbiting around the core of the galaxy

        Parameters:
        galaxy_index        -- a number 1 or 2 specifying which galaxy is to be configured
        orbital_radius      -- 1d array (n,) of the radius of each orbit
        orbital_particles   -- integer 1d array (n,) of the number of particles
                               in each orbit
        theta, phi          -- (optional) the spherical polar coordinates of the axis of
                               the orbital plane with respect to the z-axis
                               (z-axis is the normal to the plane of two-body-problem)
        m                   -- (optional) customise the mass of the core used for calculating the initial condition
        """
        orbital_particles = np.int_(orbital_particles)

        if not (galaxy_index == 1 or galaxy_index == 2):
            raise ValueError('Expect either 1 or 2 in galaxy_index')
        else:
            is_g_1 = galaxy_index == 1  # True for galaxy 1 False for galaxy 2

        if not isinstance(m, float):
            m = self.M1_feel if is_g_1 else self.M2_feel

        if is_g_1:
            if not hasattr(self, '_vx1') or not hasattr(self, '_vy1') or not hasattr(self, '_x1') or not hasattr(self, '_y1'):
                raise UserWarning('Initial phase of body 1 doesn\'t exist. Please call solve_two_body_problem() first. ' +
                                  '(Defaulting the initial phase of body 1 to a zero speed at the origin')
                initial_velocity_body = np.zeros((3))
                initial_position_body = np.zeros((3))
            else:
                initial_velocity_body = np.array([self._vx1[0], self._vy1[0], 0])
                initial_position_body = np.array([self._x1[0], self._y1[0], 0])
        else:
            if not hasattr(self, '_vx2') or not hasattr(self, '_vy2') or not hasattr(self, '_x2') or not hasattr(self, '_y2'):
                raise UserWarning('Initial phase of body 2 doesn\'t exist. Please call solve_two_body_problem() first. ' +
                                  '(Defaulting the initial phase of body 2 to a zero speed at the origin')
                initial_velocity_body = np.zeros((3))
                initial_position_body = np.zeros((3))
            else:
                initial_velocity_body = np.array([self._vx2[0], self._vy2[0], 0])
                initial_position_body = np.array([self._x2[0], self._y2[0], 0])

        # create rotation matrix
        R1 = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        R2 = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi), np.cos(phi), 0],
                       [0, 0, 1]])
        R = np.matmul(R2, R1)

        galaxy_initial_condition = []

        for (radius, particles) in np.nditer([orbital_radius, orbital_particles]):
            angular_space = np.linspace(0, 2 * np.pi, particles, endpoint=False)

            # first initialise the componenets in x-y plane in the core's frame
            initial_condition = np.zeros((particles, 6))

            initial_condition[:, 0] = radius * np.cos(angular_space)
            initial_condition[:, 1] = radius * np.sin(angular_space)
            # initial_condition[:, 2] = np.zeros

            velocity = np.sqrt(self._G_feel * m / radius)
            initial_condition[:, 3] = -velocity * np.sin(angular_space)
            initial_condition[:, 4] = velocity * np.cos(angular_space)

            # rotate the vectors
            initial_condition[:, :3] = np.matmul(R, initial_condition[:, :3].T).T
            initial_condition[:, 3:] = np.matmul(R, initial_condition[:, 3:].T).T

            # galilean transform the initial velocity to the problem frame, by adding the velocity of core
            initial_condition[:, 3:] += initial_velocity_body  # broadcasting

            # displace the origin to the location of the core, by adding the position vector of core
            initial_condition[:, :3] += initial_position_body  # broadcasting

            galaxy_initial_condition.append(initial_condition)

        # TwoGalaxyProblem._galaxy#_initial_condition is a list of (n, 6) arrays.
        # Each (n, 6) array corresponds to each orbit,
        # where n is the number of particle in each orbit.
        # The 0, 1, 2 dimension of the last axis is the x, y, z component of each particle.
        # The 3, 4, 5 dimension of the last axis is the conjugate momentum of the x, y, z component of each particle.
        if is_g_1:
            self._galaxy1_initial_condition = galaxy_initial_condition
        else:
            self._galaxy2_initial_condition = galaxy_initial_condition

    def plot_galaxies_initial_positions(self, axes, zdir='z'):
        for ic in self._galaxy1_initial_condition:
            if hasattr(axes, 'plot3D'):
                axes.plot3D(ic[:, 0], ic[:, 1], ic[:, 2], '.', zdir=zdir, color='skyblue')
            else:
                axes.plot(ic[:, 0], ic[:, 1], '.', color='skyblue')
        for ic in self._galaxy2_initial_condition:
            if hasattr(axes, 'plot3D'):
                axes.plot3D(ic[:, 0], ic[:, 1], ic[:, 2], '.', zdir=zdir, color='salmon')
            else:
                axes.plot(ic[:, 0], ic[:, 1], '.', color='salmon')

    def grad_gravi_potential_feel(self, core_mass, core_position, evaluating_position):
        """
        Calculate the gradient of the gravitational potential produced by each galaxy
        at the given positions, from, e.g.,

        dVi/dx = + G * Mi * { (x-xp)^2 + (y-yp)^2 + (z-zp)^2 }^(-3/2) * (x-xp)
        dVi/dy = + G * Mi * { (x-xp)^2 + (y-yp)^2 + (z-zp)^2 }^(-3/2) * (y-yp)
        dVi/dz = + G * Mi * { (x-xp)^2 + (y-yp)^2 + (z-zp)^2 }^(-3/2) * (z-zp)
                                               ^
                           | this is the posi_diff_powered variable  |

        in the vectorised manner.
        """

        GM = self._G_feel * core_mass

        positional_difference = evaluating_position - core_position
        posi_diff_powered = np.sum(positional_difference**2, axis=0)**(-1.5)

        grad_potential = GM * posi_diff_powered * positional_difference  # broadcast
        return grad_potential

    def _unit_test_mass_hamilton_eqm(self, t, state):
        if np.ndim(t) == 0:
            t = np.array([t])
        xyz_core1, xyz_core2 = self.evaluate_dense_cartesian_solution_at(t)
        state_d = np.zeros(state.shape)
        state_d[:3, ...] = state[3:, ...]
        state_d[3:, ...] = - self.grad_gravi_potential_feel(self.M1_feel, xyz_core1, state[:3, ...])
        state_d[3:, ...] -= self.grad_gravi_potential_feel(self.M2_feel, xyz_core2, state[:3, ...])

        return state_d

    def solve_two_galaxy_problem(self, **kwargs):
        if not hasattr(self, '_t'):
            raise Exception("No time domain exists. Please call solve_two_body_problem first.")

        for galaxy_index, ics in [(1, self._galaxy1_initial_condition), (2, self._galaxy2_initial_condition)]:
            galaxy_result = []
            galaxy_states = []
            for orbital, orbit_index in zip(ics, range(len(ics))):
                orbital_result = []
                orbital_states = np.zeros((orbital.shape[0], 6, self._t.shape[0]), order='F')
                for i in range(orbital.shape[0]):
                    particle_result = solve_ivp(fun=self._unit_test_mass_hamilton_eqm,
                                                t_span=(self._t[0], self._t[-1]),
                                                t_eval=self._t,
                                                y0=orbital[i, :],
                                                method='RK23',
                                                dense_output=False,
                                                vectorized=True,
                                                **kwargs)
                    if particle_result.success:
                        orbital_states[i, ...] = particle_result.y
                        if self.verbose:
                            print("particle#{0:d} orbit#{1:d} galaxy#{3:d} success.".format(i, orbit_index, particle_result.message, galaxy_index))
                    else:
                        if not self.suppress_error:
                            print("particle#{0:d} orbit#{1:d} galaxy#{3:d} error.\nMessage: {2}".format(i, orbit_index, particle_result.message, galaxy_index))

                    orbital_result.append(particle_result)
                galaxy_result.append(orbital_result)
                galaxy_states.append(orbital_states)

            # save the result to attr _galaxy1_result or _galaxy1_result depending on the galaxy_index
            setattr(self, '_galaxy{0:d}_result'.format(galaxy_index), galaxy_result)
            setattr(self, '_galaxy{0:d}_states'.format(galaxy_index), galaxy_states)

        return (self._galaxy1_result, self._galaxy2_result)

    def _prepare_animating_object(self, axes, zdir='z'):
        lines2body = super()._prepare_animating_object(axes, zdir)
        lines2orbital = dict()
        for galaxy_index, hue in zip(range(1, 3), (218 / 360, 34 / 360)):
            galaxy_states = getattr(self, '_galaxy{0:d}_states'.format(galaxy_index))
            for orbital_states, sat in zip(galaxy_states, np.linspace(1, 0.3, len(galaxy_states))):
                if hasattr(axes, 'plot3D'):
                    line, = axes.plot3D(orbital_states[:, 0, 0], orbital_states[:, 1, 0], orbital_states[:, 2, 0], '.', zdir=zdir, color=hsv_to_rgb((hue, sat, 1)), markersize=3.0)
                else:
                    line, = axes.plot(orbital_states[:, 0, 0], orbital_states[:, 1, 0], '.', color=hsv_to_rgb((hue, sat, 1)), markersize=3.0)
                lines2orbital[line] = orbital_states

        return (lines2body, lines2orbital)

    def _animation_func(self, frame_index, *animating_artists):
        lines = super()._animation_func(frame_index, *animating_artists[0])

        lines2orbital = animating_artists[1]
        for line, orbital_states in lines2orbital.items():
            if hasattr(line, 'set_data_3d'):
                line.set_data_3d(orbital_states[:, 0, frame_index], orbital_states[:, 1, frame_index], orbital_states[:, 2, frame_index])
            else:
                line.set_data(orbital_states[:, 0, frame_index], orbital_states[:, 1, frame_index])
            lines.append(line)

        return lines
