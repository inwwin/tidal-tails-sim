import numpy as np
from scipy.integrate import solve_ivp
from tidaltailsim.two_body_problem import TwoBodyProblem


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
            raise ValueError('A mass must be floating number')

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
            raise ValueError('A mass must be floating number')

    def configure_galaxy(self, galaxy_number, orbital_radius, orbital_particles, theta=0., phi=0., m=None):
        """
        Set up the initial conditions of the collection of particles
        orbiting around the core of the galaxy

        Parameters:
        galaxy_number       -- a number 1 or 2 specifying which galaxy is to be configured
        orbital_radius      -- 1d array (n,) of the radius of each orbit
        orbital_particles   -- integer 1d array (n,) of the number of particles
                               in each orbit
        theta, phi          -- (optional) the spherical polar coordinates of the axis of
                               the orbital plane with respect to the z-axis
                               (z-axis is the normal to the plane of two-body-problem)
        m                   -- (optional) customise the mass of the core used for calculating the initial condition
        """
        orbital_particles = np.int_(orbital_particles)

        if not (galaxy_number == 1 or galaxy_number == 2):
            raise ValueError('Expect either 1 or 2 in galaxy_number')
        else:
            is_g_1 = galaxy_number == 1  # True for galaxy 1 False for galaxy 2

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
