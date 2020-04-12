import numpy as np
from enum import IntEnum
from tidaltailsim.two_galaxy_problem import TwoGalaxyProblem
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.axes3d import Axes3D


class GalaxyOrbitalAnimator:
    def __init__(self, two_galaxy_problem, galaxy_index):
        if two_galaxy_problem is not TwoGalaxyProblem:
            raise TypeError('two_galaxy_problem must be an instance of tidaltailsim.two_galaxy_problem.TwoGalaxyProblem')
        self._problem = two_galaxy_problem
        self._galaxy_index = galaxy_index

    @property
    def problem(self):
        return self._problem

    @property
    def galaxy_index(self):
        return self._galaxy_index

    @property
    def origin_mode(self):
        return self._origin_mode

    @property
    def target_orbital_index(self):
        return self._target_orbital_index

    @property
    def target_orbital_properties(self):
        return self._target_orbital_properties

    @property
    def orbital_states_relative(self):
        return self._orbital_states_relative

    @property
    def cores_states_relative(self):
        return self._cores_states_relative

    def configure_animation(self, orbital_index, origin_mode):
        orbital_index = int(orbital_index)
        if origin_mode is not GalaxyOrbitalAnimatorOrigin:
            raise TypeError('origin_mode must be an enum in tidaltailsim.galaxy_orbital_animator.GalaxyOrbitalAnimatorOrigin')

        orbitals_properties = getattr(self._problem, 'galaxy{0:d}_orbitals_properties'.format(self._galaxy_index))
        if not orbitals_properties:
            raise Exception('Orbital properties not found')
        self._target_orbital_properties = orbitals_properties[orbital_index]
        self._target_orbital_index = orbital_index
        self._origin_mode = origin_mode

        cores_states = np.concatenate((self._problem.cartesian_coordinates_evolution,
                                       self._problem.cartesian_velocity_evolution),
                                      axis=1)

        origin_states = cores_states[origin_mode - 1, ...] if origin_mode else .0

        orbital_states = self._target_orbital_properties['states']

        self._orbital_states_relative = orbital_states - origin_states  # broadcasting
        self._cores_states_relative = cores_states - origin_states  # broadcasting

    def animate(self, figure, axes, zdir='z', rate=1.0, framerate=None):

        # if framerate is not given, all frames get rendered (potentially impacting the performance)
        if framerate:
            framestep = int(round(self._problem.sampling_points * rate / (framerate * self._problem.time_end)))
            interval = int(round(framestep * 1000 * self._problem.time_end / self._problem.sampling_points / rate))
            if interval <= 0:
                return (None, 0.)  # The supplied parameter means the animation is too slow that there is not enough data
            frames = range(0, self._problem.time_domain.shape[0], framestep)
            actual_rate = 1000.0 * framestep * self._problem.time_end / interval / self._problem.sampling_points
        else:
            interval = int(round(1000 * self._problem.time_end / self._problem.sampling_points / rate))
            if interval <= 0:
                return (None, 0.)  # The supplied parameter means the animation is too slow that there is not enough data
            frames = self._problem.time_domain.shape[0]
            actual_rate = 1000.0 * self._problem.time_end / interval / self._problem.sampling_points

        animating_artists = self._prepare_animating_object(axes, zdir)

        animation = \
            FuncAnimation(figure,
                          blit=not isinstance(axes, Axes3D),
                          frames=frames,
                          interval=interval,
                          fargs=animating_artists,
                          func=self._animation_func)

        return (animation, actual_rate)


class GalaxyOrbitalAnimatorOrigin(IntEnum):
    CENTRE_OF_MASS = 0
    GALAXY1 = 1
    GALAXY2 = 2
