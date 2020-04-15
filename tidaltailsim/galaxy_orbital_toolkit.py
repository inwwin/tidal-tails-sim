import numpy as np
from enum import IntEnum
from tidaltailsim.two_galaxy_problem import TwoGalaxyProblem
from matplotlib.animation import FuncAnimation


class GalaxyOrbitalAnimator:
    def __init__(self, two_galaxy_problem, galaxy_index):
        if not isinstance(two_galaxy_problem, TwoGalaxyProblem):
            raise TypeError('two_galaxy_problem must be an instance of tidaltailsim.two_galaxy_problem.TwoGalaxyProblem')
        if galaxy_index not in (1, 2):
            raise ValueError('galaxy_index must be either 1 or 2')
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
        if not isinstance(origin_mode, GalaxyOrbitalAnimatorOrigin):
            raise TypeError('origin_mode must be an enum in tidaltailsim.galaxy_orbital_toolkit.GalaxyOrbitalAnimatorOrigin')

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

    def plot_core_path(self, axes, galaxy_index):
        if self._cores_states_relative is None:
            raise Exception('Cores states data not found. Please call configure_animation first.')
        if galaxy_index not in (1, 2):
            raise ValueError('galaxy_index must be either 1 or 2')

        galaxy_index -= 1

        if hasattr(axes, 'plot3D'):
            core_path, = axes.plot3D(self._cores_states_relative[galaxy_index, 0, :],
                                     self._cores_states_relative[galaxy_index, 1, :],
                                     self._cores_states_relative[galaxy_index, 2, :],
                                     '-', color='skyblue')
        else:
            core_path, = axes.plot(self._cores_states_relative[galaxy_index, 0, :],
                                   self._cores_states_relative[galaxy_index, 1, :],
                                   '-', color='skyblue')

        return core_path

    def plot_test_mass_path(self, axes, test_mass_index: int, path_slice: slice, **kwargs):
        if self._orbital_states_relative is None:
            raise Exception('Orbital states data not found. Please call configure_animation first.')

        if hasattr(axes, 'plot3D'):
            test_mass_path, = axes.plot3D(self._orbital_states_relative[test_mass_index, 0, path_slice],
                                          self._orbital_states_relative[test_mass_index, 1, path_slice],
                                          self._orbital_states_relative[test_mass_index, 2, path_slice],
                                          '-', color='darkslategrey', **kwargs)
        else:
            test_mass_path, = axes.plot(self._orbital_states_relative[test_mass_index, 0, path_slice],
                                        self._orbital_states_relative[test_mass_index, 1, path_slice],
                                        '-', color='darkslategrey', **kwargs)

        return test_mass_path

    def _prepare_animating_object(self, axes, frame_initial, **kwargs):
        if hasattr(axes, 'plot3D'):
            cores, = axes.plot3D(self._cores_states_relative[:, 0, frame_initial],
                                 self._cores_states_relative[:, 1, frame_initial],
                                 self._cores_states_relative[:, 2, frame_initial],
                                 '.', color='navy', markersize=5.0)
            orbits, = axes.plot3D(self._orbital_states_relative[:, 0, frame_initial],
                                  self._orbital_states_relative[:, 1, frame_initial],
                                  self._orbital_states_relative[:, 2, frame_initial],
                                  '.-', color='maroon', markersize=3.0, **kwargs)
        else:
            cores, = axes.plot(self._cores_states_relative[:, 0, frame_initial],
                               self._cores_states_relative[:, 1, frame_initial],
                               '.', color='navy', markersize=5.0)
            orbits, = axes.plot(self._orbital_states_relative[:, 0, frame_initial],
                                self._orbital_states_relative[:, 1, frame_initial],
                                '.-', color='maroon', markersize=3.0, **kwargs)

        time_annotate = axes.annotate('frame index = {1}\nt = {0:=+8.3f}'.format(self._problem.time_domain[frame_initial], frame_initial),
                                      xy=(0, 0), xycoords='figure fraction',
                                      xytext=(6, 6), textcoords='offset pixels',
                                      fontfamily='monospace')

        return (cores, orbits, time_annotate)

    def _animation_func(self, frame_index, cores, orbits, time_annotate):
        if hasattr(cores, 'set_data_3d'):
            cores.set_data_3d(self._cores_states_relative[:, 0, frame_index],
                              self._cores_states_relative[:, 1, frame_index],
                              self._cores_states_relative[:, 2, frame_index])
        else:
            cores.set_data(self._cores_states_relative[:, 0, frame_index],
                           self._cores_states_relative[:, 1, frame_index])
        if hasattr(orbits, 'set_data_3d'):
            orbits.set_data_3d(self._orbital_states_relative[:, 0, frame_index],
                               self._orbital_states_relative[:, 1, frame_index],
                               self._orbital_states_relative[:, 2, frame_index])
        else:
            orbits.set_data(self._orbital_states_relative[:, 0, frame_index],
                            self._orbital_states_relative[:, 1, frame_index])
        time_annotate.set_text('frame index = {1}\nt = {0:=+8.3f}'.format(self._problem.time_domain[frame_index], frame_index))
        return (cores, orbits, time_annotate)

    def animate(self, figure, axes, rate=1.0, framerate=None, time_initial=None, event_source=None, **kwargs):
        if self._orbital_states_relative is None or self._cores_states_relative is None:
            raise Exception('Relative states data not found. Please call configure_animation first.')

        if time_initial is not None:
            frame_time_zero = (self._problem.time_domain.shape[0] - 1) / 2
            frame_initial = int(frame_time_zero * (1 + time_initial / self._problem.time_end))
        else:
            frame_initial = 0

        # if framerate is not given, all frames get rendered (potentially impacting the performance)
        if framerate:
            framestep = int(round(self._problem.sampling_points * rate / (framerate * self._problem.time_end)))
            interval = int(round(framestep * 1000 * self._problem.time_end / self._problem.sampling_points / rate))
            if interval <= 0:
                return (None, 0.)  # The supplied parameter means the animation is too slow that there is not enough data
            frames = range(frame_initial, self._problem.time_domain.shape[0], framestep)
            actual_rate = 1000.0 * framestep * self._problem.time_end / interval / self._problem.sampling_points
        else:
            interval = int(round(1000 * self._problem.time_end / self._problem.sampling_points / rate))
            if interval <= 0:
                return (None, 0.)  # The supplied parameter means the animation is too slow that there is not enough data
            frames = range(frame_initial, self._problem.time_domain.shape[0])
            actual_rate = 1000.0 * self._problem.time_end / interval / self._problem.sampling_points

        animating_artists = self._prepare_animating_object(axes, frame_initial, **kwargs)

        if event_source:
            event_source.interval = interval

        animation = \
            FuncAnimation(figure,
                          blit=False,  # blitting must be disabled to allow rendering the time annotation outside of the axes
                          frames=frames,
                          interval=interval,
                          fargs=animating_artists,
                          func=self._animation_func,
                          event_source=event_source)

        return (animation, actual_rate, animating_artists[1])


class GalaxyOrbitalAnimatorOrigin(IntEnum):
    CENTRE_OF_MASS = 0
    GALAXY1 = 1
    GALAXY2 = 2


class TestMassProfiler:

    def __init__(self, problem: TwoGalaxyProblem, test_mass_index: int, fig: Figure = None):
        if fig is None:
            fig = Figure()

        self._fig = fig
        fig.subplots
