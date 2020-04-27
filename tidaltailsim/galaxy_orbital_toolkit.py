import numpy as np
from enum import IntEnum, IntFlag
from tidaltailsim.two_galaxy_problem import TwoGalaxyProblem
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class GalaxyOrbitalAnimatorOrigin(IntEnum):
    CENTRE_OF_MASS = 0
    GALAXY1 = 1
    GALAXY2 = 2


class GalaxyOrbitalAnimator:
    """an animator for only a single orbital in a 2galaxy problem
    which also allows the origin to be moving with one of the galaxies"""

    def __init__(self, two_galaxy_problem: TwoGalaxyProblem, galaxy_index: int):
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

    def configure_animation(self, orbital_index: int, origin_mode: GalaxyOrbitalAnimatorOrigin):
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

    def animate(self, figure: Figure, axes: Axes, rate=1.0, framerate=None, time_initial=None, event_source=None, **kwargs):
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


class TestMassResultCategory(IntFlag):
    """Flags of the end result of each test mass after the collision event"""

    EllipticOrbitAboutGalaxy1   = 1        # var(accen) within tolerance and accen < 1
    ParabolicOrbitAboutGalaxy1  = 2        # var(accen) within tolerance and accen ~ 1
    HyperbolicOrbitAboutGalaxy1 = 4        # var(accen) within tolerance and accen > 1
    BoundBadOrbitAboutGalaxy1   = 8        # var(accen) out of tolerance and speed within tolerance
    EscapingBadOrbitNearGalaxy1 = 16       # var(accen) out of tolerance and speed out of tolerance, nearest to galaxy1

    EllipticOrbitAboutGalaxy2   = 1 * 32   # var(accen) within tolerance and accen < 1
    ParabolicOrbitAboutGalaxy2  = 2 * 32   # var(accen) within tolerance and accen ~ 1
    HyperbolicOrbitAboutGalaxy2 = 4 * 32   # var(accen) within tolerance and accen > 1
    BoundBadOrbitAboutGalaxy2   = 8 * 32   # var(accen) out of tolerance and speed within tolerance
    EscapingBadOrbitNearGalaxy2 = 16 * 32  # var(accen) out of tolerance and speed out of tolerance, nearest to galaxy2

    about_or_near_galaxy1 = EllipticOrbitAboutGalaxy1 \
        | ParabolicOrbitAboutGalaxy1 \
        | HyperbolicOrbitAboutGalaxy1 \
        | BoundBadOrbitAboutGalaxy1 \
        | EscapingBadOrbitNearGalaxy1

    about_or_near_galaxy2 = EllipticOrbitAboutGalaxy2 \
        | ParabolicOrbitAboutGalaxy2 \
        | HyperbolicOrbitAboutGalaxy2 \
        | BoundBadOrbitAboutGalaxy2 \
        | EscapingBadOrbitNearGalaxy2

    def is_somehow_about_or_near_galaxy1(self):
        return bool(self & TestMassResultCategory.about_or_near_galaxy1)

    def is_somehow_about_or_near_galaxy2(self):
        return bool(self & TestMassResultCategory.about_or_near_galaxy2)

    def is_parent_galaxy_ambiguous(self):
        return self.is_somehow_about_or_near_galaxy1() and self.is_somehow_about_or_near_galaxy2()

    def is_parabolic(self):
        return bool(self & TestMassResultCategory.ParabolicOrbitAboutGalaxy1) \
            or bool(self & TestMassResultCategory.ParabolicOrbitAboutGalaxy2)

    def is_bound_bad_orbit(self):
        return bool(self & TestMassResultCategory.BoundBadOrbitAboutGalaxy1) \
            or bool(self & TestMassResultCategory.BoundBadOrbitAboutGalaxy2)

    def should_be_reviewed(self):
        return self.is_parabolic() or self.is_bound_bad_orbit()


class TestMassResultCriteriaBase:
    """A base class of criteria for categorising the end result of a test mass"""

    def accentricity_unity_threshold(self, accentricity_variance):
        """
        Return the minimum threshold that the accentricity has to be different from 1.000
        that the orbit required to have in order to be categorised as non-parabolic

        Parameter
        =========
        `aceentricity_variance`: the variance of the accentricity of the orbit

        Return
        ======
        a (neg, pos) tuple of float or
            --> accept this orbit as elliptical if accentricity is less than 1 - neg
                or accept as hyperbolae if accentricity is greater than 1 + pos
        a float or
            --> equivalent to tuple (neg, pos) with neg=pos
        False
            --> reject this from well-behaved orbit straightaway
        """
        raise NotImplementedError()

    def is_escaping(self, displacement_magnitude_change_rate):
        """
        in case of a non well-behaved orbit,
        based on the rate of change of the magnitude of the displacement,
        should this orbit still be said to be escaping from the system

        Return
        ======
        bool
        """
        raise NotImplementedError()


class TestMassResultLogarithmicCriteria(TestMassResultCriteriaBase):
    """
    Extend `TestMassResultCriteria`. Accentricity threshold scales logarithmically with the variance.

    Paremeter
    =========
    `escaping_speed_threshold`: min speed threshold to say that the orbit is an escaping type
    """

    def __init__(self, escaping_speed_threshold: float):
        self.escaping_speed_threshold = escaping_speed_threshold  # type: float

        self.min_accentricity_variance_tolerance = 0.001  # type: float
        self.max_accentricity_variance_tolerance = 0.05  # type: float

        self.min_accentricity_unity_threshold = 0.05  # type: float
        self.max_accentricity_unity_threshold = 0.2  # type: float

    def accentricity_unity_threshold(self, accentricity_variance: float):
        from math import log10
        if accentricity_variance > self.max_accentricity_variance_tolerance:
            # reject not well-behaved orbit
            return False

        if accentricity_variance < self.min_accentricity_variance_tolerance:
            # if variance is whithin a specified tolerance
            # set accentricity unity threshold to the minimum
            return self.min_accentricity_unity_threshold

        return (log10(accentricity_variance) - log10(self.min_accentricity_variance_tolerance)) \
            / (log10(self.max_accentricity_variance_tolerance) - log10(self.min_accentricity_variance_tolerance)) \
            * (self.max_accentricity_unity_threshold - self.min_accentricity_unity_threshold) \
            + self.min_accentricity_unity_threshold

    def is_escaping(self, displacement_magnitude_change_rate: float):
        return displacement_magnitude_change_rate > self.escaping_speed_threshold


class TestMassProfiler:
    """an advanced tool for profiling each test mass"""

    def __init__(self, two_galaxy_problem: TwoGalaxyProblem,
                 galaxy_index: int, orbital_index: int, test_mass_index: int,
                 figure: Figure = None, frame_slice: slice = slice(None)):
        if not isinstance(two_galaxy_problem, TwoGalaxyProblem):
            raise TypeError('two_galaxy_problem must be an instance of tidaltailsim.two_galaxy_problem.TwoGalaxyProblem')
        if galaxy_index not in (1, 2):
            raise ValueError('galaxy_index must be either 1 or 2')
        if frame_slice is None:
            frame_slice = slice(None)
        self._problem = two_galaxy_problem
        self._galaxy_index = galaxy_index

        self._test_mass_raw_states = \
            getattr(self._problem, 'galaxy{0:d}_orbitals_properties'.format(self._galaxy_index))[orbital_index]['states'][test_mass_index, ...]

        self._orbital_index = orbital_index
        self._test_mass_index = test_mass_index

        self.default_frame_slice = frame_slice  # type: slice

        self._calculate_relative_states_from_cores()

        self.consume_figure(figure)

    @property
    def two_galaxy_problem(self) -> TwoGalaxyProblem:
        return self._problem

    @property
    def galaxy_index(self) -> int:
        return self._galaxy_index

    @property
    def test_mass_raw_states(self):
        return self._test_mass_raw_states

    @property
    def orbital_index(self) -> int:
        return self._orbital_index

    @property
    def test_mass_index(self) -> int:
        return self._test_mass_index

    @property
    def figure(self) -> Figure:
        return self._figure

    @property
    def axes_array(self):
        return self._axs

    def _calculate_relative_states_from_cores(self):
        cores_states = np.concatenate((self._problem.cartesian_coordinates_evolution,
                                       self._problem.cartesian_velocity_evolution),
                                      axis=1)

        test_mass_states_double = np.stack((self._test_mass_raw_states, self._test_mass_raw_states))

        self._relative_test_mass_states = test_mass_states_double - cores_states

        self._distace_from_cores = np.sqrt(np.sum(np.square(self._relative_test_mass_states[:, :3, ...]), axis=1))

        # specific_angular_momentum = r ^ v
        specific_angular_momentum = np.cross(self._relative_test_mass_states[:, :3, ...],
                                             self._relative_test_mass_states[:, 3:, ...],
                                             axis=1)

        squared_specific_angular_momentum = np.sum(np.square(specific_angular_momentum), axis=1)

        masses_G = np.array([self._problem.M1_feel, self._problem.M2_feel])[:, np.newaxis] * self._problem.G_feel

        # specfic_energy = (kinetic + potential) / test_mass
        specific_energy = .5 * np.sum(np.square(self._relative_test_mass_states[:, 3:, ...]), axis=1) \
            - masses_G / self._distace_from_cores

        # eccentricity defined here https://en.wikipedia.org/wiki/Orbital_eccentricity#Definition
        self._eccentricity = np.sqrt(1 + 2 * specific_energy * squared_specific_angular_momentum / np.square(masses_G))

    def _plot_distances_from_cores(self, frame_slice: slice):
        for i in range(2):
            self.axes_array[0, i].plot(self.two_galaxy_problem.time_domain[frame_slice], self._distace_from_cores[i, frame_slice])

    def _plot_eccentricity(self, frame_slice: slice):
        for i in range(2):
            self.axes_array[1, i].plot(self.two_galaxy_problem.time_domain[frame_slice], self._eccentricity[i, frame_slice])

    def _plot_distance_fits(self, frame_slice: slice):
        coeff = self.distance_linear_regression(frame_slice)
        for i in range(2):
            fit = np.poly1d(coeff[i, :])
            time_limit = self._problem.time_domain[frame_slice][(0, -1), ]
            self.axes_array[0, i].plot(time_limit, fit(time_limit), ':')

    def _annotate_figure(self):
        for i, quantity in zip(range(2), ('Distance', 'Eccentricity')):
            for j in range(2):
                ax = self.axes_array[i, j]  # type: Axes
                ax.set_xlabel('Time')
                ax.set_ylabel(quantity)
                ax.set_title(f'{quantity} relative to\nthe core of galaxy {j+1}')
        self.figure.text(0, 0, f'test_mass_index={self.test_mass_index}')

    def consume_figure(self, figure: Figure, frame_slice: slice = None):
        """plot graphs of distances from cores and aceentricities and also annotate the figure"""
        if frame_slice is None:
            frame_slice = self.default_frame_slice

        if not isinstance(figure, Figure):
            self._figure = None  # type: Figure
            self._axs = None
        else:
            self._figure = figure  # type: Figure
            self._axs = figure.subplots(2, 2, sharex=True)
            # rows: 0=>distance from core 1=>aceentricity
            # columns: 0=>rel to core1 1=>rel to core2

            self._figure.set_tight_layout(True)
            self._annotate_figure()

            self._plot_distances_from_cores(frame_slice)
            self._plot_distance_fits(frame_slice)
            self._plot_eccentricity(frame_slice)

        return self._axs

    def analyse(self, frame_slice: slice = None):
        """calculate important parameters about the trajectory of the test mass"""
        if frame_slice is None:
            frame_slice = self.default_frame_slice
        mean_distance = np.nanmean(self._distace_from_cores[:, frame_slice], axis=1)
        var_distance = np.nanvar(self._distace_from_cores[:, frame_slice], axis=1, ddof=1)

        mean_eccentricity = np.nanmean(self._eccentricity[:, frame_slice], axis=1)
        var_eccentricity = np.nanvar(self._eccentricity[:, frame_slice], axis=1, ddof=1)

        coeff = self.distance_linear_regression(frame_slice)

        times = self._problem.time_domain[frame_slice]

        result = [{
            'relative to galaxy': i + 1,
            'distance': {
                'mean': mean_distance[i],
                'variance': var_distance[i],
                'linear gradient': coeff[i, 0]
            },
            'eccentricity': {
                'mean': mean_eccentricity[i],
                'variance': var_eccentricity[i]
            }
        } for i in range(2)]

        return {
            'time limit': (times[0], times[-1]),
            'test_mass_index': self.test_mass_index,
            'result': result
        }

    def distance_linear_regression(self, frame_slice: slice = None):
        """calculate the average rate of change of the magnitude of the displacement from cores"""
        if frame_slice is None:
            frame_slice = self.default_frame_slice

        coeff = np.polyfit(self._problem.time_domain[frame_slice], self._distace_from_cores[:, frame_slice].T, 1).T

        return coeff

    def categorise(self, criteria: TestMassResultCriteriaBase, frame_slice: slice = None) -> TestMassResultCategory:
        """try to categorise the end result of the test mass after a collision event using the supplied `criteria`"""
        category = TestMassResultCategory(0)

        analysis = self.analyse(frame_slice)['result']
        analysis1 = analysis[0]
        analysis2 = analysis[1]

        accentricity_threshold1 = criteria.accentricity_unity_threshold(analysis1['eccentricity']['variance'])
        accentricity_threshold2 = criteria.accentricity_unity_threshold(analysis2['eccentricity']['variance'])

        if accentricity_threshold1:
            if not isinstance(accentricity_threshold1, tuple):
                accentricity_threshold1 = (accentricity_threshold1, accentricity_threshold1)

            if analysis1['eccentricity']['mean'] < 1 - accentricity_threshold1[0]:
                category |= TestMassResultCategory.EllipticOrbitAboutGalaxy1
            elif analysis1['eccentricity']['mean'] > 1 + accentricity_threshold1[1]:
                category |= TestMassResultCategory.HyperbolicOrbitAboutGalaxy1
            else:
                category |= TestMassResultCategory.ParabolicOrbitAboutGalaxy1
        else:
            if not criteria.is_escaping(analysis1['distance']['linear gradient']):
                category |= TestMassResultCategory.BoundBadOrbitAboutGalaxy1

        if accentricity_threshold2:
            if not isinstance(accentricity_threshold2, tuple):
                accentricity_threshold2 = (accentricity_threshold2, accentricity_threshold2)

            if analysis2['eccentricity']['mean'] < 1 - accentricity_threshold2[0]:
                category |= TestMassResultCategory.EllipticOrbitAboutGalaxy2
            elif analysis2['eccentricity']['mean'] > 1 + accentricity_threshold2[1]:
                category |= TestMassResultCategory.HyperbolicOrbitAboutGalaxy2
            else:
                category |= TestMassResultCategory.ParabolicOrbitAboutGalaxy2
        else:
            if not criteria.is_escaping(analysis2['distance']['linear gradient']):
                category |= TestMassResultCategory.BoundBadOrbitAboutGalaxy2

        if not (category & int('0111101111', 2)):
            # unable to categorise into any orbits
            # so let's say it is escaping from the system
            if analysis1['distance']['mean'] < analysis2['distance']['mean']:
                category |= TestMassResultCategory.EscapingBadOrbitNearGalaxy1
            elif analysis1['distance']['mean'] > analysis2['distance']['mean']:
                category |= TestMassResultCategory.EscapingBadOrbitNearGalaxy2
            else:
                raise Exception('Super weird orbit detected')

        return category


class TwoGalaxyProblemProfiler:
    """bulk categorising all test masses in a galaxy, including reviewing the results"""

    def __init__(self, problem: TwoGalaxyProblem, galaxy_index: int):
        if not isinstance(problem, TwoGalaxyProblem):
            raise TypeError('two_galaxy_problem must be an instance of tidaltailsim.two_galaxy_problem.TwoGalaxyProblem')
        if galaxy_index not in (1, 2):
            raise ValueError('galaxy_index must be either 1 or 2')

        self._problem = problem
        self._galaxy_index = galaxy_index

        self.raw_categories = dict()

        self._orbitals_properties = getattr(self._problem, 'galaxy{0:d}_orbitals_properties'.format(self._galaxy_index))

    def auto_categorise_single_orbital(self, criteria: TestMassResultCriteriaBase, orbital_index: int,
                                       frame_slice: slice = slice(None)):
        if frame_slice is None:
            frame_slice = slice(None)
        if not isinstance(orbital_index, int) or orbital_index < 0 or orbital_index >= len(self._orbitals_properties):
            raise IndexError()

        self.raw_categories[orbital_index] = list()

        test_mass_count = self._orbitals_properties[orbital_index]['states'].shape[0]

        for test_mass_index in range(test_mass_count):
            profiler = TestMassProfiler(self._problem, self._galaxy_index, orbital_index, test_mass_index,
                                        frame_slice=frame_slice)
            self.raw_categories[orbital_index].append(profiler.categorise(criteria))

        return self.raw_categories[orbital_index]

    def auto_categorise_all_orbitals(self, criteria: TestMassResultCriteriaBase, frame_slice: slice = slice(None)):
        for i in range(len(self._orbitals_properties)):
            self.auto_categorise_single_orbital(criteria, i, frame_slice)

    def detect_ambiguous_test_masses(self):
        ambiguous_test_masses = list()
        for orbital_index, orbital_categories in self.raw_categories.items():
            for test_mass_index, test_mass_category in enumerate(orbital_categories):
                if test_mass_category.is_parent_galaxy_ambiguous():
                    ambiguous_test_masses.append((orbital_index, test_mass_index))

        return ambiguous_test_masses

    def export_categories_csv(self, file):
        from csv import DictWriter
        fieldnames = ['orbital_index', 'test_mass_index', 'categories_flags']
        writer = DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        for orbital_index, orbital_categories in self.raw_categories.items():
            for test_mass_index, test_mass_category in enumerate(orbital_categories):
                writer.writerow({'orbital_index': orbital_index,
                                 'test_mass_index': test_mass_index,
                                 'categories_flags': int(test_mass_category)})

    def import_categories_csv(self, file):
        from csv import DictReader
        reader = DictReader(file)
        self.raw_categories = dict()

        for row in reader:
            orbital_index = int(row['orbital_index'])
            if orbital_index not in self.raw_categories:
                self.raw_categories[orbital_index] = list()

            cat_list = self.raw_categories[orbital_index]  # type: list
            if int(row['test_mass_index']) == len(cat_list):
                cat_list.append(TestMassResultCategory(int(row['categories_flags'])))
            else:
                raise Exception('test_mass_index needs to be sorted and increased one-by-one')

    color_converter = {
        TestMassResultCategory.EllipticOrbitAboutGalaxy1: 'cornflowerblue',
        TestMassResultCategory.ParabolicOrbitAboutGalaxy1: 'mediumseagreen',
        TestMassResultCategory.HyperbolicOrbitAboutGalaxy1: 'mediumseagreen',
        TestMassResultCategory.BoundBadOrbitAboutGalaxy1: 'royalblue',
        TestMassResultCategory.EscapingBadOrbitNearGalaxy1: 'seagreen',
        TestMassResultCategory.EllipticOrbitAboutGalaxy2: 'indianred',
        TestMassResultCategory.ParabolicOrbitAboutGalaxy2: 'lightcoral',
        TestMassResultCategory.HyperbolicOrbitAboutGalaxy2: 'goldenrod',
        TestMassResultCategory.BoundBadOrbitAboutGalaxy2: 'brown',
        TestMassResultCategory.EscapingBadOrbitNearGalaxy2: 'darkgoldenrod',
    }

    def to_colors(self):
        """convert the test mass categories to a list of colour codes for supllying to the animation function of TwoGalaxyProblem"""
        color_lists = list()
        for orbital_index, cat_list in sorted(self.raw_categories.items(), key=lambda kv: kv[0]):
            color_list = [self.color_converter[cat] if cat in self.color_converter else 'black' for cat in cat_list]
            color_lists.append(color_list)
        return color_lists if color_lists else None
