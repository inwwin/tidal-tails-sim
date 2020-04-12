from tidaltailsim.two_body_problem import *
from tidaltailsim.two_galaxy_problem import *
from tidaltailsim.galaxy_orbital_toolkit import *
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import argparse
import pickle
from os.path import splitext
from math import floor


def two_body_routine(args):
    problem = TwoBodyProblem(r0=args.extreme_distance, E=args.energy, M1=args.mass[0], M2=args.mass[1], G=args.gravityconst, use_reduced_mass=not args.rmoff)

    problem.solve_two_body_problem(args.time_span / 2, sampling_points=floor((args.num + 1) / 2))

    parse_animation(args, args.d3,
                    pre_animate_func=lambda ax: problem.plot_two_body_paths(ax),
                    animate_func=lambda fig, ax, speed, framerate: problem.animate(fig, ax, rate=speed, framerate=framerate))


def two_galaxy_routine(args):
    problem = TwoGalaxyProblem(r0=args.extreme_distance, E=args.energy, M1=args.mass[0], M2=args.mass[1], G=args.gravityconst, use_reduced_mass=not args.rmoff)

    problem.verbose = args.verbose
    problem.suppress_error = args.quiet

    problem.solve_two_body_problem(args.time_span / 2, sampling_points=floor((args.num + 1) / 2))

    radii1 = []
    particles1 = []
    for g in args.galaxy1:
        if g[0] > 0 and round(g[1]) > 0:
            radii1.append(g[0])
            particles1.append(int(round(g[1])))
        else:
            raise ValueError('radius {0:f} and number of particles {1:f} in -g1/--galaxy1 must be positive'.format(g[0], g[1]))
    problem.configure_galaxy(1, np.array(radii1), np.array(particles1, dtype=np.int32), theta=args.galaxy1orient[0] * np.pi / 180, phi=args.galaxy1orient[1] * np.pi / 180)

    radii2 = []
    particles2 = []
    if args.galaxy2 is not None:
        for g in args.galaxy2:
            if g[0] > 0 and round(g[1]) > 0:
                radii2.append(g[0])
                particles2.append(int(round(g[1])))
            else:
                raise ValueError('radius {0:f} and number of particles {1:f} in -g2/--galaxy2 must be positive'.format(g[0], g[1]))
        problem.configure_galaxy(2, np.array(radii2), np.array(particles2, dtype=np.int32), theta=args.galaxy2orient[0] * np.pi / 180, phi=args.galaxy2orient[1] * np.pi / 180)

    problem.solve_two_galaxy_problem(atol=args.atol, rtol=args.rtol)

    if args.out is not None:
        filename, file_extension = splitext(args.out)
        if not file_extension:
            filename += '.pkl'
        else:
            filename += file_extension
        with open(filename, 'wb') as o:
            pickle.dump(problem, o)
        if args.verbose:
            print('Solution of two-galaxy problem saved to {0}'.format(filename))

    parse_animation(args, not args.d2,
                    pre_animate_func=lambda ax: problem.plot_two_body_paths(ax),
                    animate_func=lambda fig, ax, speed, framerate: problem.animate(fig, ax, rate=speed, framerate=framerate))


def two_galaxy_pickled_routine(args):
    with open(args.path, 'rb') as f:
        problem = pickle.load(f)

    problem.verbose = args.verbose
    problem.suppress_error = args.quiet

    parse_animation(args, not args.d2,
                    pre_animate_func=lambda ax: problem.plot_two_body_paths(ax),
                    animate_func=lambda fig, ax, speed, framerate: problem.animate(fig, ax, rate=speed, framerate=framerate))


def singleorbital_pickled_routine(args):
    with open(args.path, 'rb') as f:
        problem = pickle.load(f)

    orbital_count = len(getattr(problem, 'galaxy{0:d}_orbitals_properties'.format(args.galaxy)))
    if not (args.orbital < orbital_count and args.orbital >= -orbital_count):
        raise Exception('The input orbital exceed the number of orbitals in the given galaxy\nPlease input orbital between {0} and {1} inclusive'.format(-orbital_count, orbital_count - 1))

    animator = GalaxyOrbitalAnimator(problem, args.galaxy)
    animator.configure_animation(args.orbital, GalaxyOrbitalAnimatorOrigin(args.origin))
    parse_animation(args, not args.d2,
                    animate_func=lambda fig, ax, speed, framerate: animator.animate(fig, ax, rate=speed, framerate=framerate, time_initial=args.timeinitial))


def parse_animation(args, dimension_is_3, animate_func, pre_animate_func=None):
    # if not ((args.animationout is None and args.nogui) or
    #         (args.animationout is not None and args.nogui and args.speed <= 0)):
    if (not args.nogui) or (args.speed > 0 and args.animationout is not None):
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d') if dimension_is_3 else None)

        if args.xlim is not None:
            ax.set_xlim(args.xlim)
        if args.ylim is not None:
            ax.set_ylim(args.ylim)
        if dimension_is_3:
            if args.zlim is not None:
                ax.set_zlim(args.zlim)

        if not dimension_is_3:
            ax.set_aspect('equal')

        if pre_animate_func:
            pre_animate_func(ax)

        if args.speed > 0:

            animation, actual_rate = animate_func(fig, ax, args.speed, args.framerate)  # problem.animate(fig, ax, rate=args.speed, framerate=args.framerate)

            if animation is None:
                if not args.quiet:
                    print('Data not enough to render animation at this speed/framerate combination.')
            else:
                if args.verbose:
                    print(f'Animation prepared.\nThe actual speed of the animation is {actual_rate} simulation time-unit = 1 animation second')

                if args.animationout:
                    if args.animationwriter:
                        writer = args.animationwriter
                    else:
                        writer = None

                    if args.nogui:
                        animation.save(args.animationout, writer=writer,
                                       progress_callback=lambda i, n: print(f'Saving frame {i} of {n}', end='\r'))
                        print('Animation saved to {0}'.format(args.animationout))
                    elif args.adjustgui:
                        fig.show()
                        input('When you finish adjusting the plot in the GUI, return here and press return before closing the window, to continue saving the animation output.')
                        animation.save(args.animationout, writer=writer,
                                       progress_callback=lambda i, n: print(f'Saving frame {i} of {n}', end='\r'))
                        print('Animation saved to {0}'.format(args.animationout))
                    else:
                        animation.save(args.animationout, writer=writer,
                                       progress_callback=lambda i, n: print(f'Saving frame {i} of {n}', end='\r'))
                        print('Animation saved to {0}'.format(args.animationout))
                        plt.show()
                elif not args.nogui:
                    plt.show()
        else:
            if not args.nogui:
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m tidaltailsim', add_help=False, description='A script for solving two-body problem, or two-galaxy collision problem')

    flags_group = parser.add_argument_group(title='flags')
    flags_group.add_argument('-h', '--help', action='help', help='show this help message and exit')
    flags_group.add_argument('--version', action='version', version='tidaltailsim Version 0.2.0\nCopyright © 2020 Panawat Wong-Klaew. All rights reserved.', help='show the version of this script and exit')
    verbosity_group = flags_group.add_mutually_exclusive_group()
    verbosity_group.add_argument('-v', '--verbose', action='store_true', help='be more verbose')
    verbosity_group.add_argument('-q', '--quiet', action='store_true', help='suppress error messages')

    animation_group = parser.add_argument_group(title='animation parameters',
                                                description='parameters relating to the graphical rendering of trajectories of bodies')
    animation_group.add_argument('-s', '--speed', type=float, default=1.0,
                                 metavar='speed',
                                 help='enable animation, with the optionally supplied speed relative to a second, i.e, `speed` time-unit in the simulation = 1 second in the animation. Set to zero to disable animation. (default 1.0)')
    animation_group.add_argument('-fr', '--framerate', type=float,
                                 metavar='fps',
                                 help='number of frames per second, ignored if speed=0 (default, all sampling points will be rendered)')
    animation_group.add_argument('-xlim', nargs=2, type=float,
                                 metavar=('lower', 'upper'),
                                 help='manually set the limit of the x-axis')
    animation_group.add_argument('-ylim', nargs=2, type=float,
                                 metavar=('lower', 'upper'),
                                 help='manually set the limit of the y-axis')
    animation_group.add_argument('-zlim', nargs=2, type=float,
                                 metavar=('lower', 'upper'),
                                 help='manually set the limit of the z-axis')
    dimension_group = animation_group.add_mutually_exclusive_group()
    dimension_group.add_argument('-d3', action='store_true',
                                 help='force plotting in 3D projection')
    dimension_group.add_argument('-d2', action='store_true',
                                 help='force plotting in 2D projection into x-y plane')
    animation_group.add_argument('-ao', '--animationout',
                                 metavar='file',
                                 help='render the animation to a file, ignored if speed=0 (support extensions depend on your system settings, .htm should be okay most of the time, please consult matplotlib documentation)')
    animation_group.add_argument('-aw', '--animationwriter',
                                 metavar='writer',
                                 help='the animation writer backend to be parsed to the animation.save function of matplotlib (default to the value given in matplotlibrc property animation.writer)')
    gui_group = animation_group.add_mutually_exclusive_group()
    gui_group.add_argument('--adjustgui', action='store_true',
                           help='use the gui to zoom/adjust perspective first before saving. When this option is flagged, after the gui shows up, you will have a chance to use your mouse to adjust the plot. When you are done, return to console, and type any key to continue.')
    gui_group.add_argument('--nogui', action='store_true',
                           help='do not render the result to the display screen. Useful if you want to view the resulting video only but you will not be able to manually zoom/adjust the perspective')

    # This required Python 3.7, since MCS use Python 3.6.9, it is disabled
    # subparsers = parser.add_subparsers(title='mode of operation', required=True)
    # This is a workaroun
    subparsers = parser.add_subparsers(title='mode of operation')

    twobody_parser = subparsers.add_parser('2body', help='solve two-body problem only')
    # twobody_parser.add_argument('-d', '--dimension', choices=['2d', '3d'], default='2d', help='Select the display dimension')
    twobody_parser.set_defaults(func=two_body_routine)

    twogalaxy_parser = subparsers.add_parser('2galaxy', help='solve two-galaxy collision problem by first solving two-body problem then use the trajectories of the two bodies as the trajectories of the galaxies'' cores to simulate how stars in the galaxies move')
    twogalaxy_parser.set_defaults(func=two_galaxy_routine)

    twogalaxy_pickled_parser = subparsers.add_parser('2galaxy_fromfile', help='import solved two-galaxy collision problem from a file previously exported from the 2galaxy subcommand')
    twogalaxy_pickled_parser.set_defaults(func=two_galaxy_pickled_routine)

    singleorbital_pickled_parser = subparsers.add_parser('single_orbital_fromfile', help='import solved two-galaxy collision problem from a file previously exported from the 2galaxy subcommand, and animate only a specific orbital with a configurable origin')
    singleorbital_pickled_parser.set_defaults(func=singleorbital_pickled_routine)

    for _parser in [twobody_parser, twogalaxy_parser]:
        general_group = _parser.add_argument_group(title='general simulation parameters')
        general_group.add_argument('time_span', type=float,
                                   help='total time evolution of the system, with extremum in the middle')
        general_group.add_argument('-n', '--num', type=int, default=1999,
                                   metavar='points',
                                   help="Number of sampling points to simulate within the time_span (default 1999 points)")
        general_group.add_argument('--gravityconst', action='store', type=float, default=1.0,
                                   metavar='G',
                                   help='specify the value of the gravitational constant (default 1.0)')
        if _parser is twogalaxy_parser:
            general_group.add_argument('-rtol', type=float, default=1e-3,
                                       help='relative tolorance to be supplied to the solve_ivp function (default 1e-3)')
            general_group.add_argument('-atol', type=float, default=1e-6,
                                       help='absolute tolorance to be supplied to the solve_ivp function (default 1e-6)')

        sim_group = _parser.add_argument_group(title='two-body simulation parameters',
                                               description='The parameters of the simulation relating to the trajectory of the two bodies (in 2body subcommand) or of the two cores (in 2galaxy subcommand)')
        sim_group.add_argument('extreme_distance', type=float,
                               help='distance between the two bodies/cores at extremum')
        sim_group.add_argument('energy', type=float,
                               help='total energy of the two-body system')
        sim_group.add_argument('-m', '--mass', nargs=2, type=float, default=[1.0, 1.0],
                               metavar=('mass_1', 'mass_2'),
                               help='specify the masses of the two bodies/cores (default 1.0 and 1.0)')
        sim_group.add_argument('-rmoff', action='store_true',
                               help='disable reduced mass scheme, assume mass_1 >> mass_2')

    twogalaxy_param_group = twogalaxy_parser.add_argument_group(title='two-galaxy simulation parameters')
    twogalaxy_param_group.add_argument('-g1', '--galaxy1', nargs=2, type=float, action='append', required=True,
                                       metavar=('distance_from_core', 'stars_number'),
                                       help='add `stars_number` evenly-spaceing circularly-orbiting particles to galaxy1 at a distance `distance_from_core` away from the core. `stars_number` should be an integer. (this argument can be parsed multiple times to add multiple orbits)')
    twogalaxy_param_group.add_argument('-g2', '--galaxy2', nargs=2, type=float, action='append',
                                       metavar=('distance_from_core', 'stars_number'),
                                       help='add `stars_number` evenly-spaceing circularly-orbiting particles to galaxy2 at a distance `distance_from_core` away from the core. `stars_number` should be an integer. (this argument can be parsed multiple times to add multiple orbits)')
    twogalaxy_param_group.add_argument('-g1o', '--galaxy1orient', nargs=2, type=float, default=[0.0, 0.0],
                                       metavar=('theta', 'phi'),
                                       help='specify the orientation of the galaxy1\'s galactic plane with respect to the to the two-core collision plane. The galactic plane is first created in the collision plane and then rotated along x-axis by theta and along z-axis by phi. theta and phi must be given in degrees. (default 0.0 0.0)')
    twogalaxy_param_group.add_argument('-g2o', '--galaxy2orient', nargs=2, type=float, default=[0.0, 0.0],
                                       metavar=('theta', 'phi'),
                                       help='specify the orientation of the galaxy2\'s galactic plane with respect to the to the two-core collision plane. The galactic plane is first created in the collision plane and then rotated along x-axis by theta and along z-axis by phi. theta and phi must be given in degrees. (default 0.0 0.0)')

    twogalaxy_parser.add_argument('-o', '--out',
                                  metavar='path',
                                  help='export the simulation result to a file specified by `path`. If no extension is given, the defult extension .pkl will be used.')

    twogalaxy_pickled_parser.add_argument('path', help='the two-galaxy collision problem file to be imported')

    singleorbital_pickled_parser.add_argument('path', help='the two-galaxy collision problem file to be imported')
    singleorbital_pickled_parser.add_argument('galaxy', type=int, choices=(1, 2),
                                              help='the galaxy index of the animating orbital')
    singleorbital_pickled_parser.add_argument('orbital', type=int,
                                              help='the animating orbital zero-based index')
    singleorbital_pickled_parser.add_argument('--origin', choices=(1, 2), default=0,
                                              help='the galaxy index of the origin of the axes of the plot (accept 1 or 2) (default to')
    singleorbital_pickled_parser.add_argument('-t0', '--timeinitial', type=float, default=None,
                                              metavar='t0',
                                              help='the initial time of the animation')

    args = parser.parse_args()

    # Add more logic to handle the workaround for Python 3.6.9
    if args.func:
        args.func(args)
    else:
        print('Please specify the subcommand 2body, 2galaxy, or 2galaxy_fromfile')
