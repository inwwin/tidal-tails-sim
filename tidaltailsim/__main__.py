from tidaltailsim.two_body_problem import TwoBodyProblem
from tidaltailsim.two_galaxy_problem import TwoGalaxyProblem
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import argparse


def two_body_routine(args):
    problem = TwoBodyProblem(r0=args.r0, E=args.energy, M1=args.mass[0], M2=args.mass[1], use_reduced_mass=not args.rmoff)
    # print(problem.angular_momentum)
    problem.solve_two_body_problem(args.time_span / 2)
    # print(problem.__x1,problem.__y1)
    fig, ax = plt.subplots(subplot_kw=None if not args.d3 else dict(projection='3d'))
    # fig = plt.Figure()
    # ax = fig.add_subplot(projection='3d')

    if not args.d3:
        ax.set_aspect('equal')
    problem.plot_two_body_paths(ax)  # , plot_v0=(1, [0, 800, 999, 1200, -1]))

    if args.animation:

        animation = problem.animate(fig, ax, rate=args.animation, framerate=args.framerate)

        if args.out:
            animation.save(args.animationout,
                           progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
                           )
        elif not args.nogui:
            plt.show()
    else:
        if not args.nogui:
            plt.show()


def two_galaxy_routine(args):
    problem = TwoGalaxyProblem(r0=args.r0, E=args.energy, M1=args.mass[0], M2=args.mass[1], use_reduced_mass=not args.rmoff)

    problem.solve_two_body_problem(args.time_span / 2)
    radii = np.arange(2, 7)
    problem.configure_galaxy(1, radii, radii * 6, np.pi / 12, np.pi / 6)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d') if not args.d2 else None)
    # fig = plt.Figure()
    # ax = fig.add_subplot(projection='3d')

    problem.plot_two_body_paths(ax)  # , plot_v0=(1, [0, 800, 999, 1200, -1]))
    problem.plot_galaxies_initial_positions(ax)
    plt.show()

    # if args.animation:

    #     animation = problem.animate(fig, ax, rate=args.animation, framerate=args.framerate)

    #     if args.out:
    #         animation.save(args.animationout,
    #                        progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
    #                        )
    #     elif not args.nogui:
    #         plt.show()
    # else:
    #     if not args.nogui:
    #         plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m tidaltailsim')
    parser.add_argument('r0', type=float,
                        help='distance between two massive bodies/galaxies at extremum')
    parser.add_argument('energy', type=float,
                        help='total energy of the two system')
    parser.add_argument('time_span', type=float,
                        help='total time evolution of the system, with extremum at the centre')
    parser.add_argument('-m', '--mass', nargs=2, type=float, default=[1.0, 1.0],
                        metavar=('mass_1', 'mass_2'))
    parser.add_argument('-rmoff', action='store_true',
                        help='don\'t use reduced mass, assume mass_1 >> mass_2')
    dimension_group = parser.add_mutually_exclusive_group()
    dimension_group.add_argument('-d3', action='store_true',
                                 help='force plotting in 3D')
    dimension_group.add_argument('-d2', action='store_true',
                                 help='force plotting in 2D')
    parser.add_argument('-a', '--animation', nargs='?', type=float, const=1.0, default=None,
                        metavar='speed',
                        help='enable animation, with the optionally supplied speed (default 1.0)')
    parser.add_argument('-fr', '--framerate', type=float,
                        metavar='fps',
                        help='number of frames per second, ignored if -a or --animation is not supplied (default, all sampling points will be rendered)')
    parser.add_argument('-ao', '--animationout',
                        metavar='file',
                        help='render the animation to a file')
    parser.add_argument('--nogui', action='store_true',
                        help='do not render the result to the display')

    subparsers = parser.add_subparsers()
    twobody_parser = subparsers.add_parser('2body')
    # twobody_parser.add_argument('-d', '--dimension', choices=['2d', '3d'], default='2d', help='Select the display dimension')
    twobody_parser.set_defaults(func=two_body_routine)

    twogalaxy_parser = subparsers.add_parser('2galaxy')
    twogalaxy_parser.set_defaults(func=two_galaxy_routine)

    args = parser.parse_args()
    args.func(args)
