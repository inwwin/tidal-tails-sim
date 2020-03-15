from tidaltailsim.two_body_problem import TwoBodyProblem
from matplotlib import pyplot as plt
import argparse


def two_body_routine(args):
    problem = TwoBodyProblem(r0=args.r0, E=args.energy, M1=args.mass[0], M2=args.mass[1])
    # print(problem.angular_momentum)
    problem.solve_two_body_problem(10)
    # print(problem.__x1,problem.__y1)
    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    problem.plot_two_body_paths(ax)

    if args.animation:
        animation = problem.animate(fig, ax, args.animation)

    if args.out:
        if animation:
            animation.save(args.out,
                           # fps=60,
                           progress_callback=lambda i, n: print(f'Saving frame {i} of {n}')
                           )
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tidaltailsim')
    subparsers = parser.add_subparsers()
    parser.add_argument('r0', type=float, help='Distance between two massive bodies/galaxies at extremum')
    parser.add_argument('energy', type=float, help='Total energy of the two system')
    parser.add_argument('-m', '--mass', nargs=2, type=float, default=[1.0, 1.0])
    parser.add_argument('-a', '--animation', nargs='?', type=float, const=1.0, default=None, help='Enable animation')
    parser.add_argument('-o', '--out', help='Render the result to still image file')

    twobody_parser = subparsers.add_parser('2body')
    # twobody_parser.add_argument('-d', '--dimension', choices=['2d', '3d'], default='2d', help='Select the display dimension')
    twobody_parser.set_defaults(func=two_body_routine)

    args = parser.parse_args()
    args.func(args)
