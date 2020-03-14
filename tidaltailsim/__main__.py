from tidaltailsim.two_body_problem import TwoBodyProblem
from matplotlib import pyplot as plt
import argparse


def two_body_routine(args):
    problem = TwoBodyProblem(r0=args.r0, E=args.E)
    # print(problem.angular_momentum)
    problem.solve_two_body_problem(10)
    # print(problem.__x1,problem.__y1)
    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    problem.plot_two_body_paths(ax)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='tidaltailsim')
    subparsers = parser.add_subparsers()

    twobody_parser = subparsers.add_parser('2body')
    twobody_parser.add_argument('r0', type=float, help='Distance between two massive bodies/galaxies at extremum')
    twobody_parser.add_argument('E', type=float, help='Energy of the two massive bodies/galaxies')
    # twobody_parser.add_argument('-d', '--dimension', choices=['2d', '3d'], default='2d', help='Select the display dimension')
    twobody_parser.add_argument('-a', '--animation', action='store_true', help='Enable animation')
    # twobody_parser.add_argument('-o', '--out', help='Render the result to still image file')
    twobody_parser.set_defaults(func=two_body_routine)

    args = parser.parse_args()
    args.func(args)
