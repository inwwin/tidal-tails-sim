from tidaltailsim.two_body_problem import TwoBodyProblem
from matplotlib import pyplot as plt

if __name__ == '__main__':
    problem = TwoBodyProblem(1,E=+1)
    print(problem.angular_momentum)
    problem.solve_two_body_problem(10)
    # print(problem.__x1,problem.__y1)
    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    problem.plot_two_body_paths(ax)

    plt.show()
