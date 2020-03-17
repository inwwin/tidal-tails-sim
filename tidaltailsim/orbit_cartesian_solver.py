import numpy as np
from scipy.integrate import solve_ivp


class OrbitCartesianSolver:
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

        GM = 1 * core_mass

        positional_difference = evaluating_position - core_position
        posi_diff_powered = np.sum(positional_difference**2, axis=0)**(-1.5)

        grad_potential = GM * posi_diff_powered * positional_difference  # broadcast
        # print(core_mass)
        print(evaluating_position)
        print(core_position)
        # print(positional_difference)
        # print(posi_diff_powered)
        print(grad_potential)
        # print(core_position.shape, evaluating_position.shape, positional_difference.shape, grad_potential.shape)
        return grad_potential

    def _unit_test_mass_hamilton_eqm(self, t, state):
        # if np.ndim(t) == 0:
        #     t=np.array([t])
        # xyz_core1, xyz_core2 = self.evaluate_dense_cartesian_solution_at(t)
        xyz_core1 = np.zeros((3))
        # print(t, state, xyz_core1)
        # print(xyz_core1.shape, state.shape)
        state_d = np.zeros(state.shape)
        state_d[:3, ...] = state[3:, ...]
        state_d[3:, ...] = - self.grad_gravi_potential_feel(1., xyz_core1, state[:3, ...])
        # state_d[3:, ...] -= self.grad_gravi_potential_feel(self.M2_feel, xyz_core2, state[3:, ...])

        return state_d

    def some_initial_condition(self):
        return np.array([1, 0, 0, 0, 0, 1])

    def solve(self):
        return solve_ivp(fun=self._unit_test_mass_hamilton_eqm,
                         t_span=(0, 10),  # future
                         t_eval=np.linspace(0, 10, 100),
                         y0=self.some_initial_condition(),
                         method='RK23',
                         dense_output=False,
                         vectorized=False)


if __name__ == '__main__':
    pr = OrbitCartesianSolver()
    re = pr.solve()
    print(re)
