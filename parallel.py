import os
import numpy as np
# from numpy import linspace
import multiprocessing


def run_process(process):
    print('start', process, multiprocessing.current_process())
    command = f'python -m tidaltailsim --nogui -v 2galaxy -n 9999 {process[2]} {process[1]} -o {process[0]} 2000 10 0'
    print(f'run {command}')
    # os.system(command)
    print('end', process)


if __name__ == '__main__':
    print('Prepare parallel processing 2galaxy problem')

    m2s = [.25, .5, 1., 2., 4.]
    outputs_m = ['out_f4_{0:04.0f}.pkl', 'out_f2_{0:04.0f}.pkl', 'out_1_{0:04.0f}.pkl', 'out_x2_{0:04.0f}.pkl', 'out_x4_{0:04.0f}.pkl']

    radii_space = np.linspace(2, 12, 10 * 4 + 1)
    test_masses_count_space = radii_space * 100
    test_masses_count_space = test_masses_count_space.astype(np.dtype(int))

    processes = list()
    for m, o in zip(m2s, outputs_m):
        for r, n in np.nditer((radii_space, test_masses_count_space)):
            out_str = o.format(r * 100)
            m_str = f'-m 1. {m:.2f}'
            g1_str = f'-g1 {r:.2f} {n:d}'
            processes.append((out_str, m_str, g1_str))

    print('Start parallel processing 2galaxy problem, num ex:', len(processes))
    pool = multiprocessing.Pool()
    pool.map(run_process, processes)
    print('End parallel processing 2galaxy problem')
    exit()
# python -m tidaltailsim --nogui -v 2galaxy -n 89999 -g1 5 50 -o out0x2.pkl 1800 10 0
# python -m tidaltailsim --nogui -v 2galaxy -n 9999 -g1 5 50 -o out0x3.pkl 2000 10 0
