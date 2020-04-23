import os
import numpy as np
# from numpy import linspace
import multiprocessing


def run_process(process):
    print('start', process, multiprocessing.current_process())
    # x4
    # command = f'python3.8 -m tidaltailsim --nogui -q -s 60 -xlim -200 400 -ylim -240 240 -fr 25 -d2 -aw ffmpeg -ao ./mp4/{process[0]}.mp4 2galaxy_fromfile -c1 ./csv/{process[0]}.csv ./out/{process[0]}.pkl'
    # x2
    # command = f'python3.8 -m tidaltailsim --nogui -q -s 60 -xlim -150 250 -ylim -160 160 -fr 25 -d2 -aw ffmpeg -ao ./mp4/{process[0]}.mp4 2galaxy_fromfile -c1 ./csv/{process[0]}.csv ./out/{process[0]}.pkl'
    # f2
    # command = f'python3.8 -m tidaltailsim --nogui -q -s 60 -xlim -190 190 -ylim -152 152 -fr 25 -d2 -aw ffmpeg -ao ./mp4/{process[0]}.mp4 2galaxy_fromfile -c1 ./csv/{process[0]}.csv ./out/{process[0]}.pkl'
    # f4
    command = f'python3.8 -m tidaltailsim --nogui -q -s 60 -xlim -190 190 -ylim -152 152 -fr 25 -d2 -aw ffmpeg -ao ./mp4/{process[0]}.mp4 2galaxy_fromfile -c1 ./csv/{process[0]}.csv ./out/{process[0]}.pkl'
    print(f'run {command}')
    os.system(command)
    print('end', process)


if __name__ == '__main__':
    print('Prepare parallel rendering 2galaxy problem')

    # m2s = [.25, .5, 1., 2., 4.]
    # outputs_m = ['out_f4_{0:04.0f}.pkl', 'out_f2_{0:04.0f}.pkl', 'out_1_{0:04.0f}.pkl', 'out_x2_{0:04.0f}.pkl', 'out_x4_{0:04.0f}.pkl']
    outputs_m = ['out_f4_{0:04.0f}']

    # radii_space = np.linspace(2, 12, 10 * 4 + 1)
    radii_space = np.linspace(2, 12, 10 * 1 + 1)
    test_masses_count_space = radii_space * 100
    test_masses_count_space = test_masses_count_space.astype(np.dtype(int))

    processes = list()
    for o in outputs_m:
        for r, n in np.nditer((radii_space, test_masses_count_space)):
            out_str = o.format(r * 100)
            # m_str = f'-m 1. {m:.2f}'
            g1_str = f'-g1 {r:.2f} {n:d}'
            processes.append((out_str, g1_str))

    print('Start parallel rendering 2galaxy problem, num ex:', len(processes))
    pool = multiprocessing.Pool()
    pool.map(run_process, processes)
    print('End parallel rendering 2galaxy problem')
    exit()
# python -m tidaltailsim --nogui -v 2galaxy -n 89999 -g1 5 50 -o out0x2.pkl 1800 10 0
# python -m tidaltailsim --nogui -v 2galaxy -n 9999 -g1 5 50 -o out0x3.pkl 2000 10 0
