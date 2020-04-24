import os
import numpy as np
# from numpy import linspace
#import multiprocessing
from csv import DictWriter, DictReader
from pprint import pprint


def preanalyse_csv(process):
    csv_filename = f'{process[0]}.csv'
    ambiguity_overall = list()
    should_review_overall = list()
    with open(f'.\\ec2-out\\csv\\{csv_filename}', newline='') as i, \
            open(f'.\\ec2-out\\csv-preanalyse\\{csv_filename}', 'w', newline='') as o:
        reader = DictReader(i)
        fieldnames = ['orbital_index', 'test_mass_index', 'categories_flags', 'old_categories_flags', 'flag1', 'flag2', 'is_parabolic1', 'is_parabolic2', 'is_boundbad1', 'is_boundbad2', 'ambiguity', 'should_review']
        writer = DictWriter(o, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            flags = int(row['categories_flags'])
            flag1 = flags & 31  # ie, int('11111', 2)
            flag2 = flags >> 5
            ambiguity = bool(flag1) and bool(flag2)
            is_parabolic1 = bool(flag1 & 2)
            is_parabolic2 = bool(flag2 & 2)
            is_boundbad1 = bool(flag1 & 8)
            is_boundbad2 = bool(flag2 & 8)
            should_review = is_parabolic1 or is_parabolic2 or is_boundbad1 or is_boundbad2
            writer.writerow({'orbital_index': int(row['orbital_index']),
                             'test_mass_index': int(row['test_mass_index']),
                             'categories_flags': flags,
                             'old_categories_flags': flags,
                             'flag1': flag1,
                             'flag2': flag2,
                             'is_parabolic1': is_parabolic1,
                             'is_parabolic2': is_parabolic2,
                             'is_boundbad1': is_boundbad1,
                             'is_boundbad2': is_boundbad2,
                             'ambiguity': ambiguity,
                             'should_review': should_review})
            if ambiguity:
                ambiguity_overall.append((int(row['orbital_index']), int(row['test_mass_index']), flags))
            if should_review:
                should_review_overall.append((int(row['orbital_index']), int(row['test_mass_index']), flags))
    return (ambiguity_overall, should_review_overall)


if __name__ == '__main__':
    print('Prepare pre-analysing csv 2galaxy problem')

    m2s = [.25, .5, 1., 2., 4.]
    outputs_m = ['out_f4_{0:04.0f}', 'out_f2_{0:04.0f}', 'out_1_{0:04.0f}', 'out_x2_{0:04.0f}', 'out_x4_{0:04.0f}']

    radii_space = np.linspace(2, 12, 10 * 4 + 1)
    test_masses_count_space = radii_space * 100
    test_masses_count_space = test_masses_count_space.astype(np.dtype(int))

    # processes = list()
    for m, o in zip(m2s, outputs_m):
        for r, n in np.nditer((radii_space, test_masses_count_space)):
            out_str = o.format(r * 100)
            m_str = f'-m 1. {m:.2f}'
            g1_str = f'-g1 {r:.2f} {n:d}'
            process = (out_str, m_str, g1_str)
            ambiguity_overall, should_review_overall = preanalyse_csv(process)
            if ambiguity_overall or should_review_overall:
                print(f'{out_str} pre-analysed, there is something to worry')
                print(f'{out_str} have ambiguity:')
                pprint(ambiguity_overall)
                print(f'{out_str} need review:')
                pprint(should_review_overall)
            else:
                print(f'{out_str} pre-analysed, nothing to worry')
    exit()
