import numpy as np
from tidaltailsim.galaxy_orbital_toolkit import TestMassResultCategory
from csv import DictReader
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


plt.rcParams.update({'font.size': 8})
if __name__ == '__main__':
    print('Prepare rendering figures')

    cat_considered = [
        TestMassResultCategory.EllipticOrbitAboutGalaxy1,
        TestMassResultCategory.ParabolicOrbitAboutGalaxy1,
        TestMassResultCategory.HyperbolicOrbitAboutGalaxy1,
        TestMassResultCategory.BoundBadOrbitAboutGalaxy1,
        TestMassResultCategory.EscapingBadOrbitNearGalaxy1,
        TestMassResultCategory.EllipticOrbitAboutGalaxy2,
        TestMassResultCategory.ParabolicOrbitAboutGalaxy2,
        TestMassResultCategory.HyperbolicOrbitAboutGalaxy2,
        TestMassResultCategory.BoundBadOrbitAboutGalaxy2,
        TestMassResultCategory.EscapingBadOrbitNearGalaxy2,
    ]

    cat_to_ind = dict(zip(cat_considered, range(len(cat_considered))))

    m2s = [.25, .5, 2., 4.]
    outputs_m = ['out_f4_{0:04.0f}', 'out_f2_{0:04.0f}', 'out_x2_{0:04.0f}', 'out_x4_{0:04.0f}']

    radii_space_not1 = np.linspace(2, 12, 10 * 1 + 1)
    radii_indices_not1 = np.linspace(0, 10, 11, dtype=np.dtype(int))
    # test_masses_count_space = radii_space * 100
    # test_masses_count_space = test_masses_count_space.astype(np.dtype(int))

    counter_not1 = np.zeros((4, 10 * 1 + 1, len(cat_considered)))
    for o_ind, o in enumerate(outputs_m):
        for r_ind, r in np.nditer((radii_indices_not1, radii_space_not1)):
            test_masses_count = r * 100
            out_str = o.format(r * 100)
            # m_str = f'-m 1. {m:.2f}'
            # g1_str = f'-g1 {r:.2f} {n:d}'
            sub_counter = {cat: 0 for cat in cat_considered}
            with open(f'.\\ec2-out\\csv-preanalyse\\{out_str}.csv', 'r', newline='') as i:
                reader = DictReader(i)
                for row in reader:
                    flags = int(row['categories_flags'])
                    cat = TestMassResultCategory(flags)
                    # if cat not in cat_considered:
                    #     raise Exception(f'{out_str} has unclear category at test_mass_index {row["test_mass_index"]} => {flags}')
                    sub_counter[cat] += 1
            for cat, count in sub_counter.items():
                counter_not1[o_ind, r_ind, cat_to_ind[cat]] = count / test_masses_count

    # print(counter_not1)

    radii_space_is1 = np.linspace(2, 12, 10 * 4 + 1)
    radii_indices_is1 = np.linspace(0, 40, 41, dtype=np.dtype(int))
    # test_masses_count_space = radii_space * 100
    # test_masses_count_space = test_masses_count_space.astype(np.dtype(int))

    counter_is1 = np.zeros((10 * 4 + 1, len(cat_considered)))
    o = 'out_1_{0:04.0f}'
    for r_ind, r in np.nditer((radii_indices_is1, radii_space_is1)):
        test_masses_count = r * 100
        out_str = o.format(r * 100)
        # m_str = f'-m 1. {m:.2f}'
        # g1_str = f'-g1 {r:.2f} {n:d}'
        sub_counter = {cat: 0 for cat in cat_considered}
        with open(f'.\\ec2-out\\csv-preanalyse\\{out_str}.csv', 'r', newline='') as i:
            reader = DictReader(i)
            for row in reader:
                flags = int(row['categories_flags'])
                cat = TestMassResultCategory(flags)
                # if cat not in cat_considered:
                #     raise Exception(f'{out_str} has unclear category at test_mass_index {row["test_mass_index"]} => {flags}')
                sub_counter[cat] += 1
        for cat, count in sub_counter.items():
            counter_is1[r_ind, cat_to_ind[cat]] = count / test_masses_count

    # print(counter_is1)

    fig, ((axb, axg), (axr, axy))  = plt.subplots(2, 2, 'all', 'row', figsize=(6, 5))

    catind_to_color_ax = {
        (0, 3): ('royalblue', axb, 'Bounded to perturbed galaxy1'),
        (1, 2, 4): ('seagreen', axg, 'Tail of perturbed galaxy1'),
        (5, 6, 8): ('indianred', axr, 'Bounded to perturbing galaxy2'),
        (7, 9): ('darkgoldenrod', axy, 'Tail of perturbing galaxy2'),
    }

    for catind, (color, ax, desc) in catind_to_color_ax.items():
        ax.set_title(desc)
        for i, fmt, m_str in zip(range(0, 2), ('v:', 'v--'), ('1/4', '1/2')):
            ax.plot(radii_space_not1 / 10, np.sum(counter_not1[i, :, catind], axis=0), fmt, color=color, markersize=3.5)
        ax.plot(radii_space_is1 / 10, np.sum(counter_is1[:, catind], axis=1), '.-', color=color, markersize=3)
        for i, fmt, m_str in zip(range(2, 4), ('s--', 's:'), ('2', '4')):
            ax.plot(radii_space_not1 / 10, np.sum(counter_not1[i, :, catind], axis=0), fmt, color=color, markersize=3)

    for ax in (axr, axy):
        ax.set_xlabel('initial orbital radius\ndivided by distance of closest approach')

    for ax in (axb, axr):
        ax.set_ylabel('fraction of test masses')

    legend_artists = (mlines.Line2D([], [], color='black', marker='v', linestyle=':', label='$m_2 = 1/4$', markersize=3.5),
                      mlines.Line2D([], [], color='black', marker='v', linestyle='--', label='$m_2 = 1/2$', markersize=3.5),
                      mlines.Line2D([], [], color='black', marker='.', linestyle='-', label='$m_2 = 1$', markersize=3),
                      mlines.Line2D([], [], color='black', marker='s', linestyle='--', label='$m_2 = 2$', markersize=3),
                      mlines.Line2D([], [], color='black', marker='s', linestyle=':', label='$m_2 = 4$', markersize=3),)
    fig.legend(handles=legend_artists, loc='lower center', ncol=5)
    fig.set_tight_layout({
        'rect': (0, 0.05, 1, 1)
    })
    # print('showing')
    # plt.show()
    print('saving')
    fig.savefig('figs/disruption_curve.svg', format='svg', dpi=600)
