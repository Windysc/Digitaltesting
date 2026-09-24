"""
check_forced_aiding.py -- what a wrongly trusted COG does, and what the third COG test sees.

The T3b folders hold the synthetic 'raw' files with COG / SOG recomputed from the positions
(between consecutive fixes, over two steps, between moving averages of 3 to 15 fixes).  Each is
run three ways: default (source not declared), declared as receiver COG (the three tests decide),
and with SOG / COG forced into the smoother (no tests), which is what a plain "use COG" would do.

    python check_forced_aiding.py      (after: python check_mending_plan.py tests)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ais_prep as A            # noqa: E402
from check_mending_plan import OUT, grid_errors, load_fleet, q3, run     # noqa: E402


def main():
    fleet = load_fleet()
    cases = [('receiver COG (Doppler)', os.path.join(OUT, 'csv', 'raw')), ('receiver COG, one report per 60 s', os.path.join(OUT, 'csv', 'ds60')),
             ('course between consecutive fixes', os.path.join(OUT, 'csv_derived_1')), ('course over two steps', os.path.join(OUT, 'csv_derived_2'))]
    cases += [('course between moving averages of %d fixes' % m, os.path.join(OUT, 'csv_derived_%dma' % m)) for m in (3, 5, 9, 15)]
    lines = ['# A wrongly trusted COG, and the third COG test', '',
             '| COG and SOG columns | test 1 independent | test 2 lag [s] | test 3 holdout RMS with SOG/COG over positions only | '
             'accepted when declared | true position error, default [m] P50 / P95 / max | declared | forced into the smoother |',
             '|---|---|---|---|---|---|---|---|']
    for label, folder in cases:
        rep_d, int_d = run(folder, 'fa_default')
        rep_r, int_r = run(folder, 'fa_declared', cog_source='receiver')
        rep_f, int_f = run(folder, 'fa_forced', use_sogcog=True)
        ch = rep_r['smoothing']['choice']
        ratio = ch.get('holdout_rms_with_sogcog_over_positions_only')
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            label, ch.get('cog_independent'), ch.get('cog_lag_s'), '-' if ratio is None else '%.2f' % ratio, ch.get('cog_usable'),
            q3(grid_errors(int_d, fleet)['pos']), q3(grid_errors(int_r, fleet)['pos']), q3(grid_errors(int_f, fleet)['pos'])))
    print('\n'.join(lines))
    with open(os.path.join(OUT, 'results_forced_aiding.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
