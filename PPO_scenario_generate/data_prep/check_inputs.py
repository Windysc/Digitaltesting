"""
check_inputs.py -- does the chain survive the input variants a private AIS export may have?

Each case rewrites the synthetic 'raw' CSV files in one way and runs `prepare` on them:
no SOG / COG columns, SOG all zero, SOG in m/s, mixed SOG units, epoch-millisecond timestamps,
other column names, several vessels in one file, positions rounded to 1e-5 deg at low speed.

    python check_inputs.py      (after: python check_mending_plan.py fleet)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ais_prep as A            # noqa: E402
from check_mending_plan import OUT, grid_errors, load_fleet, q3     # noqa: E402

SRC = os.path.join(OUT, 'csv', 'raw')


def rewrite(name, fn, merge=False):
    dst = os.path.join(OUT, 'csv_in_' + name)
    os.makedirs(dst, exist_ok=True)
    frames = []
    for i, f in enumerate(sorted(os.listdir(SRC))):
        df = fn(pd.read_csv(os.path.join(SRC, f)), i)
        if merge:
            df['mmsi'] = 219000000 + i
            frames.append(df)
        else:
            df.to_csv(os.path.join(dst, f), index=False)
    if merge:
        pd.concat(frames).sample(frac=1.0, random_state=0).to_csv(os.path.join(dst, 'all_vessels.csv'), index=False)
    return dst


def main():
    fleet = load_fleet()
    epoch = lambda s: (pd.to_datetime(s, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(milliseconds=1)
    cases = [
        ('as generated', lambda d, i: d, {}, False),
        ('no SOG and no COG column', lambda d, i: d.drop(columns=['speed', 'cog']), {}, False),
        ('SOG all zero', lambda d, i: d.assign(speed=0.0), {}, False),
        ('SOG in m/s', lambda d, i: d.assign(speed=(d['speed'] * A.KN).round(2)), {}, False),
        ('SOG in knots for even files, m/s for odd files', lambda d, i: d if i % 2 == 0 else d.assign(speed=(d['speed'] * A.KN).round(2)), {}, False),
        ('timestamps as epoch milliseconds', lambda d, i: d.assign(timestamp=epoch(d['timestamp'])), {}, False),
        ('columns named BaseDateTime, LON, LAT, SOG, COG', lambda d, i: d.rename(columns=dict(
            timestamp='BaseDateTime', longitude_degrees='LON', latitude_degrees='LAT', speed='SOG', cog='COG')), {}, False),
        ('24 vessels shuffled into one file with an mmsi column', lambda d, i: d, {}, True),
        ('positions rounded to 1e-5 deg', lambda d, i: d.assign(longitude_degrees=d['longitude_degrees'].round(5),
                                                                latitude_degrees=d['latitude_degrees'].round(5)), {}, False),
    ]
    print('| input variant | runs | voyages | windows | SOG units found | grouped by | stale repeats dropped | q used | aiding | '
          'true grid position error [m] P50 / P95 / max |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    lines = []
    for label, fn, kw, merge in cases:
        folder = rewrite(label.split()[0] + str(len(label)), fn, merge)
        try:
            rep, internals = A.prepare([folder], os.path.join(OUT, 'prep_inputs'), keep=True, cog_source='receiver', **kw)
            for v in internals['voyages']:                               # map merged vessels back to their truth
                if merge:
                    v['source'] = 'voyage_%02d.csv' % (int(v['vessel']) - 219000000)
            E = grid_errors(internals, fleet)
            row = [label, 'yes', rep['counts']['voyages'], rep['counts']['windows'], rep['sog_cog']['sog_units_detected'],
                   rep['split']['grouped_by'], rep['counts']['stale_repeats'], '%g' % rep['smoothing']['q_used'],
                   'on' if rep['sog_cog']['velocity_aiding'] else 'off', q3(E['pos'])]
        except Exception as e:                                           # a crash is a finding, keep going
            row = [label, 'NO: %s: %s' % (type(e).__name__, str(e)[:80]), '', '', '', '', '', '', '', '']
        line = '| ' + ' | '.join(str(x) for x in row) + ' |'
        print(line)
        lines.append(line)
    with open(os.path.join(OUT, 'results_inputs.md'), 'w', encoding='utf-8') as fh:
        fh.write('# Input variants\n\n| input variant | runs | voyages | windows | SOG units found | grouped by | stale repeats dropped | '
                 'q used | aiding | true grid position error [m] P50 / P95 / max |\n|---|---|---|---|---|---|---|---|---|---|\n'
                 + '\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
