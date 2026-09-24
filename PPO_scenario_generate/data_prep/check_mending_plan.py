"""
check_mending_plan.py -- tests of the mending plan for part 1 (data sampling and
smoothing) against synthetic AIS-like voyages whose truth is known.

Run order
    python check_mending_plan.py fleet            # write the synthetic CSV files
    <python with scipy> legacy_stage.py check_out/csv check_out/legacy.npz
    python check_mending_plan.py tests            # T0-T12, writes check_out/results.md and results.json

The legacy arrays come from the original function of data_csv2npy.py, the
environment parts call the real scenario_targets / Ship_envre_v2 code; only the
TSGM noise is reproduced from its source line (tsgm 0.1.0: sigma = variance ** 0.5,
np.random.normal per point and per feature), because tsgm does not install here.

The development fleet (seed 7) is the one the rules were worked out on.  T12 runs
the finished chain once on a second fleet with other seeds and harsher error
regimes; nothing was tuned on it.
"""
import datetime
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Ship_envre_v2: next to PPO_scenario_generate in the repository, else the Desktop checkout
V2 = next((p for p in (os.path.join(os.path.dirname(ROOT), 'Ship_envre_v2'),
                       r'C:\Users\ASUS\Desktop\Digitaltesting-main\Digitaltesting-main\Ship_envre_v2')
           if os.path.isdir(p)), os.path.join(os.path.dirname(ROOT), 'Ship_envre_v2'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import ais_prep as A            # noqa: E402
import synth_ais as S           # noqa: E402

OUT = os.path.join(HERE, 'check_out')
REGIMES = ('raw', 'ds30', 'ds60')
QS = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)
LINES = []
RES = {}


def say(*a):
    line = ' '.join(str(x) for x in a)
    print(line)
    LINES.append(line)


def table(header, rows):
    say('| ' + ' | '.join(header) + ' |')
    say('|' + '---|' * len(header))
    for r in rows:
        say('| ' + ' | '.join(str(x) for x in r) + ' |')
    say('')


def q3(a, fmt='%.1f'):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return '-'
    return ' / '.join(fmt % np.percentile(a, p) for p in (50, 95, 100))


def fd_kin(xy, dt):
    """Speed, course [deg, math] and turn rate [deg/s] that a positions-only array encodes."""
    d = np.diff(xy, axis=0)
    sp = np.linalg.norm(d, axis=1) / dt
    hd = np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))
    return sp, hd, np.diff(hd) / dt


def load_fleet():
    fleet = S.make_fleet(os.path.join(OUT, 'csv'))          # same seed -> same truth as the files on disk
    return {tr['name']: tr for tr in fleet}


def to_truth(xy, v, tr):
    return A.ll_to_xy(*A.xy_to_ll(xy, v['lon0'], v['lat0']).T, tr['lon0'], tr['lat0'])


def grid_errors(internals, fleet, positions=None):
    """Errors of grid samples against the truth, pooled over segments.  positions(sg) may supply other
    positions on the same grid (another smoother)."""
    dt = internals['cfg']['grid_dt']
    E = dict(pos=[], speed=[], course=[], rate=[], rate_turn=[], rate_straight=[], pos_turn=[], pos_straight=[], ratio=[])
    for sg in internals['segments']:
        if 'grid_pos' not in sg or len(sg['tg']) < 8:
            continue
        v = internals['voyages'][sg['voyage']]
        tr = fleet[v['source'][:-4]]
        pt, _, _, rt = S.truth_at(tr, sg['tg'] - S.EPOCH0)
        pg = to_truth(sg['grid_pos'] if positions is None else positions(sg), v, tr)
        e = np.linalg.norm(pg - pt, axis=1)
        turning = np.abs(rt) > 0.05
        E['pos'].append(e); E['pos_turn'].append(e[turning]); E['pos_straight'].append(e[~turning])
        s1, c1, r1 = fd_kin(pg, dt)
        s0, c0, r0 = fd_kin(pt, dt)
        E['speed'].append(np.abs(s1 - s0)); E['course'].append(np.abs(A.wrap180(c1 - c0)))
        E['ratio'].append(s1 / np.maximum(s0, 1e-6))
        er = np.abs(r1 - r0)
        E['rate'].append(er)
        tm = turning[1:-1]
        E['rate_turn'].append(er[tm]); E['rate_straight'].append(er[~tm])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in E.items()}


def run(folder, name, **kw):
    return A.prepare([folder], os.path.join(OUT, 'prep_' + name), keep=True, **kw)


# ----------------------------------------------------------------- T0 / T1
def t0_t1(fleet, legacy, prep):
    say('## T0 What the row-index resampling did to time (original function of data_csv2npy.py)')
    say('')
    rows = []
    for reg in REGIMES:
        mn, md, mx, ratio, f20, f9 = [], [], [], [], [], []
        for name, tr in fleet.items():
            t = tr['reports'][reg]['t_true']
            d = np.diff(np.interp(np.linspace(0, len(t) - 1, 100), np.arange(len(t)), t))
            mn.append(d.min()); md.append(np.median(d)); mx.append(d.max()); ratio.append(d.max() / d.min())
            f20.append(np.median(d) / 20.0); f9.append(np.median(d) / 9.0)
        rows.append([reg, '%.0f' % np.median(mn), '%.0f' % np.median(md), '%.0f' % np.median(mx),
                     '%.0f (worst %.0f)' % (np.median(ratio), np.max(ratio)), '%.2f' % np.median(f20), '%.2f' % np.median(f9)])
    table(['regime', 'shortest step [s]', 'median step [s]', 'longest step [s]', 'longest / shortest',
           'median step / 20 s', 'median step / 9 s'], rows)
    say('Each of the 100 legacy points stands for a different time step. A consumer that reads the array with one '
        'fixed step sees speeds multiplied by the last two columns on average and by the longest / shortest ratio within a trace.')
    say('')
    say('## T1 Legacy arrays against mended arrays, both against the truth')
    say('')
    rows = []
    for reg in REGIMES:
        P, Pu, SR, S20, CE, RE = [], [], [], [], [], []
        for name, tr in fleet.items():
            arr = legacy['%s/%s' % (reg, name)]
            t = tr['reports'][reg]['t_true']
            tk = np.interp(np.linspace(0, len(t) - 1, 100), np.arange(len(t)), t)
            p = A.ll_to_xy(arr[:, 0], arr[:, 1], tr['lon0'], tr['lat0'])
            pt, spt, _, _ = S.truth_at(tr, tk)
            P.append(np.linalg.norm(p - pt, axis=1))
            dtu = (tk[-1] - tk[0]) / 99.0
            ptu = S.truth_at(tr, tk[0] + dtu * np.arange(100))[0]
            Pu.append(np.linalg.norm(p - ptu, axis=1))
            s1, c1, r1 = fd_kin(p, dtu)
            s0, c0, r0 = fd_kin(ptu, dtu)
            SR.append(s1 / np.maximum(s0, 1e-6)); CE.append(np.abs(A.wrap180(c1 - c0))); RE.append(np.abs(r1 - r0))
            S20.append(np.median(np.linalg.norm(np.diff(p, axis=0), axis=1) / 20.0) / np.median(spt))
        sr = np.concatenate(SR)
        rows.append([reg + ' legacy', q3(np.concatenate(P)), q3(np.concatenate(Pu), '%.0f'),
                     '%.2f to %.2f' % (np.percentile(sr, 5), np.percentile(sr, 95)), '%.2f' % np.median(S20),
                     q3(np.concatenate(CE)), q3(np.concatenate(RE), '%.2f')])
        for label, mode in ((' mended, positions only', 'off'), (' mended, receiver SOG/COG declared', 'on')):
            E = grid_errors(prep[reg][mode][1], fleet)
            rows.append([reg + label, q3(E['pos']), q3(E['pos'], '%.0f'), '%.2f to %.2f' % (
                np.percentile(E['ratio'], 5), np.percentile(E['ratio'], 95)), '1.00', q3(E['course']), q3(E['rate'], '%.2f')])
            RES['T1_%s_%s' % (reg, mode)] = dict(pos=q3(E['pos']), course=q3(E['course']), rate=q3(E['rate'], '%.3f'))
        RES['T1_%s_legacy' % reg] = dict(pos_true_time=q3(np.concatenate(P)), pos_uniform=q3(np.concatenate(Pu)),
                                         speed_ratio=[float(np.percentile(sr, 5)), float(np.percentile(sr, 95))])
    table(['chain', 'position error at the time the point really stands for [m] (P50 / P95 / max)',
           'position error when read with a uniform step [m]', 'implied speed / true speed (P5 to P95)',
           'speed factor when read at 20 s', 'course error [deg]', 'turn-rate error [deg/s]'], rows)


# ----------------------------------------------------------------------- T2
def savgol(y, win, order=2):
    n, h = len(y), win // 2
    out = np.empty_like(y)
    c = np.linalg.pinv(np.vander(np.arange(-h, h + 1), order + 1, increasing=True))[0]
    for j in range(y.shape[1]):
        out[h:n - h, j] = np.convolve(y[:, j], c[::-1], mode='valid')
        for i in list(range(h)) + list(range(n - h, n)):
            lo, hi = max(0, i - h), min(n, i + h + 1)
            out[i, j] = np.polyval(np.polyfit(np.arange(lo, hi), y[lo:hi, j], order), i)
    return out


def t2(fleet, prep):
    say('## T2 Which smoother: errors of the 20 s arrays against the truth (finite differences of positions, the view a generator has)')
    say('')
    for reg in ('raw', 'ds60'):
        int_off, int_on = prep[reg]['off'][1], prep[reg]['on'][1]

        def linear(sg):
            tu, xu = sg['t'][sg['use']], sg['xy'][sg['use']]
            return np.stack([np.interp(sg['tg'], tu, xu[:, 0]), np.interp(sg['tg'], tu, xu[:, 1])], axis=1)
        cases = [('linear on timestamps', grid_errors(int_off, fleet, linear)),
                 ('linear + Savitzky-Golay 100 s', grid_errors(int_off, fleet, lambda sg: savgol(linear(sg), 5))),
                 ('RTS, positions only', grid_errors(int_off, fleet)),
                 ('RTS with receiver SOG/COG', grid_errors(int_on, fleet))]
        rows = []
        for name, E in cases:
            rows.append([name, q3(E['pos']), q3(E['pos_turn']), '%.2f' % np.percentile(E['speed'], 95),
                         '%.2f' % np.percentile(E['course'], 95), '%.3f' % np.percentile(E['rate_straight'], 95),
                         '%.3f' % np.percentile(E['rate_turn'], 95)])
            RES['T2_%s_%s' % (reg, name)] = dict(pos_p95=float(np.percentile(E['pos'], 95)),
                                                 rate_p95_straight=float(np.percentile(E['rate_straight'], 95)),
                                                 rate_p95_turning=float(np.percentile(E['rate_turn'], 95)))
        say('Regime %s (q used: positions only %g, with SOG/COG %g; aiding on: %s)' % (
            reg, int_off['cfg']['q_used'], int_on['cfg']['q_used'], prep[reg]['on'][0]['sog_cog']['velocity_aiding']))
        say('')
        table(['smoother', 'position error [m] P50 / P95 / max', 'same, while turning', 'speed error P95 [m/s]',
               'course error P95 [deg]', 'turn-rate error P95 on straights [deg/s]', 'turn-rate error P95 in turns [deg/s]'], rows)
    say('True turn rates in this fleet are 0.15-0.5 deg/s, so a turn-rate error of 0.1 deg/s or more on straights means the '
        'array shows manoeuvres that did not happen.')
    say('')


# ----------------------------------------------------------------------- T3
def truth_by_q(internals, fleet, aided):
    out = []
    for q in QS:
        pos, crs, rate = [], [], []
        for sg in internals['segments']:
            if 'grid_pos' not in sg or len(sg['tg']) < 8:
                continue
            v = internals['voyages'][sg['voyage']]
            tr = fleet[v['source'][:-4]]
            sm = A.rts_smooth(sg['t'], sg['xy'], sg['tg'], q, internals['cfg']['sigma_p'],
                              sg['vel_meas'] if aided else None, internals['cfg']['sigma_v'], sg['use'])
            pt = S.truth_at(tr, sg['tg'] - S.EPOCH0)[0]
            pg = to_truth(sm['pos'], v, tr)
            pos.append(np.linalg.norm(pg - pt, axis=1))
            s1, c1, r1 = fd_kin(pg, 20.0); s0, c0, r0 = fd_kin(pt, 20.0)
            crs.append(A.wrap180(c1 - c0)); rate.append(r1 - r0)
        f = lambda a: float(np.sqrt(np.mean(np.concatenate(a) ** 2)))
        out.append((f(pos), f(crs), f(rate)))
    out = np.array(out)
    return out, (out / out.min(axis=0)).mean(axis=1)


def t3(fleet, prep):
    say('## T3 Smoothing strength q: what the data can tell (holdout, COG) against what the truth prefers')
    say('')
    say('The truth columns are RMS errors of the 20 s arrays; "composite" is the mean of the three, each divided by its own '
        'best value, so 1.000 marks the q that serves position, course and turn rate best together.')
    say('')
    white = os.path.join(OUT, 'csv_white10')
    rng = np.random.default_rng(99)
    for k, (name, tr) in enumerate(fleet.items()):
        rep = S.sample_reports(np.random.default_rng(rng.integers(1 << 30)), tr, 'raw', gaps=tr['gaps'], white_only=10.0)
        S.write_csv(os.path.join(white, name + '.csv'), rep, tr['lon0'], tr['lat0'], rng, mmsi=219000001 + k)
    cases = [('Gauss-Markov 3 m + white 2 m, as received', prep['raw']['on']),
             ('same noise, one report per 30 s', prep['ds30']['on']), ('same noise, one report per 60 s', prep['ds60']['on']),
             ('white 10 m (the assumption behind sigma_p = 10 m), as received', run(white, 'white10', cog_source='receiver'))]
    summary = []
    for label, (rep, internals) in cases:
        tru, comp = truth_by_q(internals, fleet, False)
        ch = rep['smoothing']['choice']
        cv = {r['q']: r for r in rep['smoothing']['table']}
        plain = min(rep['smoothing']['table'], key=lambda r: r['single_rms'] + r['block_rms'])['q']
        best = [QS[int(np.argmin(tru[:, k]))] for k in range(3)] + [QS[int(np.argmin(comp))]]
        say('%s. Truth prefers q = %g (position), %g (course), %g (turn rate), %g (composite). Plain holdout minimum %g; '
            'holdout with ties to the smaller q %g; COG choice %g; chain uses %g without a declared COG source and %g with '
            'receiver COG declared.' % (label, best[0], best[1], best[2], best[3], plain, ch['q_holdout'], ch['q_cog'],
                                        float(min(max(ch['q_holdout'], 1e-3), 1e-2)), rep['smoothing']['q_used']))
        say('')
        rows = []
        for q, (p, c, r), k in zip(QS, tru, comp):
            h = cv.get(q)
            rows.append(['%g' % q, '%.2f' % p, '%.3f' % c, '%.4f' % r, '%.3f' % k,
                         ('%.2f / %.2f' % (h['single_rms'], h['block_rms'])) if h else '-',
                         ('%.2f / %.3f' % (h['course_vs_cog_rms_deg'], h['speed_vs_sog_rms_ms'])) if h and 'course_vs_cog_rms_deg' in h else '-'])
        table(['q [m^2/s^3]', 'true position RMS [m]', 'true course RMS [deg]', 'true turn-rate RMS [deg/s]', 'composite',
               'holdout RMS single / block [m]', 'positions-only smooth against COG [deg] / SOG [m/s], RMS'], rows)
        qi = {q: i for i, q in enumerate(QS)}
        cost = lambda q: comp[qi[q]]
        summary.append([label, '%g' % best[3], '%g (%.3f)' % (plain, cost(plain)), '%g (%.3f)' % (ch['q_holdout'], cost(ch['q_holdout'])),
                        '%g (%.3f)' % (float(min(max(ch['q_holdout'], 1e-3), 1e-2)), cost(float(min(max(ch['q_holdout'], 1e-3), 1e-2)))),
                        '%g (%.3f)' % (rep['smoothing']['q_used'], cost(rep['smoothing']['q_used'])), '%g (%.3f)' % (3e-3, cost(3e-3))])
        RES['T3_' + label[:20]] = dict(truth=best, plain=plain, holdout=ch['q_holdout'], cog=ch['q_cog'], used=rep['smoothing']['q_used'])
    say('Summary: q chosen by each rule, with the composite cost against the truth in brackets (1.000 is the best possible)')
    say('')
    table(['case', 'truth-preferred q', 'plain holdout minimum', 'holdout, ties to the smaller q', 'same, inside the bounds 0.001 to 0.01',
           'with receiver COG declared (smaller of holdout and COG, inside the bounds)', 'fixed q = 0.003'], summary)


def t3b(fleet):
    say('## T3b Can the data tell a receiver COG from one that was computed from the positions?')
    say('')
    import pandas as pd
    src = os.path.join(OUT, 'csv', 'raw')
    kinds = [('receiver COG (Doppler), as in the other tests', None), ('course between consecutive fixes', 1),
             ('course over two steps', 2), ('course between moving averages of 3 fixes', -3),
             ('course between moving averages of 5 fixes', -5), ('course between moving averages of 9 fixes', -9),
             ('course between moving averages of 15 fixes', -15)]
    rows = []
    for label, kind in kinds:
        folder = src if kind is None else os.path.join(OUT, 'csv_derived_%s' % abs(kind) + ('ma' if kind < 0 else ''))
        if kind is not None:
            os.makedirs(folder, exist_ok=True)
            for name in sorted(os.listdir(src)):
                df = pd.read_csv(os.path.join(src, name))
                xy = A.ll_to_xy(df['longitude_degrees'].to_numpy(), df['latitude_degrees'].to_numpy(),
                                df['longitude_degrees'][0], df['latitude_degrees'][0])
                t = A._parse_time(df['timestamp'])
                if kind > 0:
                    d = xy[kind:] - xy[:-kind]
                    d = np.concatenate([np.repeat(d[:1], kind, 0), d])
                    dt = np.concatenate([np.repeat(t[kind] - t[0], kind), t[kind:] - t[:-kind]])
                else:
                    m = -kind
                    k = np.ones(m) / m
                    pad = np.concatenate([np.repeat(xy[:1], m // 2, 0), xy, np.repeat(xy[-1:], m // 2, 0)])
                    sm = np.stack([np.convolve(pad[:, j], k, 'valid') for j in range(2)], 1)
                    d = np.concatenate([sm[1:2] - sm[0:1], sm[1:] - sm[:-1]])
                    dt = np.concatenate([[t[1] - t[0]], np.diff(t)])
                df['cog'] = np.round(np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0, 1)
                df['speed'] = np.round(np.linalg.norm(d, axis=1) / np.maximum(dt, 1.0) / A.KN, 1)
                df.to_csv(os.path.join(folder, name), index=False)
        rep, internals = run(folder, 't3b', cog_source='receiver')
        ch = rep['smoothing']['choice']
        E = grid_errors(internals, fleet)
        rep_u, int_u = run(folder, 't3b_u')
        Eu = grid_errors(int_u, fleet)
        rows.append([label, '%.2f' % ch['cog_vs_raw_position_course_rms_deg'], '%.2f' % ch['cog_raw_correlation_on_straights'],
                     ch['cog_independent'], '%g' % ch['q_cog'], '%g' % ch['q_holdout'],
                     '%g, aiding %s: %s' % (rep['smoothing']['q_used'], 'on' if rep['sog_cog']['velocity_aiding'] else 'off', q3(E['pos'])),
                     '%g, aiding off: %s' % (rep_u['smoothing']['q_used'], q3(Eu['pos']))])
        RES['T3b_' + label[:24]] = dict(independent=ch['cog_independent'], declared=q3(E['pos']), undeclared=q3(Eu['pos']))
    table(['COG and SOG columns', 'COG against the raw fix-to-fix course on straights, RMS [deg]', 'correlation of the two deviations',
           'test says independent', 'q by COG', 'q by holdout', 'if the owner declares receiver COG: q, aiding, true position error [m] P50 / P95 / max',
           'with the source left undeclared (default)'], rows)
    say('The test catches a COG taken between raw fixes. It cannot catch one taken from smoothed positions, so a pass is no '
        'proof. That is why COG counts only when the owner declares it a receiver value, and why the default leaves it out.')
    say('')


def t3c(fleet):
    say('## T3c A real receiver COG lags and gets noisy at low speed: does its use survive that?')
    say('')
    import pandas as pd
    rows = []
    for regime in ('raw', 'ds60'):
        for lag, sv in ((0.0, 0.05), (5.0, 0.10), (10.0, 0.10), (20.0, 0.15)):
            folder = os.path.join(OUT, 'csv_cog_%s_%g' % (regime, lag))
            os.makedirs(folder, exist_ok=True)
            rng = np.random.default_rng(int(lag * 10 + sv * 1000))
            for name, tr in fleet.items():
                df = pd.read_csv(os.path.join(OUT, 'csv', regime, name + '.csv'))
                t = tr['reports'][regime]['t_true']
                c = np.radians(tr['course'])
                vx, vy = tr['speed'] * np.sin(c), tr['speed'] * np.cos(c)
                if lag > 0:                             # first-order low-pass of the velocity, as a receiver does
                    fx, fy = vx.copy(), vy.copy()
                    for k in range(1, len(vx)):
                        fx[k] = fx[k - 1] + (vx[k] - fx[k - 1]) / lag
                        fy[k] = fy[k - 1] + (vy[k] - fy[k - 1]) / lag
                    vx, vy = fx, fy
                ex = np.interp(t, tr['t'], vx) + rng.normal(0, sv, len(t))
                ey = np.interp(t, tr['t'], vy) + rng.normal(0, sv, len(t))
                df['speed'] = np.round(np.hypot(ex, ey) / A.KN, 1)
                df['cog'] = np.round(np.degrees(np.arctan2(ex, ey)) % 360.0, 1)
                df.to_csv(os.path.join(folder, name + '.csv'), index=False)
            rep, internals = run(folder, 't3c', cog_source='receiver')
            ch = rep['smoothing']['choice']
            pos, comp_p = truth_by_q(internals, fleet, False)
            aid, comp_a = truth_by_q(internals, fleet, True)
            qi = QS.index(rep['smoothing']['q_used'])
            on = rep['sog_cog']['velocity_aiding']
            rows.append([regime, '%g s, %.2f m/s' % (lag, sv), '%g' % ch['cog_lag_s'], ch['cog_usable'], '%g' % ch['q_cog'],
                         '%g' % ch['q_holdout'], '%g' % rep['smoothing']['q_used'], '%.2f / %.2f' % (pos[qi, 0], aid[qi, 0]),
                         'on' if on else 'off', '%.2f' % (aid if on else pos)[qi, 0], '%.2f' % min(pos[:, 0].min(), aid[:, 0].min())])
    table(['regime', 'true COG lag, velocity noise', 'estimated lag [s]', 'COG accepted', 'q by COG', 'q by holdout', 'q used',
           'true position RMS [m] at the used q: positions only / with SOG and COG', 'aiding chosen',
           'true RMS of the chain as run [m]', 'best true RMS over all q and both modes [m]'], rows)
    say('A COG that trails the ship by 10 s brings nothing and one that trails by 20 s makes the aided track worse, so the '
        'chain takes SOG / COG only when the estimated lag is at most 5 s.')
    say('')


# ----------------------------------------------------------------------- T4
def t4(prep):
    q = prep['raw']['off'][1]['cfg']['q_used']
    say('## T4 How long a reception gap may be bridged: error inside the gap against the truth (reported timestamps, q = %g as chosen by the chain)' % q)
    say('')
    rng = np.random.default_rng(5)
    voys = []
    for i in range(30):
        tr = S.make_truth(rng, 2400.0, turns=0)
        rate = rng.uniform(0.2, 0.5) * rng.choice([-1.0, 1.0])
        dur = int(rng.uniform(40.0, 100.0) / abs(rate))
        r_cmd = np.zeros(2400)
        r_cmd[1200 - dur // 2:1200 + dur // 2] = rate
        r = np.zeros(2400)
        for k in range(1, 2400):
            r[k] = r[k - 1] + (r_cmd[k] - r[k - 1]) / 20.0
        tr['rate'], tr['course'] = r, tr['course'][0] + np.cumsum(r)
        c = np.radians(tr['course'])
        tr['x'] = np.concatenate([[0.0], np.cumsum(tr['speed'] * np.sin(c))[:-1]])
        tr['y'] = np.concatenate([[0.0], np.cumsum(tr['speed'] * np.cos(c))[:-1]])
        voys.append(tr)
    rows = []
    for G in (30, 60, 90, 120, 180, 300, 600):
        res = {}
        for i, tr in enumerate(voys):
            for where, tc in (('straight', 500.0), ('turn', 1200.0)):
                rep = S.sample_reports(np.random.default_rng(1000 + i), tr, 'raw', gaps=[(tc - G / 2.0, float(G))])
                t = rep['t_rep'] - S.EPOCH0                                   # what the chain sees
                t, first = np.unique(t, return_index=True)
                xy = rep['pos'][first]
                vel = A.velocity_from_sogcog(rep['sog_kn'][first] * A.KN, rep['cog'][first])
                te = np.arange(tc - G / 2.0 + 2.5, tc + G / 2.0, 5.0)
                pt = S.truth_at(tr, te)[0]
                lin = np.stack([np.interp(te, t, xy[:, 0]), np.interp(te, t, xy[:, 1])], axis=1)
                for key, est in ((where + ' rts', A.rts_smooth(t, xy, te, q, 10.0)['pos']),
                                 (where + ' rts+vel', A.rts_smooth(t, xy, te, q, 10.0, vel, 0.2)['pos']), (where + ' linear', lin)):
                    res.setdefault(key, []).append(np.linalg.norm(est - pt, axis=1))
        row = [G]
        for k in ('straight rts', 'straight linear', 'turn rts', 'turn rts+vel', 'turn linear'):
            e = np.concatenate(res[k])
            row.append('%.0f / %.0f' % (np.percentile(e, 50), np.percentile(e, 95)))
            RES['T4_%d_%s' % (G, k)] = float(np.percentile(e, 95))
        rows.append(row)
    table(['gap [s]', 'on a straight, RTS [m] P50 / P95', 'on a straight, linear', 'gap hides part of a turn, RTS',
           'same, RTS with SOG/COG', 'same, linear'], rows)
    say('What happens inside a gap that hides part of a turn is lost, and the error there grows roughly with the '
        'square of the gap length. These figures belong to this fleet (turns of 0.2 to 0.5 deg/s); they size the effect and '
        'are no general bound.')
    say('')


# ----------------------------------------------------------------------- T5
def t5(fleet):
    say('## T5 Outlier rules under a wider fault model')
    say('')
    say('Dirty regime: 0.4 % isolated bad fixes of 150 to 3000 m, one run of two bad fixes, one run of 3 to 10 fixes with a '
        'lasting offset of 300 to 800 m, 1 % duplicated rows, 5 % of reports without SOG, 1 % with the AIS "not available" '
        'values (102.3 kn, 360 deg).')
    say('')
    rows = []
    for label, on in (('rules on', True), ('rules off', False)):
        rep, internals = run(os.path.join(OUT, 'csv', 'dirty'), 'dirty_' + ('on' if on else 'off'), outlier_rules=on)
        inj = hit = false = 0
        for vi, v in enumerate(internals['voyages']):
            r = fleet[v['source'][:-4]]['reports']['dirty']
            bad_t = set(np.unique(r['t_rep'][r['is_outlier']]).tolist())
            used = set(np.concatenate([sg['t'][sg['use']] for sg in internals['segments'] if sg['voyage'] == vi] or [np.zeros(0)]).tolist())
            removed = set(np.asarray(v['removed_t']).tolist()) - used
            inj += len(bad_t); hit += len(bad_t & used); false += len(removed - bad_t)
        E = grid_errors(internals, fleet)
        c = rep['counts']
        rows.append([label, inj, hit, false, c['removed_by_distance_rule'], c['removed_by_residual_pass'], c['cuts_at_lasting_jumps'],
                     c['sog_or_cog_not_available'], rep['sog_cog']['sog_units_detected'], q3(E['pos'])])
        RES['T5_' + label] = dict(injected=inj, still_used=hit, false_removals=false, pos=q3(E['pos']))
    table(['setting', 'bad fixes injected', 'of those still used', 'good fixes removed', 'by the distance rule', 'by the residual pass',
           'cuts at lasting jumps', 'SOG / COG values not available', 'SOG units detected', 'grid position error [m] P50 / P95 / max'], rows)
    say('Hard manoeuvres must not be mistaken for bad fixes: a second fleet turns at 0.8 to 1.5 deg/s (small craft), with no bad '
        'fixes injected.')
    say('')
    hard = S.make_fleet(os.path.join(OUT, 'csv_hard'), n_voyages=16, seed=21, regimes=('raw',), rate_range=(0.8, 1.5))
    hf = {tr['name']: tr for tr in hard}
    rep, internals = run(os.path.join(OUT, 'csv_hard', 'raw'), 'hard')
    E = grid_errors(internals, hf)
    c = rep['counts']
    table(['fleet', 'reports', 'removed by the distance rule', 'removed by the residual pass', 'cuts', 'q used',
           'grid position error [m] P50 / P95 / max', 'same, while turning'],
          [['turns of 0.8 to 1.5 deg/s', c['raw_reports'], c['removed_by_distance_rule'], c['removed_by_residual_pass'],
            c['cuts_at_lasting_jumps'], '%g' % rep['smoothing']['q_used'], q3(E['pos']), q3(E['pos_turn'])]])
    RES['T5_hard'] = dict(removed=c['removed_by_distance_rule'] + c['removed_by_residual_pass'], pos=q3(E['pos']), pos_turn=q3(E['pos_turn']))
    say('Limits of this test: offsets below the distance gate (about 170 m at a 10 s interval for a 6 m/s ship) pass as small '
        'lateral bends; runs longer than 12 fixes or 180 s are cut out as separate segments instead of being removed.')
    say('')


# ----------------------------------------------------------------------- T6
def t6():
    say('## T6 Two ships of one encounter: is the closest approach preserved?')
    say('')
    pair = S.make_pair(os.path.join(OUT, 'csv_pair'))
    rep, internals = run(os.path.join(OUT, 'csv_pair', 'crossing', '1'), 'pair', pairs=True, window_points=60, stride=10)
    tracks = {}
    for sg in internals['segments']:
        v = internals['voyages'][sg['voyage']]
        tracks[v['source']] = (sg['tg'], A.ll_to_xy(*A.xy_to_ll(sg['grid_pos'], v['lon0'], v['lat0']).T, pair['lon0'], pair['lat0']))
    (ta, pa), (tb, pb) = tracks['1.csv'], tracks['2.csv']
    common = np.intersect1d(ta, tb)
    d = np.linalg.norm(pa[np.searchsorted(ta, common)] - pb[np.searchsorted(tb, common)], axis=1)
    rows = [['truth', '%.0f' % pair['dcpa'], '%.0f' % pair['tcpa'], '-'],
            ['mended, common absolute 20 s grid', '%.0f' % d.min(), '%.0f' % (common[int(np.argmin(d))] - S.EPOCH0),
             '%d common samples; both files in split %s' % (len(common), sorted({v['split'] for v in internals['voyages']}))]]
    arrs, spans = [], []
    for r in pair['reports']:
        t = r['t_true']
        idx = np.linspace(0, len(t) - 1, 100)
        arrs.append(np.stack([np.interp(idx, np.arange(len(t)), r['pos'][:, 0]), np.interp(idx, np.arange(len(t)), r['pos'][:, 1])], axis=1))
        spans.append(np.interp(idx, np.arange(len(t)), t))
    dl = np.linalg.norm(arrs[0] - arrs[1], axis=1)
    j = int(np.argmin(dl))
    rows.append(['legacy, each ship to 100 points by row index, paired by index', '%.0f' % dl.min(),
                 'ship 1 at %.0f s, ship 2 at %.0f s' % (spans[0][j], spans[1][j]),
                 'the two clocks differ by up to %.0f s' % np.max(np.abs(spans[0] - spans[1]))])
    table(['chain', 'closest approach [m]', 'time of closest approach [s]', 'note'], rows)
    RES['T6'] = dict(truth=pair['dcpa'], mended=float(d.min()), legacy=float(dl.min()))


# ----------------------------------------------------------------------- T7
def dense_yaw(route, speed, step_s=0.1):
    """Yaw rate of the playback [deg/s], sampled every step_s seconds, from the heading that is sailed."""
    tt = np.arange(0.0, route['s'][-1] / speed, step_s)
    psi = A.route_eval(route, speed * tt)[1]
    return np.abs(np.degrees(np.diff(psi))) / step_s


def legacy_load_trace_shapes(ST, path):
    """The environments' trace loader as it was before step 4 of the plan (2026-09-24): one global-mean
    projection and box-5 smoothing.  The live loader (scenario_targets.load_trace_shapes) no longer smooths
    and projects each trace on its own tangent plane; the box filter is kept there for this comparison."""
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[None]
    lat0, lon0 = float(np.nanmean(arr[..., 1])), float(np.nanmean(arr[..., 0]))
    shapes = []
    for tr in arr:
        x = np.radians(tr[:, 0] - lon0) * ST.EARTH_R * math.cos(math.radians(lat0))
        y = np.radians(tr[:, 1] - lat0) * ST.EARTH_R
        xy = np.stack([x, y], axis=1)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) > 3:
            shapes.append(ST.smooth_path(xy))
    return shapes


def t7(fleet, prep):
    say('## T7 Scenario routes: the environment code before step 4 (until 2026-09-24) against the mended route, '
        'on correctly timed windows')
    say('')
    import scenario_targets as ST
    rep, internals = prep['raw']['off']
    folder = os.path.join(OUT, 'prep_raw_off')
    shapes = legacy_load_trace_shapes(ST, os.path.join(folder, 'windows_lonlat.npy'))   # old chain: projection + box-5
    wins = np.load(os.path.join(folder, 'windows_xy.npy'))
    cur, new = dict(yaw=[], jumps=[], perp=[], c0=[]), dict(yaw=[], yaw1=[], jumps=[], perp=[], c0=[], fit=[], knot=[])
    for m, sh, w in zip(internals['meta'], shapes, wins):
        sg = internals['segments'][m['segment']]
        v = internals['voyages'][sg['voyage']]
        tr = fleet[v['source'][:-4]]
        true_course = math.radians(90.0 - S.truth_at(tr, [sg['tg'][m['grid_index']] - S.EPOCH0])[2][0])
        truth_xy = S.truth_at(tr, np.arange(sg['tg'][m['grid_index']], sg['tg'][m['grid_index'] + len(w) - 1], 2.0) - S.EPOCH0)[0]
        route = ST.rotate_translate(sh, (0.0, 0.0), 0.0)
        tgt = ST.TargetShip(route, 6.0, automation='fixed')
        psi = [tgt.psi]
        for _ in range(int(ST.route_length(route) / 6.0) - 2):
            tgt.advance(1.0)
            psi.append(tgt.psi)
        yaw = np.abs(A.wrap180(np.diff(psi)))
        cur['yaw'].append(yaw.max()); cur['jumps'].append(int(np.sum(yaw > 1.0)))
        cur['perp'].append(np.max(A.perpendicular_distance(sh - sh[0], w)))
        seg = sh[min(3, len(sh) - 1)] - sh[0]
        cur['c0'].append(abs(A.wrap180(math.degrees(math.atan2(seg[1], seg[0]) - true_course))))
        r = A.build_route(w)
        yw = dense_yaw(r, 6.0)
        new['yaw'].append(yw.max()); new['jumps'].append(int(np.sum(yw > 1.0)))
        new['yaw1'].append(math.degrees(6.0 * np.abs(r['kappa']).max()))
        new['perp'].append(np.max(A.perpendicular_distance(r['xy'][::4], w)))
        new['c0'].append(abs(A.wrap180(math.degrees(r['psi'][0] - true_course))))
        new['fit'].append(abs(A.wrap180(math.degrees(r['course0'] - true_course))))
    table(['route chain', 'largest yaw rate of the playback at 6 m/s [deg/s] P50 / P95 / max', 'samples above 1 deg/s (all routes)',
           'largest distance between route and window polyline [m]', 'start heading against the true course [deg]'],
          [['before step 4: load_trace_shapes (box 5) + rotate_translate (points 0 to 3) + TargetShip.advance (segment heading), 1 s steps',
            q3(cur['yaw'], '%.2f'), int(np.sum(cur['jumps'])), q3(cur['perp']), q3(cur['c0'], '%.2f')],
           ['mended: build_route (C2 spline, 5 m table) + route_eval, sampled every 0.1 s',
            q3(new['yaw'], '%.2f'), int(np.sum(new['jumps'])), q3(new['perp']), q3(new['c0'], '%.2f')]])
    say('Check that the curvature which is tested is the curvature which is sailed: speed x largest table curvature = %s deg/s, '
        'largest yaw rate sampled from the playback every 0.1 s = %s deg/s (P50 / P95 / max over routes).' % (
            q3(new['yaw1'], '%.3f'), q3(new['yaw'], '%.3f')))
    say('')
    RES['T7'] = dict(current_yaw=q3(cur['yaw'], '%.2f'), mended_yaw=q3(new['yaw'], '%.2f'), table_yaw=q3(new['yaw1'], '%.3f'),
                     current_c0=q3(cur['c0'], '%.2f'), mended_c0=q3(new['c0'], '%.2f'))

    starts = []
    for sg in internals['segments']:
        if 'grid_pos' not in sg or len(sg['tg']) < 45:
            continue
        v = internals['voyages'][sg['voyage']]
        tr = fleet[v['source'][:-4]]
        pg = to_truth(sg['grid_pos'], v, tr)
        for i0 in range(0, len(sg['tg']) - 40, 3):
            _, _, co, ra = S.truth_at(tr, [sg['tg'][i0] - S.EPOCH0])
            starts.append((pg[i0:i0 + 40] - pg[i0], math.radians(90.0 - co[0]), abs(ra[0]) > 0.05))
    say('Start course on %d window starts (%d of them in a turn)' % (len(starts), sum(1 for x in starts if x[2])))
    say('')
    rules = [('points 0 to 3 of the window (current rule)', lambda w: math.atan2(*(w[3] - w[0])[::-1])),
             ('tangent of the spline at its first point (used to place the route)', lambda w: A.build_route(w)['psi'][0]),
             ('least-squares fit over the first 300 m (diagnostic)', lambda w: A.build_route(w)['course0'])]
    rows = []
    for name, fn in rules:
        es, et = [], []
        for w, tc, turning in starts:
            (et if turning else es).append(abs(A.wrap180(math.degrees(fn(w) - tc))))
        rows.append([name, q3(es, '%.2f'), q3(et, '%.2f')])
        RES['T7_start_' + name[:10]] = dict(straight=q3(es, '%.2f'), turning=q3(et, '%.2f'))
    table(['start-course rule', 'error when the window starts on a straight [deg] P50 / P95 / max', 'error when it starts in a turn [deg]'], rows)

    say('Scale: a real-size route inside the arena standard (scale 35 / 244.74 = 0.143, yaw limit 1.964 deg/s, 6 m/s), yaw rate '
        'from the playback of the scaled route sampled every 0.1 s')
    say('')
    sc = 35.0 / 244.74
    un, scd, fitted, ok = [], [], [], []
    for w in wins:
        r = A.build_route(w)
        un.append(dense_yaw(r, 6.0).max())
        scd.append(dense_yaw(A.place_route(r, (0.0, 0.0), 0.0, scale=sc), 6.0).max())
        rf, info = A.fit_route_to_limit(w, 6.0, 1.964, scale=sc)
        ok.append(info['ok'])
        fitted.append(dense_yaw(A.place_route(rf, (0.0, 0.0), 0.0, scale=sc), 6.0).max())
    fitted, ok = np.asarray(fitted), np.asarray(ok)
    table(['route geometry', 'largest yaw rate at 6 m/s [deg/s] P50 / P95 / max', 'share above the arena limit'],
          [['unscaled, as the current code places it', q3(un, '%.2f'), '%.2f' % np.mean(np.asarray(un) > 1.964)],
           ['scaled by 0.143 like every other length of the standard', q3(scd, '%.2f'), '%.2f' % np.mean(np.asarray(scd) > 1.964)],
           ['scaled, after fit_route_to_limit: routes it accepts (%.0f %% of all)' % (100 * ok.mean()),
            q3(fitted[ok], '%.2f'), '%.2f' % np.mean(fitted[ok] > 1.964 * 1.001)]])
    RES['T7_scale'] = dict(unscaled=q3(un, '%.2f'), scaled=q3(scd, '%.2f'), accepted=float(ok.mean()),
                           accepted_over_limit=float(np.mean(fitted[ok] > 1.964 * 1.001)))


# ----------------------------------------------------------------------- T8
def t8(fleet):
    say('## T8 How the private-data checks read: the same data smoothed too much, in range, and too little (positions only)')
    say('')
    rows = []
    for label, q in (('too much smoothing, q = 1e-6', 1e-6), ('q = 1e-4', 1e-4), ('q = 3e-3', 3e-3), ('too little smoothing, q = 10', 10.0)):
        rep, internals = run(os.path.join(OUT, 'csv', 'raw'), 't8', q=q)
        E = grid_errors(internals, fleet)
        r, h = rep['residuals_m'], rep['smoothing']['table'][0]
        st = rep['smoothing']['choice'].get('course_vs_cog_by_speed', {})
        turn = [v['p95_abs'] for k, v in st.items() if k.endswith('turning')]
        strt = [v['p95_abs'] for k, v in st.items() if k.endswith('straight')]
        rows.append([label, '%.1f / %.1f' % (np.percentile(E['pos_straight'], 95), np.percentile(E['pos_turn'], 95)),
                     '%.3f' % np.percentile(E['rate_straight'], 95),
                     '%.1f / %.1f' % (r['along_interior']['rms'], r['cross_interior']['rms']),
                     '%.2f / %.2f' % (r['lag1_autocorr_along'], r['lag1_autocorr_cross']),
                     '%.1f / %.1f' % (h['single_rms'], h['block_rms']), '%.2f' % h.get('course_vs_cog_rms_deg', float('nan')),
                     '%.1f / %.1f' % (np.mean(strt) if strt else float('nan'), np.mean(turn) if turn else float('nan'))])
    table(['setting', 'TRUE position error P95 straight / turning [m]', 'TRUE turn-rate error P95 on straights [deg/s]',
           'residual RMS along / cross [m]', 'residual lag-1 autocorrelation along / cross', 'holdout RMS single / block [m]',
           'course against COG, RMS [deg]', 'course against COG P95, straight / turning (mean over speed bands) [deg]'], rows)


# ----------------------------------------------------------------------- T9
def t9(fleet, legacy, prep):
    say('## T9 Compatibility with the existing consumers')
    say('')
    sys.path.insert(0, V2)
    import target_ship as TS
    folder = os.path.join(OUT, 'prep_raw_off')
    ll = np.load(os.path.join(folder, 'windows_lonlat.npy'))
    sv = np.load(os.path.join(folder, 'windows_speed_course.npy'))
    proj = lambda a: np.stack([np.radians(a[:, 0] - a[0, 0]) * 6371000.0 * math.cos(math.radians(a[0, 1])),
                               np.radians(a[:, 1] - a[0, 1]) * 6371000.0], axis=1)          # the v2 projection
    ratio_new = [TS.route_from_trace(proj(a), 20.0, 2.0)[1] / np.mean(s[:, 0]) for a, s in zip(ll, sv)]
    ratio_old = []
    for name, tr in fleet.items():
        t = tr['reports']['raw']['t_true']
        ratio_old.append(TS.route_from_trace(proj(legacy['raw/' + name]), 20.0, 2.0)[1]
                         / np.mean(S.truth_at(tr, np.linspace(t[0], t[-1], 200))[1]))
    table(['array', 'v2 route_from_trace speed (trace_dt = 20 s) over the true mean speed, P50 / P95 / max'],
          [['legacy (row-index resampling)', q3(ratio_old, '%.2f')], ['mended windows', q3(ratio_new, '%.3f')]])
    import env_moving_obj as E
    from attack_scenarios import EncounterAttackEnv, hold_action
    env = EncounterAttackEnv(E.ownship(0, 0, 0, 0, 0, 0), scenario='mix', scale='maritime',
                             trace=os.path.join(folder, 'windows_lonlat.npy'), seed=3, save_dir=os.path.join(OUT, 'env_tmp'))
    env.reset()
    done, n = False, 0
    while not done and n < 400:
        _, _, done, _ = env.step(hold_action(env))
        n += 1
    ev = env.evaluation()
    say('EncounterAttackEnv(trace = mended windows, scale = maritime) runs: %d decisions, outcome %s, data_shaped = %s.' % (
        n, ev.get('outcome'), ev.get('data_shaped')))
    say('')
    RES['T9'] = dict(old=q3(ratio_old, '%.2f'), new=q3(ratio_new, '%.3f'))


# ---------------------------------------------------------------------- T10
def t10(prep):
    say('## T10 Augmentation values: reviewed plan against the first draft')
    say('')
    rep, internals = prep['raw']['off']
    cfg, dt = internals['cfg'], internals['cfg']['grid_dt']
    wins = np.load(os.path.join(OUT, 'prep_raw_off', 'windows_xy.npy'))
    real = [A.window_kinematics(w, dt) for w in wins]
    rows = [['real windows', '-', q3(np.concatenate([np.abs(r[1]) for r in real]), '%.4f'),
             q3(np.concatenate([np.abs(r[2]) for r in real]), '%.3f'), '-']]
    for label, kw in (('reviewed plan: offset SD 10-20 m, correlation 120-300 s, speed x0.95-1.05', dict(amp=(10.0, 20.0), corr_s=(120.0, 300.0), speed=(0.95, 1.05))),
                      ('first draft: offset SD 20-50 m, correlation 300-600 s, speed x0.9-1.1', dict(amp=(20.0, 50.0), corr_s=(300.0, 600.0), speed=(0.9, 1.1))),
                      ('for contrast: offset SD 20 m with a 30 s correlation', dict(amp=(20.0, 20.0), corr_s=(30.0, 30.0), speed=(1.0, 1.0)))):
        rng = np.random.default_rng(1)
        kin, dev = [], []
        for m, w in zip(internals['meta'], wins):
            sg = internals['segments'][m['segment']]
            a = A.augment_window(sg, m['grid_index'], cfg, rng, **kw)
            kin.append(A.window_kinematics(a, dt))
            dev.append(np.max(A.perpendicular_distance(a, sg['grid_pos'])))
        rows.append([label, q3(dev), q3(np.concatenate([np.abs(k[1]) for k in kin]), '%.4f'),
                     q3(np.concatenate([np.abs(k[2]) for k in kin]), '%.3f'), '%.2f' % np.mean([np.abs(k[2]).max() for k in kin])])
    table(['set', 'largest lateral shift inside a window [m] P50 / P95 / max', 'abs acceleration [m/s^2] P50 / P95 / max',
           'abs turn rate [deg/s] P50 / P95 / max', 'mean of the per-window largest turn rate'], rows)


# ---------------------------------------------------------------------- T11
def t11():
    say('## T11 Unit checks of the building blocks on exact geometry')
    say('')
    rows = []
    rng = np.random.default_rng(0)
    xy = rng.uniform(-40000, 40000, (2000, 2))
    back = A.ll_to_xy(*A.xy_to_ll(xy, 11.9, 57.6).T, 11.9, 57.6)
    rows.append(['projection round trip within 40 km of the origin', 'largest error %.2e m' % np.abs(back - xy).max()])
    la0, dl = 57.6, 0.1
    mid = math.radians(la0 + dl / 2)
    m_mid = A.A_WGS * (1 - A.E2_WGS) / (1 - A.E2_WGS * math.sin(mid) ** 2) ** 1.5
    north = A.ll_to_xy(11.9, la0 + dl, 11.9, la0)
    n0 = A.A_WGS / math.sqrt(1 - A.E2_WGS * math.sin(math.radians(la0)) ** 2)
    east = A.ll_to_xy(11.9 + dl, la0, 11.9, la0)
    rows.append(['tangent plane against closed forms, 0.1 deg away (about 11 km north, 6 km east)',
                 'north: %.3f m against meridian arc %.3f m (chord effect %.1e); east: %.4f m against N cos(lat) sin(dlon) = %.4f m' % (
                     north[1], m_mid * math.radians(dl), abs(north[1] - m_mid * math.radians(dl)) / north[1],
                     east[0], n0 * math.cos(math.radians(la0)) * math.sin(math.radians(dl)))])
    th = np.arange(0, 1.2, 0.1)
    arc = 1200.0 * np.stack([np.sin(th), 1 - np.cos(th)], axis=1)
    r = A.build_route(arc)
    k = r['kappa'] * 1200.0
    rows.append(['route through an exact arc of radius 1200 m sampled every 120 m',
                 'curvature x radius between %.4f and %.4f over the whole route; start heading error %.4f deg; largest distance '
                 'from the circle %.3f m' % (k.min(), k.max(), abs(math.degrees(r['psi'][0])),
                                             np.abs(np.hypot(r['xy'][:, 0], r['xy'][:, 1] - 1200.0) - 1200.0).max())])
    yw = dense_yaw(r, 6.0)
    rows.append(['playback of that arc at 6 m/s, every 0.1 s (exact yaw rate %.4f deg/s)' % math.degrees(6.0 / 1200.0),
                 'yaw rate between %.4f and %.4f deg/s' % (yw.min(), yw.max())])
    line = np.stack([np.linspace(0, 12000, 101), np.linspace(0, 3000, 101)], axis=1)
    r = A.build_route(line)
    rows.append(['route through a straight line', 'largest curvature %.2e 1/m, length %.3f m (exact %.3f)' % (
        np.abs(r['kappa']).max(), r['s'][-1], math.hypot(12000, 3000))])
    r1 = A.place_route(A.build_route(arc), (100.0, -50.0), 0.7, scale=0.143)
    r2 = A.extend_route(r1, 200.0)
    j = len(r1['s']) - 1
    rows.append(['place_route (scale 0.143, course 0.7 rad) and extend_route by 200 m',
                 'start (%.1f, %.1f), first heading %.4f rad, curvature x 1200 x 0.143 = %.4f, heading step at the joint %.1e rad, '
                 'length %.1f -> %.1f m' % (r2['xy'][0, 0], r2['xy'][0, 1], r2['psi'][0], np.median(r1['kappa']) * 1200 * 0.143,
                                            abs(r2['psi'][j + 1] - r2['psi'][j]), r1['s'][-1], r2['s'][-1])])
    t = np.cumsum(rng.uniform(2, 10, 300))
    z = np.stack([3.0 + 6.0 * t, -2.0 * t], axis=1)
    te = np.arange(math.ceil(t[0] / 20) * 20, t[-1], 20.0)
    sm = A.rts_smooth(t, z, te, 1e-3, 10.0)
    rows.append(['smoother on a noise-free constant-velocity track, irregular times',
                 'largest position error %.2e m, velocity error %.2e m/s' % (
                     np.abs(sm['pos'] - np.stack([3.0 + 6.0 * te, -2.0 * te], axis=1)).max(), np.abs(sm['vel'] - np.array([6.0, -2.0])).max())])
    # independent check of the scalar recursion: the same problem with dense 4 x 4 matrices
    tt = np.cumsum(rng.uniform(3, 9, 60))
    zz = np.stack([4.0 * tt, 2.0 * tt], axis=1) + rng.normal(0, 8.0, (60, 2))
    vv = np.tile([4.0, 2.0], (60, 1)) + rng.normal(0, 0.2, (60, 2))
    fast = A.rts_smooth(tt, zz, tt, 5e-3, 8.0, vv, 0.2)
    ref = dense_rts(tt, zz, vv, 5e-3, 8.0, 0.2)
    rows.append(['scalar covariance recursion against a dense-matrix Kalman / RTS smoother (positions and velocities measured)',
                 'largest difference %.2e m in position, %.2e m/s in velocity' % (
                     np.abs(fast['pos'] - ref[:, [0, 2]]).max(), np.abs(fast['vel'] - ref[:, [1, 3]]).max())])
    bad, cut = A.distance_outliers(np.arange(0, 400, 10.0), np.stack([6.0 * np.arange(0, 400, 10.0), np.zeros(40)], 1)
                                   + np.where((np.arange(40)[:, None] >= 15) & (np.arange(40)[:, None] < 22), [0.0, 500.0], 0.0),
                                   np.full(40, 6.0))
    rows.append(['distance rule on a run of 7 fixes offset by 500 m', 'removed fixes %s, cuts %d' % (np.flatnonzero(bad).tolist(), cut.sum())])
    bad, cut = A.distance_outliers(np.arange(0, 400, 10.0), np.stack([6.0 * np.arange(0, 400, 10.0), np.where(np.arange(40) >= 20, 900.0, 0.0)], 1),
                                   np.full(40, 6.0))
    rows.append(['distance rule on a lasting jump of 900 m', 'removed fixes %s, cut after fix %s' % (np.flatnonzero(bad).tolist(), np.flatnonzero(cut).tolist())])
    table(['check', 'result'], rows)


def dense_rts(t, z, vel, q, sp, sv):
    """Plain textbook Kalman filter and RTS smoother with 4 x 4 matrices, state (x, vx, y, vy)."""
    n = len(t)
    H, R = np.eye(4), np.diag([sp ** 2, sv ** 2, sp ** 2, sv ** 2])
    x = np.array([z[0, 0], vel[0, 0], z[0, 1], vel[0, 1]])
    P = np.diag([1e6, 25.0, 1e6, 25.0])
    xf, Pf, xp, Pp, Fs = [], [], [], [], []
    for k in range(n):
        if k:
            dt = t[k] - t[k - 1]
            f = np.array([[1, dt], [0, 1]])
            F = np.block([[f, np.zeros((2, 2))], [np.zeros((2, 2)), f]])
            qb = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
            Q = np.block([[qb, np.zeros((2, 2))], [np.zeros((2, 2)), qb]])
            x, P = F @ x, F @ P @ F.T + Q
            Fs.append(F)
        xp.append(x.copy()); Pp.append(P.copy())
        y = np.array([z[k, 0], vel[k, 0], z[k, 1], vel[k, 1]]) - H @ x
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x, P = x + K @ y, (np.eye(4) - K @ H) @ P
        xf.append(x.copy()); Pf.append(P.copy())
    xs = [None] * n
    xs[-1] = xf[-1]
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fs[k].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1])
    return np.array(xs)


# ---------------------------------------------------------------------- T12
def t12():
    say('## T12 The finished chain on a second fleet it was not developed on')
    say('')
    say('Other seeds, 20 voyages per regime, no setting changed. Regime A: as the development fleet. Regime B: position error twice '
        'as large and slower (Gauss-Markov 6 m / 300 s plus white 4 m), timestamps late by up to 5 s, 35 % message loss. Regime C: '
        'B with one report per 30 s (Class B like). Regime D: turns of 0.4 to 1.0 deg/s with the development noise.')
    say('')
    rows = []
    specs = [('A', dict(), (0.15, 0.5), 'raw'), ('B', dict(sigma_c=6.0, tau_c=300.0, sigma_w=4.0, latency_max=5.0, loss=0.35), (0.15, 0.5), 'raw'),
             ('C', dict(sigma_c=6.0, tau_c=300.0, sigma_w=4.0, latency_max=5.0, loss=0.35), (0.15, 0.5), 'ds30'),
             ('D', dict(), (0.4, 1.0), 'raw')]
    for name, kw, rates, reg in specs:
        folder = os.path.join(OUT, 'csv_val_' + name)
        fl = {tr['name']: tr for tr in S.make_fleet(folder, n_voyages=20, seed=1234 + ord(name), regimes=(reg,), rate_range=rates, **kw)}
        for label, extra in (('positions only (default)', {}), ('receiver COG declared', dict(cog_source='receiver'))):
            rep, internals = run(os.path.join(folder, reg), 'val_' + name, **extra)
            E = grid_errors(internals, fl)
            rows.append([name, label, '%g' % rep['smoothing']['q_used'], 'on' if rep['sog_cog']['velocity_aiding'] else 'off',
                         rep['counts']['windows'], q3(E['pos']), q3(E['pos_turn']), '%.2f to %.2f' % (np.percentile(E['ratio'], 5), np.percentile(E['ratio'], 95)),
                         q3(E['course'], '%.2f'), '%.3f / %.3f' % (np.percentile(E['rate_straight'], 95), np.percentile(E['rate_turn'], 95))])
            RES['T12_%s_%s' % (name, label[:9])] = dict(pos=q3(E['pos']), course=q3(E['course'], '%.2f'))
    table(['regime', 'mode', 'q used', 'aiding', 'windows', 'position error [m] P50 / P95 / max', 'same, while turning',
           'implied speed / true speed (P5 to P95)', 'course error [deg] P50 / P95 / max', 'turn-rate error P95 straight / turning [deg/s]'], rows)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'tests'
    os.makedirs(OUT, exist_ok=True)
    if cmd == 'fleet':
        S.make_fleet(os.path.join(OUT, 'csv'))
        print('synthetic CSV files written to', os.path.join(OUT, 'csv'))
        return
    fleet = load_fleet()
    legacy = np.load(os.path.join(OUT, 'legacy.npz'))
    say('# Check of the mending plan for part 1 (sampling and smoothing) on synthetic voyages with a known truth')
    say('')
    stamp = lambda f: datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(HERE, f))).strftime('%Y-%m-%d %H:%M:%S')
    say('Run %s. Code as of: ais_prep.py %s, synth_ais.py %s, check_mending_plan.py %s. Every table below was produced by this run.' % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), stamp('ais_prep.py'), stamp('synth_ais.py'), stamp('check_mending_plan.py')))
    say('')
    say('Development fleet: %d voyages of 50-80 min, speeds 2-11 m/s, turns of 0.15-0.5 deg/s, Class A report schedule with 20 %% message loss, '
        'reception gaps of 1-10 min, position error = Gauss-Markov 3 m (100 s) + white 2 m + timestamps rounded to the second. '
        'Regimes: raw, ds30 and ds60 (a provider that keeps one report per 30 s / 60 s). The rules of the chain were worked out on this fleet; '
        'T12 runs the finished chain on a second one.' % len(fleet))
    say('')
    prep = {reg: {'off': run(os.path.join(OUT, 'csv', reg), reg + '_off'),
                  'on': run(os.path.join(OUT, 'csv', reg), reg + '_on', cog_source='receiver')} for reg in REGIMES}
    t0_t1(fleet, legacy, prep)
    t2(fleet, prep)
    t3(fleet, prep)
    t3b(fleet)
    t3c(fleet)
    t4(prep)
    t5(fleet)
    t6()
    t7(fleet, prep)
    t8(fleet)
    t9(fleet, legacy, prep)
    t10(prep)
    t11()
    t12()
    with open(os.path.join(OUT, 'results.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LINES) + '\n')
    with open(os.path.join(OUT, 'results.json'), 'w') as fh:
        json.dump(RES, fh, indent=1, default=str)


if __name__ == '__main__':
    main()
