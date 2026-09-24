"""
check_long_block_holdout.py -- would long hidden blocks with guard bands choose the smoothing
strength better than the short blocks the chain uses?  (Suggested in the external review.)

Blocks of 120 to 300 s are hidden together with 60 s guard bands on both sides, only the fixes
inside the block are scored.  The choice is compared with the q that the truth prefers.

    python check_long_block_holdout.py      (after: python check_mending_plan.py fleet)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ais_prep as A            # noqa: E402
import synth_ais as S           # noqa: E402
from check_mending_plan import OUT, fd_kin, load_fleet, to_truth     # noqa: E402

QS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)


def long_block_errors(sg, q, cfg, rng, block=(120.0, 300.0), guard=60.0, share=0.25):
    t, use = sg['t'], sg['use']
    u = np.flatnonzero(use)
    if len(u) < 30 or t[u[-1]] - t[u[0]] < 3 * (block[1] + 2 * guard):
        return np.zeros(0)
    hidden, score = np.zeros(len(t), bool), np.zeros(len(t), bool)
    span0, span1 = t[u[0]] + guard + 60.0, t[u[-1]] - guard - 60.0
    tries = 0
    while score.sum() < share * len(u) and tries < 50:
        tries += 1
        L = rng.uniform(*block)
        t0 = rng.uniform(span0, max(span0 + 1.0, span1 - L))
        if hidden[(t >= t0 - guard) & (t <= t0 + L + guard)].any():
            continue
        hidden |= use & (t >= t0 - guard) & (t <= t0 + L + guard)
        score |= use & (t >= t0) & (t <= t0 + L)
    m = use & ~hidden
    if not score.any() or m.sum() < 8:
        return np.zeros(0)
    idx = np.flatnonzero(score)
    return np.linalg.norm(A.rts_smooth(t, sg['xy'], t[idx], q, cfg['sigma_p'], None, cfg['sigma_v'], m)['pos'] - sg['xy'][idx], axis=1)


def truth_scores(internals, fleet, q):
    pos, crs, rate = [], [], []
    for sg in internals['segments']:
        if 'grid_pos' not in sg or len(sg['tg']) < 8:
            continue
        v = internals['voyages'][sg['voyage']]
        tr = fleet[v['source'][:-4]]
        sm = A.rts_smooth(sg['t'], sg['xy'], sg['tg'], q, internals['cfg']['sigma_p'], None, 0.2, sg['use'])
        pt = S.truth_at(tr, sg['tg'] - S.EPOCH0)[0]
        pg = to_truth(sm['pos'], v, tr)
        pos.append(np.linalg.norm(pg - pt, axis=1))
        s1, c1, r1 = fd_kin(pg, 20.0); s0, c0, r0 = fd_kin(pt, 20.0)
        crs.append(A.wrap180(c1 - c0)); rate.append(r1 - r0)
    f = lambda a: float(np.sqrt(np.mean(np.concatenate(a) ** 2)))
    return f(pos), f(crs), f(rate)


def main():
    fleet = load_fleet()
    lines = ['# Long hidden blocks (120-300 s, 60 s guard bands) as a referee for q', '',
             '| regime | q by long-block RMS (ties to the smaller q) | its plain minimum | q by long-block median | '
             'truth-preferred q (composite of position, course, turn rate) | composite cost of the long-block RMS choice | '
             'composite cost of the short-block choice inside the bounds (the chain) |', '|---|---|---|---|---|---|---|']
    for name in ('raw', 'ds30', 'ds60'):
        rep, internals = A.prepare([os.path.join(OUT, 'csv', name)], os.path.join(OUT, 'prep_longblock'), keep=True)
        cfg = internals['cfg']
        rms, med, tru = [], [], []
        for q in QS:
            e = np.concatenate([long_block_errors(sg, q, cfg, np.random.default_rng(100 + i))
                                for i, sg in enumerate(internals['segments'])] or [np.zeros(0)])
            rms.append(np.sqrt(np.mean(e ** 2))); med.append(np.median(e)); tru.append(truth_scores(internals, fleet, q))
        tru = np.array(tru)
        comp = (tru / tru.min(axis=0)).mean(axis=1)
        pick = lambda s: QS[int(np.flatnonzero(np.asarray(s) <= np.min(s) * 1.02)[0])]
        lines.append('| %s | %g | %g | %g | %g | %.3f | %.3f |' % (
            name, pick(rms), QS[int(np.argmin(rms))], pick(med), QS[int(np.argmin(comp))], comp[QS.index(pick(rms))],
            comp[QS.index(rep['smoothing']['q_used'])]))
    lines += ['', 'The long-block score is set by the few blocks that hide a turn, where a large q lets the end velocities follow the '
              'most recent motion. It rewards extrapolation across a hole of four to seven minutes, which is a different task from '
              'smoothing between reports that are seconds apart.']
    print('\n'.join(lines))
    with open(os.path.join(OUT, 'results_long_block_holdout.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
