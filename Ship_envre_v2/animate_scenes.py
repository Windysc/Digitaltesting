"""
Replay ShipAttackEnv episodes as chart-style animations (GIF, no ffmpeg
needed) and storyboards (PNG), for a trained checkpoint or the scripted
baseline of the task.

The map is drawn like a navigation display: sea background, graticule in
nautical miles, scale bar and north arrow, a camera that follows the two
ships (with an overview inset), true-scale hull silhouettes with point
markers, velocity vectors (6 min of run), fading trails with 1-minute dots,
the target's COLREG sectors (head-on, starboard / port crossing, overtaking),
its ship-domain ellipse and d_safe ring, the predicted positions of both
ships at the CPA with the DCPA segment, and a data box with range, bearing,
DCPA, TCPA, CRI, severity, encounter type and the target's automation state.
Side panels: range and DCPA, TCPA, severity and CRI, rudder and throttle.
A second, plain FULL-SCOPE map (fixed extent = the whole map, white
background, routes, trails and the two ships only, no data overlays) sits
next to the detailed display so the routes and the relative position of the
ships are always visible without the camera moving.  Storyboards and grid
GIFs use the plain full-scope map.

Examples
--------
  python animate_scenes.py --ckpt checkpoints/ppo_attack_best.pth --episodes 3 --grid 6 --out animations
  python animate_scenes.py --task navigate --ckpt checkpoints/ppo_navigate_best.pth --episodes 2
  python animate_scenes.py --episodes 2                       # scripted attack baseline
  python animate_scenes.py --task navigate --episodes 2       # scripted navigation baseline
  python animate_scenes.py --scenario head_on --automation autonomous --episodes 1

Extra arguments go to main_attack_ppo_ship.gen_args (scenario, automation,
danger criteria, data files, ...).
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Circle, Ellipse, Wedge, Rectangle
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main_attack_ppo_ship as M
from encounter import SEVERITY_NAMES

NM = 1852.0
HULL = np.array([[0.5, 0.0], [0.3, 0.5], [-0.45, 0.5], [-0.5, 0.3], [-0.5, -0.3], [-0.45, -0.5], [0.3, -0.5]])
SEA = '#dbe9f4'
GRID = '#b9cfe0'
OWN_C, TGT_C = '#1f4fbf', '#c62828'
SEV_C = {0: '#c62828', 1: '#f39c12', 2: '#e53935', 3: '#111111'}


# ------------------------------------------------------------------ record
def record_episode(env, agent):
    """Run one episode and return a dict of per-step arrays plus static data."""
    obs = env.reset()
    keys = ['state', 'target', 'target_psi', 'target_speed', 'own_speed', 'd', 'dcpa', 'tcpa', 'severity',
            'cri', 'evading', 'bearing', 'obs']
    rec = {k: [] for k in keys}
    rec['action'], rec['reward'], rec['encounter'] = [], [], []

    def _push(e, o):
        rec['state'].append(env.dyn.state.copy())
        rec['target'].append(env.target.pos.copy())
        rec['target_psi'].append(env.target.psi)
        rec['target_speed'].append(env.target.speed)
        rec['own_speed'].append(e['own_speed'])
        rec['d'].append(e['dist'])
        rec['dcpa'].append(e['dcpa'] if np.isfinite(e['tcpa']) and e['tcpa'] >= 0 else e['dist'])
        rec['tcpa'].append(e['tcpa'] if np.isfinite(e['tcpa']) else -1.0)
        rec['severity'].append(e['severity']); rec['cri'].append(e['cri'])
        rec['evading'].append(float(env.target.evading)); rec['bearing'].append(e['bearing'])
        rec['encounter'].append(e['encounter']); rec['obs'].append(o.copy())
    _push(env.enc_hist[-1], obs)
    done = False
    while not done:
        a = M.baseline_policy(env, obs) if agent is None else agent.select_action(obs, deterministic=True)
        obs, r, done, flag = env.step(a)
        rec['action'].append(env.ownship_action[-1]); rec['reward'].append(r)
        _push(env.enc_hist[-1], obs)
    enc_types = rec.pop('encounter')
    for k in list(rec):
        rec[k] = np.asarray(rec[k], dtype=float)
    rec['encounter'] = enc_types
    rec['task'] = env.task
    rec['scenario'] = env.scenario
    rec['automation'] = getattr(env.target, 'automation', 'custom')
    rec['termination'] = env.termination
    rec['evaluation'] = env.evaluation()
    rec['guideline'] = env.guide_route.copy()
    rec['target_route'] = env.target.route.copy()
    rec['borders'] = env.borders.copy()
    rec['dest'] = env.dest.copy()
    rec['dest_radius'] = env.dest_radius
    rec['dt'] = env.dt
    rec['d_safe'], rec['t_safe'] = env.enc_params.d_safe, env.enc_params.t_safe
    rec['domain_a'], rec['domain_b'] = env.enc_params.domain_a, env.enc_params.domain_b
    rec['L'], rec['B'] = env.dyn.L, env.dyn.B
    return rec


# -------------------------------------------------------------------- draw
def hull_patch(x, y, psi, L, B, scale, color, alpha=0.95, lw=0.6):
    pts = HULL * np.array([L, B]) * scale
    c, s = np.cos(psi), np.sin(psi)
    rot = pts @ np.array([[c, s], [-s, c]])
    return Polygon(rot + np.array([x, y]), closed=True, fc=color, ec='k', lw=lw, alpha=alpha, zorder=6)


def fading_trail(ax, pts, color, n=40, lw=1.8):
    if len(pts) < 2:
        return
    pts = pts[-n:]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    alphas = np.linspace(0.08, 0.9, len(segs))
    rgb = matplotlib.colors.to_rgb(color)
    lc = LineCollection(segs, colors=[(rgb[0], rgb[1], rgb[2], a) for a in alphas], linewidths=lw, zorder=4)
    ax.add_collection(lc)


def graticule(ax, x0, x1, y0, y1, step=NM):
    xs = np.arange(np.floor(x0 / step) * step, x1 + step, step)
    ys = np.arange(np.floor(y0 / step) * step, y1 + step, step)
    for x in xs:
        ax.plot([x, x], [y0, y1], color=GRID, lw=0.6, zorder=1)
    for y in ys:
        ax.plot([x0, x1], [y, y], color=GRID, lw=0.6, zorder=1)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels(['%.0f' % (x / NM) for x in xs], fontsize=7)
    ax.set_yticklabels(['%.0f' % (y / NM) for y in ys], fontsize=7)


def colreg_sectors(ax, c, psi, radius):
    deg = np.degrees(psi)
    ax.add_patch(Wedge(c, radius, deg - 6, deg + 6, fc='#e53935', ec='none', alpha=0.10, zorder=2))          # head-on
    ax.add_patch(Wedge(c, radius, deg - 112.5, deg - 6, fc='#fb8c00', ec='none', alpha=0.10, zorder=2))      # starboard
    ax.add_patch(Wedge(c, radius, deg + 6, deg + 112.5, fc='#43a047', ec='none', alpha=0.08, zorder=2))      # port
    ax.add_patch(Wedge(c, radius, deg + 112.5, deg + 247.5, fc='#607d8b', ec='none', alpha=0.07, zorder=2))  # astern


def camera(rec, k, min_half=1.5 * NM):
    o, t = rec['state'][k, :2], rec['target'][k]
    c = 0.5 * (o + t)
    half = max(0.62 * np.linalg.norm(o - t), min_half)
    return c, half


def draw_map(ax, rec, k, small=False, title=None, hull_scale=None):
    ax.clear()
    ax.set_facecolor(SEA)
    st, tg = rec['state'], rec['target']
    c, half = camera(rec, k)
    x0, x1, y0, y1 = c[0] - half, c[0] + half, c[1] - half, c[1] + half
    if hull_scale is None:
        hull_scale = max(1.0, half / (2 * NM))            # true scale when zoomed in, larger when zoomed out
    graticule(ax, x0, x1, y0, y1, step=NM if half < 4 * NM else 2 * NM)

    ax.plot(rec['guideline'][:, 0], rec['guideline'][:, 1], color='0.45', ls='--', lw=0.9, zorder=2)
    ax.plot(rec['target_route'][:, 0], rec['target_route'][:, 1], color='#ef9a9a', ls='--', lw=0.9, zorder=2)
    if rec['task'] == 'navigate':
        ax.add_patch(Circle(rec['dest'], rec['dest_radius'], fc='#a5d6a7', ec='#2e7d32', lw=1.2, alpha=0.6, zorder=3))
    bx = rec['borders']
    ax.add_patch(Rectangle(bx[0], bx[3, 0] - bx[0, 0], bx[3, 1] - bx[0, 1], fc='none', ec='k', ls=':', lw=0.8, zorder=3))

    # trails with 1-minute dots
    fading_trail(ax, st[:k + 1, :2], OWN_C)
    fading_trail(ax, tg[:k + 1], TGT_C)
    dots = np.arange(max(0, k - 40), k + 1, 6)
    ax.plot(st[dots, 0], st[dots, 1], '.', color=OWN_C, ms=3, zorder=4)
    ax.plot(tg[dots, 0], tg[dots, 1], '.', color=TGT_C, ms=3, zorder=4)

    ox, oy, opsi = st[k, 0], st[k, 1], st[k, 2]
    tx, ty, tpsi = tg[k, 0], tg[k, 1], rec['target_psi'][k]
    sev = int(rec['severity'][k])
    tcol = SEV_C[sev]

    # target: COLREG sectors, domain, d_safe
    colreg_sectors(ax, (tx, ty), tpsi, rec['d_safe'])
    ax.add_patch(Ellipse((tx, ty), 2 * rec['domain_a'], 2 * rec['domain_b'], angle=np.degrees(tpsi),
                         fc='#e53935' if sev >= 2 else 'none', alpha=0.18 if sev >= 2 else 1.0,
                         ec='#e53935', ls=':', lw=1.0, zorder=3))
    ax.add_patch(Circle((tx, ty), rec['d_safe'], fc='none', ec='#fb8c00', ls=':', lw=0.8, zorder=3))

    # velocity vectors (6 minutes of run)
    ov = rec['own_speed'][k] * 360.0 * np.array([np.cos(opsi), np.sin(opsi)])
    tv = rec['target_speed'][k] * 360.0 * np.array([np.cos(tpsi), np.sin(tpsi)])
    for (px, py), v, col in [((ox, oy), ov, OWN_C), ((tx, ty), tv, tcol)]:
        ax.annotate('', xy=(px + v[0], py + v[1]), xytext=(px, py),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2, shrinkA=0, shrinkB=0), zorder=7)

    # CPA prediction
    tcpa = rec['tcpa'][k]
    if 0 <= tcpa <= 3600:
        oc = np.array([ox, oy]) + ov / 360.0 * tcpa
        tc = np.array([tx, ty]) + tv / 360.0 * tcpa
        ax.plot([ox, oc[0]], [oy, oc[1]], color=OWN_C, lw=0.7, ls=':', zorder=5)
        ax.plot([tx, tc[0]], [ty, tc[1]], color=TGT_C, lw=0.7, ls=':', zorder=5)
        ax.plot(oc[0], oc[1], marker='s', mfc='none', mec=OWN_C, ms=5, zorder=6)
        ax.plot(tc[0], tc[1], marker='s', mfc='none', mec=TGT_C, ms=5, zorder=6)
        ax.plot([oc[0], tc[0]], [oc[1], tc[1]], color='#6a1b9a', lw=1.2, zorder=6)
        if not small:
            ax.text(0.5 * (oc[0] + tc[0]), 0.5 * (oc[1] + tc[1]), ' DCPA %.2f nm' % (rec['dcpa'][k] / NM),
                    fontsize=7, color='#6a1b9a', zorder=8)
    ax.plot([ox, tx], [oy, ty], color='0.35', lw=0.5, alpha=0.6, zorder=5)

    # ships: silhouettes + point markers
    ax.add_patch(hull_patch(ox, oy, opsi, rec['L'], rec['B'], hull_scale, OWN_C))
    ax.add_patch(hull_patch(tx, ty, tpsi, rec['L'], rec['B'], hull_scale, tcol))
    ax.plot(ox, oy, 'o', color=OWN_C, mec='w', ms=6, zorder=8)
    ax.plot(tx, ty, 'o', color=tcol, mec='w', ms=6, zorder=8)
    if rec['evading'][k] > 0:
        ax.plot(tx, ty, marker='*', color='#ffb300', mec='k', mew=0.4, ms=12, zorder=9)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    if small:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel('east [nm]', fontsize=8); ax.set_ylabel('north [nm]', fontsize=8)
        w = 2 * half
        ax.plot([x0 + 0.06 * w, x0 + 0.06 * w + NM], [y0 + 0.05 * w] * 2, 'k-', lw=2, zorder=9)
        ax.text(x0 + 0.06 * w, y0 + 0.07 * w, '1 nm', fontsize=7, zorder=9)
        ax.annotate('N', xy=(x1 - 0.06 * w, y1 - 0.06 * w), xytext=(x1 - 0.06 * w, y1 - 0.16 * w),
                    ha='center', fontsize=8, arrowprops=dict(arrowstyle='-|>', color='k'), zorder=9)
        enc = rec['encounter'][k]
        ka = min(k, len(rec['action']) - 1) if len(rec['action']) else 0
        rd = rec['action'][ka, 0] if len(rec['action']) else 0.0
        th = rec['action'][ka, 1] if len(rec['action']) else 0.0
        txt = ('%s  |  target: %s%s\n'
               'range %.2f nm   bearing %+.0f deg   DCPA %.2f nm   TCPA %s\n'
               'CRI %.2f   severity %d %s   encounter %s\n'
               'own %.1f m/s  rudder %+.1f  thr %.2f   |   target %.1f m/s') % (
            rec['scenario'], rec['automation'], '  EVADING' if rec['evading'][k] > 0 else '',
            rec['d'][k] / NM, np.degrees(rec['bearing'][k]), rec['dcpa'][k] / NM,
            '%.1f min' % (tcpa / 60) if tcpa >= 0 else 'passed',
            rec['cri'][k], sev, SEVERITY_NAMES[sev], enc, rec['own_speed'][k], rd, th, rec['target_speed'][k])
        ax.text(0.01, 0.99, txt, transform=ax.transAxes, va='top', ha='left', fontsize=7, family='monospace',
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.9), zorder=10)
    if title is None:
        if small:
            title = ('t=%.0f s  range %.2f nm  DCPA %.2f nm' + chr(10) + '%s%s%s') % (
                k * rec['dt'], rec['d'][k] / NM, rec['dcpa'][k] / NM, SEVERITY_NAMES[sev],
                ' (evading)' if rec['evading'][k] > 0 else '',
                '  -> ' + str(rec['termination']) if k == len(st) - 1 else '')
        else:
            title = 't = %4.0f s   step %3d' % (k * rec['dt'], k)
            if k == len(st) - 1:
                title += '   -> ' + str(rec['termination'])
    ax.set_title(title, fontsize=8 if small else 10)


def full_extent(rec, margin=0.12):
    """Fixed square extent covering both ships' whole-episode tracks and the
    own route (not the long scenario route), so the view never moves."""
    pts = np.concatenate([rec['state'][:, :2], rec['target'], rec['guideline']])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    half = 0.5 * max(hi - lo) * (1 + 2 * margin)
    half = max(half, 2 * NM)
    return np.array([[c[0] - half, c[1] - half], [c[0] + half, c[1] - half],
                     [c[0] - half, c[1] + half], [c[0] + half, c[1] + half]])


def draw_full(ax, rec, k, small=False, title=None):
    """Fixed-extent, plain map: routes, trails, current positions, domain outline."""
    ax.clear()
    ax.set_facecolor('white')
    st, tg = rec['state'], rec['target']
    bx = full_extent(rec)
    ax.plot(rec['guideline'][:, 0], rec['guideline'][:, 1], color='0.55', ls='--', lw=0.9, label='own route')
    ax.plot(rec['target_route'][:, 0], rec['target_route'][:, 1], color='#ef9a9a', ls='--', lw=0.9, label='target route')
    if rec['task'] == 'navigate':
        ax.add_patch(Circle(rec['dest'], rec['dest_radius'], fc='none', ec='#2e7d32', lw=1.0))
    ax.plot(st[:k + 1, 0], st[:k + 1, 1], color=OWN_C, lw=1.4, label='own ship')
    ax.plot(tg[:k + 1, 0], tg[:k + 1, 1], color=TGT_C, lw=1.4, label='target ship')
    ticks = np.arange(0, k + 1, 30)                                  # 5-minute marks
    ax.plot(st[ticks, 0], st[ticks, 1], '.', color=OWN_C, ms=3)
    ax.plot(tg[ticks, 0], tg[ticks, 1], '.', color=TGT_C, ms=3)
    ax.plot(st[0, 0], st[0, 1], 'o', mfc='none', mec=OWN_C, ms=5)
    ax.plot(tg[0, 0], tg[0, 1], 'o', mfc='none', mec=TGT_C, ms=5)
    span = bx[3, 0] - bx[0, 0]
    for (x, y, psi), col in [((st[k, 0], st[k, 1], st[k, 2]), OWN_C),
                             ((tg[k, 0], tg[k, 1], rec['target_psi'][k]), TGT_C)]:
        ax.plot(x, y, 'o', color=col, ms=6, mec='w', zorder=6)
        ax.plot([x, x + 0.04 * span * np.cos(psi)], [y, y + 0.04 * span * np.sin(psi)], color=col, lw=1.2, zorder=6)
    ax.plot([st[k, 0], tg[k, 0]], [st[k, 1], tg[k, 1]], color='0.5', lw=0.6, ls=':')
    ax.add_patch(Ellipse(tg[k], 2 * rec['domain_a'], 2 * rec['domain_b'], angle=np.degrees(rec['target_psi'][k]),
                         fc='none', ec=TGT_C, ls=':', lw=0.8))
    ax.set_xlim(bx[0, 0], bx[3, 0]); ax.set_ylim(bx[0, 1], bx[3, 1])
    ax.set_aspect('equal')
    step = 2 * NM if span > 8 * NM else NM
    xs = np.arange(np.ceil(bx[0, 0] / step) * step, bx[3, 0], step)
    ys = np.arange(np.ceil(bx[0, 1] / step) * step, bx[3, 1], step)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.grid(True, color='0.9', lw=0.5)
    if small:
        ax.set_xticklabels([]); ax.set_yticklabels([])
    else:
        ax.set_xticklabels(['%.0f' % (x / NM) for x in xs], fontsize=7)
        ax.set_yticklabels(['%.0f' % (y / NM) for y in ys], fontsize=7)
        ax.set_xlabel('east [nm]', fontsize=8); ax.set_ylabel('north [nm]', fontsize=8)
        ax.legend(loc='best', fontsize=7)
    if title is None:
        title = 'full scope   t = %.0f s   range %.2f nm' % (k * rec['dt'], rec['d'][k] / NM)
        if small:
            title += ('' if k < len(st) - 1 else chr(10) + '-> ' + str(rec['termination']))
    ax.set_title(title, fontsize=8 if small else 10)


def draw_panels(axes, rec, k):
    ax_d, ax_t, ax_s, ax_a = axes
    t = np.arange(len(rec['d'])) * rec['dt']
    for ax in axes:
        ax.clear(); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
    ax_d.plot(t, rec['d'] / NM, color='0.75', lw=1)
    ax_d.plot(t, rec['dcpa'] / NM, color='#ef9a9a', lw=1, ls='--')
    ax_d.plot(t[:k + 1], rec['d'][:k + 1] / NM, 'k-', lw=1.5, label='range')
    ax_d.plot(t[:k + 1], rec['dcpa'][:k + 1] / NM, color=TGT_C, lw=1.2, ls='--', label='DCPA')
    ax_d.axhline(rec['d_safe'] / NM, color='#fb8c00', ls=':', lw=1)
    ax_d.legend(loc='upper right', fontsize=6); ax_d.set_ylabel('nm', fontsize=7)
    ax_d.set_ylim(0, max(rec['d']) / NM * 1.05)
    tc = np.clip(rec['tcpa'] / 60.0, -1, 30)
    ax_t.plot(t, tc, color='0.75', lw=1)
    ax_t.plot(t[:k + 1], tc[:k + 1], color='#6a1b9a', lw=1.3)
    ax_t.axhline(rec['t_safe'] / 60.0, color='#fb8c00', ls=':', lw=1)
    ax_t.set_ylabel('TCPA [min]', fontsize=7); ax_t.set_ylim(-1.5, 31)
    ax_s.step(t, rec['severity'], where='post', color='0.75', lw=1)
    ax_s.step(t[:k + 1], rec['severity'][:k + 1], where='post', color=TGT_C, lw=1.5, label='severity')
    ax_s.plot(t[:k + 1], rec['cri'][:k + 1] * 3, color='#6a1b9a', lw=1.1, label='CRI x3')
    ax_s.set_yticks([0, 1, 2, 3]); ax_s.set_ylim(-0.1, 3.2); ax_s.legend(loc='upper left', fontsize=6)
    ax_s.set_ylabel('danger', fontsize=7)
    if len(rec['action']):
        ta = t[1:len(rec['action']) + 1]
        kk = min(k, len(ta))
        ax_a.step(ta, rec['action'][:, 0], where='post', color='0.8', lw=1)
        ax_a.step(ta, rec['action'][:, 1], where='post', color='0.85', lw=1)
        ax_a.step(ta[:kk], rec['action'][:kk, 0], where='post', color=OWN_C, lw=1.4, label='rudder')
        ax_a.step(ta[:kk], rec['action'][:kk, 1], where='post', color='#2e7d32', lw=1.4, label='throttle')
        ax_a.legend(loc='upper right', fontsize=6); ax_a.set_ylim(-1.1, 1.1)
    ax_a.set_ylabel('action', fontsize=7); ax_a.set_xlabel('time [s]', fontsize=7)
    for ax in axes:
        ax.axvline(k * rec['dt'], color='k', lw=0.7, alpha=0.5)


def animate_episode(rec, path, fps=8, max_frames=240):
    n = len(rec['state'])
    frames = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
    fig = plt.figure(figsize=(18, 7.2))
    gs = fig.add_gridspec(4, 3, width_ratios=[1.1, 1.35, 1])
    ax_full = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[:, 1])
    axes = [fig.add_subplot(gs[i, 2]) for i in range(4)]

    def update(k):
        draw_full(ax_full, rec, k)
        draw_map(ax_map, rec, k)
        draw_panels(axes, rec, k)
        fig.suptitle('%s  |  %s task, %s scenario, other ship: %s' % (
            os.path.basename(path), rec['task'], rec['scenario'], rec['automation']), fontsize=9)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


def storyboard(rec, path, n_panels=4):
    n = len(rec['state'])
    ks = np.unique(np.linspace(0, n - 1, n_panels).astype(int))
    fig, axs = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4.6))
    for ax, k in zip(np.atleast_1d(axs), ks):
        draw_full(ax, rec, k, small=True)
    ev = rec['evaluation']
    fig.suptitle(('%s task, %s scenario, other ship %s -> %s in %d steps' + chr(10)
                  + 'event grade %d (%s), score %.0f, min DCPA %.2f nm') % (
        rec['task'], rec['scenario'], rec['automation'], rec['termination'], len(rec['action']),
        ev['event_grade'], ev['event_grade_name'], ev['event_score'], rec['dcpa'].min() / NM), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def animate_grid(recs, path, fps=8, max_frames=240):
    n = max(len(r['state']) for r in recs)
    frames = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
    cols = min(3, len(recs)); rows = int(np.ceil(len(recs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.6 * rows), squeeze=False)

    def update(k):
        for i, rec in enumerate(recs):
            ax = axs[i // cols][i % cols]
            kk = min(k, len(rec['state']) - 1)
            draw_full(ax, rec, kk, small=True,
                      title=('%s / %s' + chr(10) + 't=%.0f s  range %.2f nm  sev %d%s') % (
                         rec['scenario'], rec['automation'], kk * rec['dt'], rec['d'][kk] / NM,
                         int(rec['severity'][kk]),
                         '  -> ' + rec['termination'] if kk == len(rec['state']) - 1 else ''))
        for j in range(len(recs), rows * cols):
            axs[j // cols][j % cols].axis('off')
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path


# -------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=str, default='', help='PPO checkpoint; omit for the scripted baseline of the task')
    p.add_argument('--episodes', type=int, default=3, help='single-episode animations to produce')
    p.add_argument('--grid', type=int, default=0, help='if > 0, also animate this many scenarios in one grid GIF')
    p.add_argument('--out', type=str, default=os.path.join(HERE, 'animations'))
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--anim_seed', type=int, default=123)
    a, rest = p.parse_known_args()
    args = M.gen_args(rest)
    args.device = 'cpu'
    M.set_device(args)
    os.makedirs(a.out, exist_ok=True)

    env = M.build_env(args, seed=a.anim_seed, evaluation=True)
    agent = None
    if a.ckpt:
        sd = env.observation_space.shape[0]
        ad = env.action_space.shape[0] if args.continuous else env.action_space.n
        agent = M.PPO(sd, ad, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip,
                      bool(args.continuous), 0.6)
        agent.load(a.ckpt)
    tag = a.tag or ('%s_%s' % (args.task, 'agent' if agent is not None else 'baseline'))

    summary = []
    for i in range(a.episodes):
        rec = record_episode(env, agent)
        base = os.path.join(a.out, '%s_%s_episode%d' % (tag, rec['scenario'], i))
        animate_episode(rec, base + '.gif', fps=a.fps)
        storyboard(rec, base + '_storyboard.png')
        ev = rec['evaluation']; ev['files'] = [base + '.gif', base + '_storyboard.png']
        summary.append(ev)
        print('episode %d [%s / %s]: %s in %d steps, min DCPA %.0f m, grade %d (%s) score %.0f -> %s.gif' % (
            i, rec['scenario'], rec['automation'], rec['termination'], len(rec['action']), rec['dcpa'].min(),
            ev['event_grade'], ev['event_grade_name'], ev['event_score'], base))
    if a.grid > 0:
        recs = [record_episode(env, agent) for _ in range(a.grid)]
        gp = os.path.join(a.out, '%s_grid%d.gif' % (tag, a.grid))
        animate_grid(recs, gp, fps=a.fps)
        summary.append(dict(grid=[r['evaluation'] for r in recs], files=[gp]))
        print('grid: %s  (%s)' % (gp, ', '.join('%s:%s' % (r['scenario'][:8], r['termination']) for r in recs)))
    with open(os.path.join(a.out, '%s_summary.json' % tag), 'w') as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == '__main__':
    main()
