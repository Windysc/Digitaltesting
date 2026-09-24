"""
chart_viz.py -- chart-style episode renderer for the scenario environments of
this folder (layout of the animate_scenes.py of the earlier Ship_envre_v2 build,
metres; renewed 2026-09-15 for the collision standard and the encounter lifecycle).

Single-episode GIF:
  header     scenario, rule and meaning; initial range / DCPA / TCPA; the
             thresholds of the standard in force
  left       FULL-SCOPE map, fixed extent: target's fixed track, both tracks
             (the attacker's coloured by severity), time ticks, start labels,
             closest-approach segment, the target's domain and d_safe ring
  centre     DETAIL navigation display: sea, km graticule, scale bar, north
             arrow, smoothed camera following both ships, hull silhouettes
             (magnified when they would be invisible, factor printed), velocity
             vectors, fading trails, COLREG sectors, the target's domain
             ellipse (filled when violated), d_safe ring and collision circle,
             the attacker's domain, predicted CPA positions with the DCPA
             segment, a data box (phase, range, bearing, encounter type, DCPA,
             TCPA, CRI, severity, hold count, domain margin) and an outcome
             banner on the final frames
  right      panels over time with the lifecycle phase shaded: range + DCPA,
             TCPA with the warning window, domain margin + CRI, severity,
             turn command
Storyboards pick the event frames (start, first warning, domain entered,
closest approach, end); grid GIFs animate several episodes; catalogue()
draws the initial geometry of every attack scenario next to the standard.

record_episode() works for every MassTestingEnv variant.  With an
EncounterAttackEnv (attack_scenarios.py) the severity is the v2 standard
(0 clear, 1 CPA warning, 2 domain violation, 3 collision); for the older
scenes it is 0 clear, 1 inside the risk ring, 2 inside the contact zone,
3 hull contact.
"""
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, Wedge

import env_moving_obj as E
from scenario_targets import cpa as cpa_simple

HULL = np.array([[0.5, 0.0], [0.3, 0.5], [-0.45, 0.5], [-0.5, 0.3], [-0.5, -0.3], [-0.45, -0.5], [0.3, -0.5]])
SEA, GRID = '#dbe9f4', '#bdd3e6'
OWN_C, TGT_C, CPA_C = '#1f4fbf', '#c62828', '#6a1b9a'
SEV_PAL = {0: '#78909c', 1: '#f9a825', 2: '#e53935', 3: '#111111'}
PHASE_C = {0: '#ffffff', 1: '#fff1dc', 2: '#e1efff', 3: '#e3f4e4'}
PHASE_NAMES = {0: 'approach', 1: 'action', 2: 'passing', 3: 'clear'}
STD_SEV_NAMES = {0: 'clear', 1: 'cpa_warning', 2: 'domain_violation', 3: 'collision'}
OLD_SEV_NAMES = {0: 'clear', 1: 'risk_zone', 2: 'contact', 3: 'collision'}
SEVERITY_NAMES = STD_SEV_NAMES
OUTCOME_C = {'success': '#2e7d32', 'clear': '#c62828', 'off_map': '#ef6c00', 'unresolved': '#546e7a',
             'collision': '#c62828', 'timeout': '#546e7a', 'running': '#546e7a'}
OUTCOME_TEXT = {'success': 'SUCCESS', 'clear': 'FAILED: encounter cleared', 'off_map': 'FAILED: left the map',
                'unresolved': 'UNRESOLVED: safety cap', 'collision': 'FAILED: collision', 'timeout': 'TIMEOUT'}
KM = 1000.0
TRAIL_STEPS = 40


# ------------------------------------------------------------------ record
def record_episode(env, agent=None, deterministic=False, policy=None):
    """Run one episode with a PPO agent or a scripted policy(env, obs); per-decision arrays + static scene."""
    obs = env.reset()
    std_env = hasattr(env, 'std') and hasattr(env, 'enc')
    scen = bool(getattr(env, 'per_episode_targets', False))
    tgt = env.targets[0] if env.targets else (env.objects[0] if env.objects else None)
    if std_env:
        contact = env.std.domain_b
    else:
        contact = env._attack_range(tgt) if (env.task == 'attack' and tgt is not None) else 50.0
    keys = ['own', 'tgt', 'others', 'd', 'dh', 'dcpa', 'tcpa', 'bearing', 'sev', 'evading', 'phase', 'hold', 'cri',
            'margin']
    R = {k: [] for k in keys}
    R['encounter'], R['action'], R['reward'] = [], [], []

    def push():
        own = env.live_ownship
        R['own'].append([own.x, own.y, math.radians(own.cog), own.sp])
        others = []
        for o in env.objects:
            ox, oy = o.position(env.t)
            others.append([ox, oy, math.radians(o.Direction)])
        R['others'].append(others if others else [[0.0, 0.0, 0.0]])
        if tgt is None:
            R['tgt'].append([0.0] * 6)
            for k in ('d', 'dh', 'dcpa', 'tcpa', 'bearing', 'sev', 'evading', 'phase', 'hold', 'cri', 'margin'):
                R[k].append(0.0)
            R['encounter'].append('')
            return
        tx, ty = tgt.position(env.t)
        tvx, tvy = tgt.velocity
        R['tgt'].append([tx, ty, math.radians(tgt.Direction), math.hypot(tvx, tvy), tvx, tvy])
        d = math.hypot(tx - own.x, ty - own.y)
        dh = tgt.hull_distance(own.x, own.y, env.t)
        if std_env:
            e = env.enc
            dcpa, tcpa, sev = e['dcpa'], e['tcpa'], e['severity']
            phase, hold, cri, margin, enc = e['phase'], e['hold'], e['cri'], e['margin'], e['encounter']
            brg = math.degrees(e['bearing'])
        else:
            dcpa, tcpa = cpa_simple((own.x, own.y), own.velocity, (tx, ty), (tvx, tvy))
            brg = E.wrap_deg(math.degrees(math.atan2(ty - own.y, tx - own.x)) - own.cog)
            sev = 3 if dh <= E.OWN_RADIUS else 2 if dh <= contact else 1 if dh <= tgt.risk_range else 0
            phase, hold, cri, margin, enc = 0, 0, 0.0, dh / max(contact, 1.0), ''
        for k, v in (('d', d), ('dh', dh), ('dcpa', dcpa), ('tcpa', tcpa), ('bearing', brg), ('sev', sev),
                     ('evading', float(getattr(tgt, 'evading', False))), ('phase', phase), ('hold', hold),
                     ('cri', cri), ('margin', margin)):
            R[k].append(v)
        R['encounter'].append(enc)

    push()
    done = False
    while not done:
        a = policy(env, obs) if policy is not None else agent.select_action(obs, deterministic=deterministic)
        obs, r, done, flag = env.step(a)
        R['action'].append(int(a)); R['reward'].append(float(r))
        push()
    if agent is not None:
        agent.buffer.clear()

    rec = {k: np.asarray(v, dtype=float) for k, v in R.items() if k not in ('action', 'encounter')}
    rec['action'] = np.asarray(R['action'], dtype=int)
    rec['encounter'] = R['encounter']
    rec['task'] = env.task
    rec['outcome'] = env.outcome
    rec['termination'] = getattr(env, 'termination', env.outcome)
    rec['eval'] = env.evaluation()
    rec['steps'] = env.n_steps
    rec['dt'] = env.dt_decision
    rec['n_sub'] = env.n_sub
    rec['path'] = np.asarray(env.live_ownship.path)
    rec['map'] = (env.X_LEN, env.Y_LEN)
    rec['map_box'] = tuple(env.map_box) if std_env else (0.0, env.X_LEN, -env.Y_LEN, env.Y_LEN)
    rec['action_table'] = list(env.action_table)
    rec['action_names'] = list(env.action_names)
    rec['contact'] = contact
    rec['standard'] = env.std.as_dict() if std_env else None
    rec['meta'] = dict(env.current) if scen else {}
    rec['scenario'] = env.current.get('scenario', '') if scen else ('obst_%s' % tgt.id if tgt is not None else env.task)
    rec['automation'] = getattr(env, 'automation', 'none') if scen else 'none'
    rec['has_target'] = tgt is not None
    if tgt is not None:
        rec['L'], rec['W'], rec['risk_range'] = tgt.L, tgt.W, tgt.risk_range
        rec['target_track'] = env.target_track() if scen else rec['tgt'][:, :2]
        if scen:
            rec['target_route'] = np.asarray(env.target.route)
        elif tgt.is_moving:
            rec['target_route'] = np.stack([tgt.position(0.0), tgt.position(env.t + 60.0)])
        else:
            rec['target_route'] = None
        rec['target_moving'] = bool(tgt.is_moving)
    else:
        rec['L'], rec['W'], rec['risk_range'] = 0.0, 0.0, 0.0
        rec['target_track'], rec['target_route'], rec['target_moving'] = None, None, False
    rec['own_L'], rec['own_B'] = ((rec['L'], rec['W']) if scen else (40.0, 10.0))
    rec['hazards'] = [np.asarray(o.corners(0.0)) for o in env.hazards if not o.is_moving]
    rec['obj_LW'] = [(o.L, o.W) for o in env.objects]
    rec['dest'] = ((env.nt.x, env.nt.y, env.nt.target_deviation_distance, env.nt.direction)
                   if env.task == 'navigate' and env.nt is not None else None)
    st = rec['standard']
    rec['t_show'] = 1.5 * st['t_safe'] if st else 900.0
    rec['vec_s'] = 0.25 * st['t_safe'] if st else 60.0
    rec['limits'] = dict(env.limits) if (std_env and hasattr(env, 'limits')) else None
    return rec


# ------------------------------------------------------------------ helpers
def sev_names(rec):
    return STD_SEV_NAMES if rec.get('standard') else OLD_SEV_NAMES


def zones(rec):
    st = rec.get('standard')
    if st:
        return dict(domain=(st['domain_a'], st['domain_b']), ring=st['d_safe'], collision=st['collision_dist'],
                    ring_label='d_safe', sector=1.2 * st['d_safe'])
    c = rec['contact']
    return dict(domain=(rec['L'] / 2 + c, rec['W'] / 2 + c), ring=max(rec['risk_range'], 1.0), collision=None,
                ring_label='risk range', sector=max(rec['risk_range'], 1.0))


def nice_step(span, n=5):
    raw = span / max(n, 1)
    for s in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000):
        if s >= raw:
            return float(s)
    return float(10 ** math.ceil(math.log10(raw)))


def tick_every(rec):
    total = (len(rec['own']) - 1) * rec['dt']
    for s in (6, 12, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if total / s <= 10:
            return s
    return 3600


def _fmt_km(v, step):
    return ('%.2f' if step < 100 else '%.1f' if step < 1000 else '%.0f') % (v / KM)


def hull_patch(x, y, psi, L, B, scale, color, alpha=0.95, lw=0.6, z=6):
    pts = HULL * np.array([L, B]) * scale
    c, s = np.cos(psi), np.sin(psi)
    rot = pts @ np.array([[c, s], [-s, c]])
    return Polygon(rot + np.array([x, y]), closed=True, fc=color, ec='k', lw=lw, alpha=alpha, zorder=z)


def fading_trail(ax, pts, color, n=TRAIL_STEPS, lw=1.8):
    if len(pts) < 2:
        return
    pts = pts[-n:]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    rgb = matplotlib.colors.to_rgb(color)
    alphas = np.linspace(0.08, 0.9, len(segs))
    ax.add_collection(LineCollection(segs, colors=[(*rgb, a) for a in alphas], linewidths=lw, zorder=4))


def sev_trail(ax, pts, sevs, base, lw=1.8, fade=False, zorder=4):
    """Track segments coloured by the severity at their end (base colour while clear)."""
    if len(pts) < 2:
        return
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    m = len(segs)
    alphas = np.linspace(0.15, 1.0, m) if fade else np.ones(m)
    cols = []
    for i in range(m):
        s = int(sevs[i + 1])
        cols.append((*matplotlib.colors.to_rgb(base if s == 0 else SEV_PAL[s]), alphas[i]))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=zorder, capstyle='round'))


def graticule(ax, x0, x1, y0, y1):
    step = nice_step(x1 - x0, 5)
    xs = np.arange(np.floor(x0 / step) * step, x1 + step, step)
    ys = np.arange(np.floor(y0 / step) * step, y1 + step, step)
    for x in xs:
        ax.plot([x, x], [y0, y1], color=GRID, lw=0.6, zorder=1)
    for y in ys:
        ax.plot([x0, x1], [y, y], color=GRID, lw=0.6, zorder=1)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.set_xticklabels([_fmt_km(x, step) for x in xs], fontsize=7)
    ax.set_yticklabels([_fmt_km(y, step) for y in ys], fontsize=7)
    return step


def colreg_sectors(ax, c, psi, radius):
    deg = np.degrees(psi)
    ax.add_patch(Wedge(c, radius, deg - 6, deg + 6, fc='#e53935', ec='none', alpha=0.10, zorder=2))          # head-on
    ax.add_patch(Wedge(c, radius, deg - 112.5, deg - 6, fc='#fb8c00', ec='none', alpha=0.09, zorder=2))      # starboard
    ax.add_patch(Wedge(c, radius, deg + 6, deg + 112.5, fc='#43a047', ec='none', alpha=0.07, zorder=2))      # port
    ax.add_patch(Wedge(c, radius, deg + 112.5, deg + 247.5, fc='#607d8b', ec='none', alpha=0.06, zorder=2))  # astern


def target_zones(ax, rec, tx, ty, tpsi, sev, lw=1.0):
    z = zones(rec)
    a, b = z['domain']
    ang = np.degrees(tpsi)
    if sev >= 2:
        ax.add_patch(Ellipse((tx, ty), 2 * a, 2 * b, angle=ang, fc='#e53935', ec='none', alpha=0.2, zorder=3))
    ax.add_patch(Ellipse((tx, ty), 2 * a, 2 * b, angle=ang, fc='none', ec='#e53935', ls='--', lw=lw, zorder=3))
    ax.add_patch(Circle((tx, ty), z['ring'], fc='none', ec='#fb8c00', ls=':', lw=lw, zorder=3))
    if z['collision']:
        ax.add_patch(Circle((tx, ty), z['collision'], fc='none', ec='k', lw=0.6 * lw, alpha=0.7, zorder=3))


def own_domain(ax, rec, ox, oy, opsi):
    if rec.get('standard'):
        a, b = zones(rec)['domain']
        ax.add_patch(Ellipse((ox, oy), 2 * a, 2 * b, angle=np.degrees(opsi), fc='none', ec=OWN_C, ls=':', lw=0.8,
                             alpha=0.8, zorder=3))


def camera_track(rec, window=6):
    """Per-frame (centre x, centre y, half width), smoothed so the detail view glides."""
    n = len(rec['own'])
    mh = max(1.5 * zones(rec)['ring'], 250.0)
    cx, cy, hs = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        lo = max(0, k - window)
        pts = rec['own'][lo:k + 1, :2]
        if rec['has_target']:
            pts = np.vstack([pts, rec['tgt'][lo:k + 1, :2]])
        pmin, pmax = pts.min(0), pts.max(0)
        c = 0.5 * (pmin + pmax)
        cx[k], cy[k] = c
        hs[k] = max(0.5 * float(np.max(pmax - pmin)) * 1.3 + 0.35 * mh, mh)
    for arr in (cx, cy, hs):
        for k in range(1, n):
            arr[k] = 0.6 * arr[k - 1] + 0.4 * arr[k]
    for k in range(n):
        need = [abs(rec['own'][k, 0] - cx[k]), abs(rec['own'][k, 1] - cy[k])]
        if rec['has_target']:
            need += [abs(rec['tgt'][k, 0] - cx[k]), abs(rec['tgt'][k, 1] - cy[k])]
        hs[k] = max(hs[k], 1.15 * max(need) + 0.2 * mh)
    return cx, cy, hs


def full_extent(rec, margin=0.08):
    pts = [rec['own'][:, :2]]
    if rec['has_target'] and rec['target_moving']:
        pts.append(rec['tgt'][:, :2])
    elif rec['has_target']:
        pts.append(rec['tgt'][:1, :2])
    if not rec.get('standard'):
        b = rec['map_box']
        pts.append(np.array([[b[0], b[2]], [b[1], b[3]]]))
    if rec['dest'] is not None:
        pts.append(np.array([rec['dest'][:2]]))
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = max(margin * float(np.max(hi - lo)), 0.8 * zones(rec)['ring'])
    c = 0.5 * (lo + hi)
    half = 0.5 * float(np.max(hi - lo)) + pad
    return c[0] - half, c[0] + half, c[1] - half, c[1] + half


def _map_rect(rec):
    b = rec['map_box']
    return Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc='none', ec='k', ls=':', lw=0.8, zorder=3)


def _draw_scene_static(ax, rec):
    for corners in rec['hazards']:
        ax.add_patch(Polygon(corners, closed=True, fc='#8d6e63', ec='#5d4037', alpha=0.6, zorder=3))
    if rec['dest'] is not None:
        dx, dy, dr, dd = rec['dest']
        ax.add_patch(Circle((dx, dy), dr, fc='#a5d6a7', ec='#2e7d32', lw=1.2, alpha=0.5, zorder=3))
        rad = math.radians(dd)
        ax.annotate('', xy=(dx + 150 * math.cos(rad), dy + 150 * math.sin(rad)), xytext=(dx, dy),
                    arrowprops=dict(arrowstyle='-|>', color='#2e7d32', lw=1.0), zorder=4)


def _tcpa_text(tcpa):
    if not np.isfinite(tcpa):
        return 'none'
    return '%.0f s' % tcpa if tcpa >= 0 else 'passed'


def outcome_banner(ax, rec, fontsize=11, x=0.02, y=0.03, ha='left'):
    oc = rec['outcome']
    txt = OUTCOME_TEXT.get(oc, str(oc).upper())
    if oc == 'success' and rec.get('termination') not in (None, 'success'):
        txt += ' (%s)' % str(rec['termination']).replace('_', ' ')
    ev = rec['eval']
    if 'grade_name' in ev:
        txt += '\ngrade %s %s, event score %.0f, min range %.0f m' % (
            ev['event_grade'], ev['grade_name'].replace('_', ' '), ev['event_score'], ev['min_distance'])
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va='bottom', fontsize=fontsize, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.45', fc=OUTCOME_C.get(oc, '#333333'), ec='none', alpha=0.92), zorder=12)


# ------------------------------------------------------------------ detail display
def draw_map(ax, rec, k, cam=None, small=False, title=None):
    ax.clear()
    ax.set_facecolor(SEA)
    n = len(rec['own'])
    if cam is None:
        cam = camera_track(rec)
    cx, cy, half = cam[0][k], cam[1][k], cam[2][k]
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    graticule(ax, x0, x1, y0, y1)
    names, z = sev_names(rec), zones(rec)
    if rec['target_route'] is not None:
        ax.plot(rec['target_route'][:, 0], rec['target_route'][:, 1], color='#ef9a9a', ls='--', lw=0.9, zorder=2)
    ax.add_patch(_map_rect(rec))
    _draw_scene_static(ax, rec)
    st, sev = rec['own'], rec['sev']
    lo = max(0, k - TRAIL_STEPS)
    sev_trail(ax, st[lo:k + 1, :2], sev[lo:k + 1], OWN_C, lw=2.2, fade=True)
    ox, oy, opsi, osp = st[k]
    vs = rec['vec_s']
    ov = osp * vs * np.array([np.cos(opsi), np.sin(opsi)])
    cur = int(sev[k])
    hs = max(1.0, 0.045 * 2 * half / max(rec['own_L'], 1.0))
    if rec['has_target']:
        tx, ty, tpsi, tsp, tvx, tvy = rec['tgt'][k]
        if rec['target_moving']:
            fading_trail(ax, rec['tgt'][lo:k + 1, :2], TGT_C)
        colreg_sectors(ax, (tx, ty), tpsi if rec['target_moving'] else opsi + np.pi, z['sector'])
        target_zones(ax, rec, tx, ty, tpsi, cur)
        own_domain(ax, rec, ox, oy, opsi)
        tv = np.array([tvx, tvy]) * vs
        for (px, py), v, col in (((ox, oy), ov, OWN_C), ((tx, ty), tv, TGT_C)):
            if np.hypot(*v) > 1.0:
                ax.annotate('', xy=(px + v[0], py + v[1]), xytext=(px, py),
                            arrowprops=dict(arrowstyle='-|>', color=col, lw=1.3, shrinkA=0, shrinkB=0), zorder=7)
        tcpa = rec['tcpa'][k]
        if np.isfinite(tcpa) and 0 <= tcpa <= rec['t_show']:
            oc = np.array([ox, oy]) + ov / vs * tcpa
            tc = np.array([tx, ty]) + tv / vs * tcpa
            ax.plot([ox, oc[0]], [oy, oc[1]], color=OWN_C, lw=0.7, ls=':', zorder=5)
            ax.plot([tx, tc[0]], [ty, tc[1]], color=TGT_C, lw=0.7, ls=':', zorder=5)
            ax.plot(oc[0], oc[1], marker='s', mfc='none', mec=OWN_C, ms=5, zorder=6)
            ax.plot(tc[0], tc[1], marker='s', mfc='none', mec=TGT_C, ms=5, zorder=6)
            ax.plot([oc[0], tc[0]], [oc[1], tc[1]], color=CPA_C, lw=1.3, zorder=6)
            if not small:
                ax.text(0.5 * (oc[0] + tc[0]), 0.5 * (oc[1] + tc[1]), ' DCPA %.0f m' % rec['dcpa'][k],
                        fontsize=7, color=CPA_C, zorder=8)
        ax.plot([ox, tx], [oy, ty], color='0.35', lw=0.5, alpha=0.6, zorder=5)
        tcol = TGT_C if cur == 0 else SEV_PAL[cur]
        ax.add_patch(hull_patch(tx, ty, tpsi, rec['L'], rec['W'], hs, tcol))
        ax.plot(tx, ty, 'o', color=tcol, mec='w', ms=5, zorder=8)
        if rec['evading'][k] > 0:
            ax.plot(tx, ty, marker='*', color='#ffb300', mec='k', mew=0.4, ms=12, zorder=9)
    else:
        ax.annotate('', xy=(ox + ov[0], oy + ov[1]), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle='-|>', color=OWN_C, lw=1.2, shrinkA=0, shrinkB=0), zorder=7)
    for j, (Lj, Wj) in enumerate(rec['obj_LW']):
        if rec['has_target'] and j == 0:
            continue
        xj, yj, pj = rec['others'][k, j]
        ax.add_patch(hull_patch(xj, yj, pj, Lj, Wj, 1.0, '#8d6e63', alpha=0.7))
    ax.add_patch(hull_patch(ox, oy, opsi, rec['own_L'], rec['own_B'], hs, OWN_C))
    ax.plot(ox, oy, 'o', color=OWN_C, mec='w', ms=5, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    if small:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel('east [km]', fontsize=8); ax.set_ylabel('north [km]', fontsize=8)
        w = 2 * half
        bar = nice_step(w, 5)
        bx = x1 - 0.06 * w - bar
        ax.plot([bx, bx + bar], [y0 + 0.04 * w] * 2, 'k-', lw=2.5, zorder=9)
        ax.text(bx + bar / 2, y0 + 0.055 * w, '%.0f m' % bar, fontsize=7, ha='center', zorder=9)
        ax.annotate('N', xy=(x1 - 0.05 * w, y1 - 0.05 * w), xytext=(x1 - 0.05 * w, y1 - 0.14 * w),
                    ha='center', fontsize=8, arrowprops=dict(arrowstyle='-|>', color='k'), zorder=9)
        if hs > 1.0:
            ax.text(0.99, 0.005, 'hulls drawn x%.0f' % hs, transform=ax.transAxes, fontsize=6, ha='right',
                    va='bottom', color='0.35', zorder=9)
        ka = min(k - 1, len(rec['action']) - 1) if k > 0 and len(rec['action']) else -1
        a_name = rec['action_names'][rec['action'][ka]] if ka >= 0 else '-'
        st_ = rec.get('standard')
        hold_txt = 'hold %d/%d' % (rec['hold'][k], st_['hold_steps']) if st_ else ''
        lines = ['t %5.0f s   decision %3d   phase %s' % (k * rec['dt'], k, PHASE_NAMES[int(rec['phase'][k])].upper()
                                                          if st_ else '-')]
        if rec['has_target']:
            lines += ['range %6.0f m   bearing %+5.0f deg   %s' % (rec['d'][k], rec['bearing'][k], rec['encounter'][k]),
                      'DCPA %6.0f m   TCPA %-7s  CRI %.2f' % (rec['dcpa'][k], _tcpa_text(rec['tcpa'][k]), rec['cri'][k]),
                      'severity %d %-17s %s' % (cur, names[cur], hold_txt),
                      'domain margin %.2f   closest so far %.0f m' % (rec['margin'][k], rec['d'][:k + 1].min())]
        lines += ['attacker %.1f m/s  cmd %-9s | target %s %.1f m/s' % (
            osp, a_name, rec['automation'], rec['tgt'][k, 3] if rec['has_target'] else 0.0)]
        ax.text(0.01, 0.99, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left', fontsize=7,
                family='monospace', zorder=10,
                bbox=dict(boxstyle='round', fc='white', ec=SEV_PAL[cur] if cur else '0.6', lw=1.6 if cur else 0.8,
                          alpha=0.93))
        if k == n - 1:
            outcome_banner(ax, rec)
    if title is None:
        title = 't = %4.0f s   decision %3d' % (k * rec['dt'], k)
    ax.set_title(title, fontsize=8 if small else 10)


# ------------------------------------------------------------------ full scope
def draw_full(ax, rec, k, small=False, title=None, extent=None):
    ax.clear()
    ax.set_facecolor('white')
    n = len(rec['own'])
    x0, x1, y0, y1 = extent or full_extent(rec)
    span = x1 - x0
    st, sev = rec['own'], rec['sev']
    if not (small and rec.get('standard')):
        ax.add_patch(_map_rect(rec))
    _draw_scene_static(ax, rec)
    if rec['target_route'] is not None:
        ax.plot(rec['target_route'][:, 0], rec['target_route'][:, 1], color='#ef9a9a', ls='--', lw=0.9, zorder=2)
    sev_trail(ax, st[:k + 1, :2], sev[:k + 1], OWN_C, lw=2.0 if not small else 1.6)
    step = max(1, int(round(tick_every(rec) / rec['dt'])))
    ticks = np.arange(0, k + 1, step)
    ax.plot(st[ticks, 0], st[ticks, 1], '.', color=OWN_C, ms=3.5, zorder=5)
    ax.plot(st[0, 0], st[0, 1], 'o', mfc='white', mec=OWN_C, ms=6, zorder=6)
    if not small:
        ax.text(st[0, 0], st[0, 1], '  A start', fontsize=7, color=OWN_C, va='top', zorder=6)
    poses = [((st[k, 0], st[k, 1], st[k, 2]), OWN_C)]
    if rec['has_target']:
        tg = rec['tgt']
        if rec['target_moving']:
            ax.plot(tg[:k + 1, 0], tg[:k + 1, 1], color=TGT_C, lw=2.0 if not small else 1.6, zorder=4)
            ax.plot(tg[ticks, 0], tg[ticks, 1], '.', color=TGT_C, ms=3.5, zorder=5)
            ax.plot(tg[0, 0], tg[0, 1], 'o', mfc='white', mec=TGT_C, ms=6, zorder=6)
            if not small:
                ax.text(tg[0, 0], tg[0, 1], '  T start', fontsize=7, color=TGT_C, va='top', zorder=6)
                for i in ticks[::2][1:]:
                    ax.text(tg[i, 0], tg[i, 1], ' %ds' % (i * rec['dt']), fontsize=5.5, color='#b71c1c', zorder=5)
                    ax.text(st[i, 0], st[i, 1], ' %ds' % (i * rec['dt']), fontsize=5.5, color='#0d47a1', zorder=5)
        poses.append(((tg[k, 0], tg[k, 1], tg[k, 2]), TGT_C))
        target_zones(ax, rec, tg[k, 0], tg[k, 1], tg[k, 2], int(sev[k]), lw=0.8)
        ax.plot([st[k, 0], tg[k, 0]], [st[k, 1], tg[k, 1]], color='0.5', lw=0.6, ls=':')
        kmin = int(np.argmin(rec['d']))
        if k >= kmin:
            ax.plot([st[kmin, 0], tg[kmin, 0]], [st[kmin, 1], tg[kmin, 1]], color=CPA_C, lw=1.6, zorder=7)
            ax.plot([st[kmin, 0], tg[kmin, 0]], [st[kmin, 1], tg[kmin, 1]], 'x', color=CPA_C, ms=5, zorder=7)
            if not small:
                ax.text(tg[kmin, 0], tg[kmin, 1], '  closest %.0f m @ %ds' % (rec['d'][kmin], kmin * rec['dt']),
                        fontsize=7, color=CPA_C, zorder=8)
    for (x, y, psi), col in poses:
        ax.plot(x, y, 'o', color=col, ms=6, mec='w', zorder=8)
        ax.plot([x, x + 0.045 * span * np.cos(psi)], [y, y + 0.045 * span * np.sin(psi)], color=col, lw=1.4, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    gstep = nice_step(span, 5)
    xs = np.arange(np.ceil(x0 / gstep) * gstep, x1, gstep)
    ys = np.arange(np.ceil(y0 / gstep) * gstep, y1, gstep)
    ax.set_xticks(xs); ax.set_yticks(ys)
    ax.grid(True, color='0.9', lw=0.5)
    if small:
        ax.set_xticklabels([]); ax.set_yticklabels([])
    else:
        ax.set_xticklabels([_fmt_km(x, gstep) for x in xs], fontsize=7)
        ax.set_yticklabels([_fmt_km(y, gstep) for y in ys], fontsize=7)
        ax.set_xlabel('east [km]', fontsize=8); ax.set_ylabel('north [km]', fontsize=8)
        names = sev_names(rec)
        handles = [Line2D([], [], color=OWN_C, lw=2, label='attacker (own ship)'),
                   Line2D([], [], color=TGT_C, lw=2, label='target'),
                   Line2D([], [], color='#ef9a9a', lw=1, ls='--', label='target fixed track' if rec.get('standard')
                          else 'target route')]
        handles += [Line2D([], [], color=SEV_PAL[s], lw=2, label='attacker at %s' % names[s]) for s in (1, 2, 3)]
        handles += [Line2D([], [], color='#e53935', lw=1, ls='--', label='target domain'),
                    Line2D([], [], color='#fb8c00', lw=1, ls=':', label=zones(rec)['ring_label']),
                    Line2D([], [], color=CPA_C, lw=1.6, label='closest approach')]
        ax.legend(handles=handles, loc='lower right', fontsize=6, framealpha=0.9)
    if title is None:
        title = 'full scope   t = %.0f s   range %.0f m' % (k * rec['dt'], rec['d'][k])
    ax.set_title(title, fontsize=8 if small else 10)


# ------------------------------------------------------------------ panels
def _phase_spans(axes, rec, k, t):
    ph = rec['phase'][:k + 1].astype(int)
    if not rec.get('standard') or len(ph) < 2:
        return
    i0 = 0
    for i in range(1, len(ph) + 1):
        if i == len(ph) or ph[i] != ph[i0]:
            if ph[i0] > 0:
                for ax in axes:
                    ax.axvspan(t[i0], t[min(i, len(t) - 1)], color=PHASE_C[ph[i0]], zorder=0, lw=0)
            i0 = i


def draw_panels(axes, rec, k):
    ax_d, ax_t, ax_m, ax_s, ax_a = axes[:5]
    ax_v = axes[5] if len(axes) > 5 else None
    n = len(rec['d'])
    t = np.arange(n) * rec['dt']
    for ax in axes:
        ax.clear(); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
        ax.set_xlim(0, max(t[-1], rec['dt']))
    _phase_spans(axes, rec, k, t)
    z, st = zones(rec), rec.get('standard')
    # range and DCPA
    ax_d.plot(t, rec['d'], color='0.8', lw=1)
    ax_d.plot(t[:k + 1], rec['d'][:k + 1], 'k-', lw=1.5, label='range')
    ax_d.plot(t[:k + 1], rec['dcpa'][:k + 1], color=CPA_C, lw=1.2, ls='--', label='DCPA')
    top = max(float(rec['d'].max()), z['ring'] * 1.3, 1.0) * 1.05
    ax_d.axhline(z['ring'], color='#fb8c00', ls=':', lw=1)
    ax_d.text(t[-1], z['ring'], '%s %.0f m ' % (z['ring_label'], z['ring']), fontsize=6, color='#e65100', ha='right',
              va='bottom')
    if z['collision']:
        ax_d.axhline(z['collision'], color='k', ls=':', lw=0.8)
        ax_d.text(t[-1], z['collision'], 'collision %.0f m ' % z['collision'], fontsize=6, ha='right', va='bottom')
    ax_d.set_ylim(0, top); ax_d.set_ylabel('m', fontsize=7); ax_d.legend(loc='upper left', fontsize=6)
    # TCPA
    tmax = rec['t_show']
    tc = np.where(np.isfinite(rec['tcpa']), rec['tcpa'], np.nan)
    tc = np.clip(tc, -0.08 * tmax, tmax)
    if st:
        ax_t.axhspan(0, st['t_safe'], color='#fb8c00', alpha=0.08, lw=0)
        ax_t.text(0.0, st['t_safe'], ' warning window TCPA < %.0f s' % st['t_safe'], fontsize=6, color='#e65100',
                  va='bottom')
    ax_t.plot(t, tc, color='0.8', lw=1)
    ax_t.plot(t[:k + 1], tc[:k + 1], color=CPA_C, lw=1.3)
    ax_t.set_ylabel('TCPA [s]', fontsize=7); ax_t.set_ylim(-0.1 * tmax, 1.05 * tmax)
    # domain margin and CRI
    mg = np.clip(rec['margin'], 0, 5)
    ax_m.plot(t, mg, color='0.8', lw=1)
    ax_m.plot(t[:k + 1], mg[:k + 1], color='#e53935', lw=1.3, label='domain margin (<1 inside)')
    ax_m.axhline(1.0, color='#e53935', ls='--', lw=0.8)
    if st:
        ax_m.plot(t[:k + 1], rec['cri'][:k + 1] * 5, color='#00897b', lw=1.1, label='CRI x5')
    ax_m.set_ylim(0, 5.2); ax_m.set_ylabel('margin', fontsize=7); ax_m.legend(loc='upper left', fontsize=6)
    # severity
    names = sev_names(rec)
    ax_s.step(t, rec['sev'], where='post', color='0.8', lw=1)
    ax_s.step(t[:k + 1], rec['sev'][:k + 1], where='post', color='#e53935', lw=1.6)
    if st:
        held = np.where(rec['hold'][:k + 1] > 0)[0]
        ax_s.plot(t[held], rec['sev'][held], 'o', color='k', ms=2.5, zorder=5)
        ax_s.text(0.99, 0.95, 'phase: %s' % PHASE_NAMES[int(rec['phase'][k])], transform=ax_s.transAxes, ha='right',
                  va='top', fontsize=7)
    else:
        ax_s.step(t[:k + 1], rec['evading'][:k + 1] * 0.5, where='post', color='#ffb300', lw=1.1)
    ax_s.set_yticks([0, 1, 2, 3])
    ax_s.set_yticklabels([names[i].replace('_', ' ') for i in range(4)], fontsize=6)
    ax_s.set_ylim(-0.15, 3.3)
    # command
    if len(rec['action']):
        tab = np.asarray([rec['action_table'][a] for a in rec['action']], dtype=float)
        ta = t[1:len(rec['action']) + 1]
        kk = min(k, len(ta))
        ax_a.step(ta, tab[:, 1], where='post', color='0.8', lw=1)
        ax_a.step(ta[:kk], tab[:kk, 1], where='post', color=OWN_C, lw=1.4, label='turn cmd [deg/s], + = port')
        if np.any(tab[:, 0] != 0):
            ax_a.step(ta[:kk], tab[:kk, 0] * 10, where='post', color='#2e7d32', lw=1.4, label='accel x10 [m/s2]')
        ax_a.legend(loc='upper left', fontsize=6)
        lim = max(abs(tab[:, 1]).max(), 1.0) * 1.4
        ax_a.set_ylim(-lim, lim)
    ax_a.set_ylabel('command', fontsize=7)
    if ax_v is not None:
        sp = rec['own'][:, 3]
        ax_v.plot(t, sp, color='0.8', lw=1)
        ax_v.plot(t[:k + 1], sp[:k + 1], color=OWN_C, lw=1.5, label='attacker speed')
        if rec['has_target']:
            ax_v.plot(t[:k + 1], rec['tgt'][:k + 1, 3], color=TGT_C, lw=1.0, ls='--', label='target speed')
        lim = rec.get('limits')
        top = float(sp.max())
        if lim:
            for v, lab in ((lim['v_range'][0], 'min'), (lim['v_range'][1], 'max')):
                ax_v.axhline(v, color='k', ls=':', lw=0.8)
                ax_v.text(t[-1], v, '%s %.1f m/s ' % (lab, v), fontsize=6, ha='right', va='bottom')
            top = max(top, lim['v_range'][1])
        ax_v.set_ylim(0, top * 1.15 + 0.5)
        ax_v.set_ylabel('m/s', fontsize=7)
        ax_v.legend(loc='upper left', fontsize=6)
    (ax_v if ax_v is not None else ax_a).set_xlabel('time [s]', fontsize=7)
    for ax in axes:
        ax.axvline(k * rec['dt'], color='k', lw=0.8, alpha=0.6)


# ------------------------------------------------------------------ figures
def _scenario_text(rec):
    try:
        from attack_scenarios import ATTACK_SCENARIOS
        return ATTACK_SCENARIOS.get(rec['scenario'], {}).get('text', '')
    except Exception:
        return ''


def header(fig, rec, label=''):
    m, st = rec['meta'], rec.get('standard')
    if st:
        l1 = '%s   |   %s  (%s): %s' % (label, rec['scenario'], m.get('rule', ''), _scenario_text(rec))
        tc = m.get('initial_tcpa_s', -1)
        l2 = ('DCPA band %s: initial range %.0f m, DCPA %.0f m, TCPA %s, bearing %+.0f deg, encounter %s   |   '
              'target sails a fixed track at %.1f m/s') % (
            m.get('band'), m.get('initial_range_m', 0), m.get('initial_dcpa_m', 0), ('%.0f s' % tc) if tc and tc > 0
            else 'none', m.get('initial_bearing_deg', 0), m.get('initial_encounter', ''), rec['tgt'][0, 3])
        l3 = ('collision standard (Ship_envre_v2, scale %s x%.3f): ship L %.0f m | 1 CPA warning DCPA < %.0f m within '
              '%.0f s | 2 domain violation inside %.0f x %.0f m | 3 collision < %.0f m   ||   lifecycle (v3): success = '
              'severity >= %d held %d decisions, failure = clear (range > %.0f m opening %.0f s) or off map') % (
            st['scale_name'], st['scale'], st['L'], st['d_safe'], st['t_safe'], 2 * st['domain_a'], 2 * st['domain_b'],
            st['collision_dist'], st['success_severity'], st['hold_steps'], st['r_clear'], st['n_clear_s'])
    else:
        l1 = '%s   |   %s task, %s' % (label, rec['task'], rec['scenario'])
        l2 = 'target: %s' % rec['automation']
        l3 = 'severity: 1 inside the risk ring, 2 inside the contact zone, 3 hull contact'
    lim = rec.get('limits')
    l4 = ''
    if lim:
        box = rec['map_box']
        l4 = ('attacker limits (%s control): long %.0f..%.0f m, lat %.0f..%.0f m, speed %.1f..%.1f m/s, '
              'accel %+.2f..%+.2f m/s2, turn rate +-%.2f deg/s, course change %s, %d x %d action levels') % (
            lim['control'], box[0], box[1], box[2], box[3], lim['v_range'][0], lim['v_range'][1], lim['a_range'][0],
            lim['a_range'][1], lim['rot_range'][1], ('+-%.0f deg' % lim['cog_limit']) if lim['cog_limit'] < 180 else 'free',
            lim['acc_levels'], lim['rot_levels'])
    fig.text(0.01, 0.99, l1, fontsize=12, weight='bold', va='top')
    fig.text(0.01, 0.958, l2, fontsize=9, va='top')
    fig.text(0.01, 0.935, l3, fontsize=8, color='0.3', va='top')
    if l4:
        fig.text(0.01, 0.913, l4, fontsize=8, color='#0d47a1', va='top')


def animate_episode(rec, path, fps=8, max_frames=200, label='', hold_s=2.0, dpi=85):
    n = len(rec['own'])
    frames = list(np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int)))
    frames += [n - 1] * int(round(fps * hold_s))
    fig = plt.figure(figsize=(19.5, 9.2))
    gs = fig.add_gridspec(6, 3, width_ratios=[1.0, 1.3, 0.95], left=0.035, right=0.99, top=0.87, bottom=0.06,
                          wspace=0.17, hspace=0.38)
    ax_full = fig.add_subplot(gs[:, 0])
    ax_map = fig.add_subplot(gs[:, 1])
    axes = [fig.add_subplot(gs[i, 2]) for i in range(6)]
    header(fig, rec, label or os.path.splitext(os.path.basename(path))[0])
    cam, ext = camera_track(rec), full_extent(rec)

    def update(k):
        draw_full(ax_full, rec, k, extent=ext)
        draw_map(ax_map, rec, k, cam=cam)
        draw_panels(axes, rec, k)
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def key_frames(rec):
    n = len(rec['own'])
    names = sev_names(rec)
    ev = [(0, 'start')]
    sev = rec['sev']
    for level, lab in ((1, 'first %s' % names[1].replace('_', ' ')), (2, '%s entered' % names[2].replace('_', ' '))):
        idx = np.where(sev >= level)[0]
        if len(idx):
            ev.append((int(idx[0]), lab))
    if rec['has_target']:
        ev.append((int(np.argmin(rec['d'])), 'closest approach'))
    ev.append((n - 1, 'end'))
    out = {}
    for k, lab in ev:
        out[k] = lab if k not in out else out[k] + ' / ' + lab
    return sorted(out.items())


def storyboard(rec, path, n_panels=None):
    frames = key_frames(rec)
    fig, axs = plt.subplots(1, len(frames), figsize=(4.3 * len(frames), 5.0))
    ext = full_extent(rec)
    names = sev_names(rec)
    for ax, (k, lab) in zip(np.atleast_1d(axs), frames):
        draw_full(ax, rec, k, small=True, extent=ext,
                  title='%s\nt=%.0f s  range %.0f m  %s' % (lab, k * rec['dt'], rec['d'][k], names[int(rec['sev'][k])]))
    ev = rec['eval']
    oc = rec['outcome']
    grade = (', grade %s %s, score %.0f' % (ev['event_grade'], ev['grade_name'], ev['event_score'])
             if 'grade_name' in ev else '')
    fig.suptitle('%s | %s %s -> %s (%s) in %d decisions%s | min range %.0f m, min DCPA %.0f m' % (
        rec['task'], rec['scenario'], ('band ' + str(rec['meta'].get('band'))) if rec['meta'].get('band') else '',
        OUTCOME_TEXT.get(oc, oc), rec.get('termination'), rec['steps'], grade, rec['d'].min() if rec['has_target'] else 0,
        np.nanmin(rec['dcpa']) if rec['has_target'] else 0), fontsize=10, color=OUTCOME_C.get(oc, 'k'))
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def animate_grid(recs, path, fps=8, max_frames=200, hold_s=2.0, dpi=72):
    n = max(len(r['own']) for r in recs)
    frames = list(np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))) + [n - 1] * int(round(fps * hold_s))
    cols = min(4, len(recs)); rows = int(np.ceil(len(recs) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.8 * rows), squeeze=False)
    exts = [full_extent(r) for r in recs]

    def update(k):
        for i, rec in enumerate(recs):
            ax = axs[i // cols][i % cols]
            kk = min(k, len(rec['own']) - 1)
            end = kk == len(rec['own']) - 1
            draw_full(ax, rec, kk, small=True, extent=exts[i],
                      title=('%s\nt=%.0f s  range %.0f m  %s' % (rec['scenario'], kk * rec['dt'], rec['d'][kk],
                                                                sev_names(rec)[int(rec['sev'][kk])].replace('_', ' '))))
            if end:
                ax.title.set_color(OUTCOME_C.get(rec['outcome'], 'k'))
                ax.text(0.5, 0.03, OUTCOME_TEXT.get(rec['outcome'], rec['outcome']), transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=8, color='white', weight='bold',
                        bbox=dict(boxstyle='round', fc=OUTCOME_C.get(rec['outcome'], '#333'), ec='none', alpha=0.9))
        for j in range(len(recs), rows * cols):
            axs[j // cols][j % cols].axis('off')
        return []

    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    ani.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ scenario catalogue
def draw_standard_panel(ax, std):
    L, W = std['L'], std['W']
    a, b, ring, col = std['domain_a'], std['domain_b'], std['d_safe'], std['collision_dist']
    ax.set_facecolor('white')
    ax.add_patch(Circle((0, 0), ring, fc='#fff3e0', ec='#fb8c00', ls=':', lw=1.4, zorder=1))
    ax.add_patch(Ellipse((0, 0), 2 * a, 2 * b, fc='#ffebee', ec='#e53935', ls='--', lw=1.4, alpha=0.9, zorder=2))
    ax.add_patch(Circle((0, 0), col, fc='#e0e0e0', ec='k', lw=1.0, zorder=3))
    ax.add_patch(hull_patch(0, 0, 0, L, W, 1.0, TGT_C, z=4))
    R = 1.5 * max(ring, a)
    ax.annotate('1 CPA warning\nDCPA < %.0f m, TCPA < %.0f s' % (ring, std['t_safe']), xy=(-0.7 * ring, 0.7 * ring),
                xytext=(-0.95 * R, 0.72 * R), fontsize=8, color='#e65100', arrowprops=dict(arrowstyle='-', color='#e65100'))
    ax.annotate('2 domain violation\n%.0f x %.0f m (4 L x 1.6 L)' % (2 * a, 2 * b), xy=(0.8 * a, -0.6 * b),
                xytext=(0.05 * R, -0.8 * R), fontsize=8, color='#c62828', arrowprops=dict(arrowstyle='-', color='#c62828'))
    ax.annotate('3 collision\ncentre distance < %.0f m (1 L)' % col, xy=(-0.7 * col, -0.7 * col),
                xytext=(-0.95 * R, -0.62 * R), fontsize=8, arrowprops=dict(arrowstyle='-', color='k'))
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('collision standard (v2) at scale %s: ship L %.0f m\nsuccess: severity >= %d held %d decisions; '
                 'failure: clear (> %.0f m, %.0f s) / off map' % (std['scale_name'], L, std['success_severity'],
                                                                  std['hold_steps'], std['r_clear'], std['n_clear_s']),
                 fontsize=8.5)


def catalogue(samples, path, std, band='', title=None):
    """samples: {scenario name: [make_attack_scenario() dicts]} drawn in the attacker's start frame (at 0,0 heading east)."""
    names = list(samples)
    n = len(names) + 1
    ncols = 3 if n <= 9 else 4
    nrows = int(math.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.9 * nrows), squeeze=False)
    a, b, ring = std['domain_a'], std['domain_b'], std['d_safe']
    for ax, name in zip(axs.flat, names):
        ax.set_facecolor(SEA)
        pts = []
        for j, sc in enumerate(samples[name]):
            m, route, v = sc['meta'], np.asarray(sc['route']), sc['speed']
            dvec = (route[1] - route[0]) / max(np.linalg.norm(route[1] - route[0]), 1e-9)
            tc = m['initial_tcpa_s']
            horizon = 1.5 * tc if tc and tc > 0 else 0.5 * std['t_meet'][1]
            own_end = np.array([v * horizon, 0.0])
            t0 = route[0]
            t_end = t0 + dvec * v * horizon
            al, lw = (1.0, 1.8) if j == 0 else (0.25, 0.9)
            ax.plot([0, own_end[0]], [0, own_end[1]], color=OWN_C, lw=lw, alpha=al, zorder=4)
            ax.plot([t0[0], t_end[0]], [t0[1], t_end[1]], color=TGT_C, lw=lw, alpha=al, zorder=4)
            ax.plot(t0[0], t0[1], 'o', color=TGT_C, ms=4 if j == 0 else 2.5, alpha=al, zorder=5)
            if j == 0:
                pts += [[0, 0], own_end, t0, t_end]
                if tc and tc > 0:
                    po, pt = np.array([v * tc, 0.0]), t0 + dvec * v * tc
                    psi_t = math.degrees(math.atan2(dvec[1], dvec[0]))
                    ax.add_patch(Ellipse(pt, 2 * a, 2 * b, angle=psi_t, fc='none', ec='#e53935', ls='--', lw=1.0, zorder=5))
                    ax.add_patch(Circle(pt, ring, fc='none', ec='#fb8c00', ls=':', lw=1.0, zorder=5))
                    ax.plot([po[0], pt[0]], [po[1], pt[1]], color=CPA_C, lw=1.5, zorder=6)
                    ax.plot(*po, 's', mfc='none', mec=OWN_C, ms=5, zorder=6)
                    ax.plot(*pt, 's', mfc='none', mec=TGT_C, ms=5, zorder=6)
                    ax.text(0.5 * (po[0] + pt[0]), 0.5 * (po[1] + pt[1]), '  DCPA %.0f m' % m['initial_dcpa_m'],
                            fontsize=7, color=CPA_C, zorder=7)
                else:
                    psi_t = math.degrees(math.atan2(dvec[1], dvec[0]))
                    ax.add_patch(Ellipse(t0, 2 * a, 2 * b, angle=psi_t, fc='none', ec='#e53935', ls='--', lw=1.0, zorder=5))
                    ax.add_patch(Circle(t0, ring, fc='none', ec='#fb8c00', ls=':', lw=1.0, zorder=5))
                    ax.text(t0[0], t0[1] - ring, 'abeam %.0f m, astern %.0f m' % (abs(t0[1]), abs(t0[0])),
                            fontsize=7, color=CPA_C, ha='center', va='top', zorder=7)
                first = (m, t0, dvec)
        pts = np.asarray(pts)
        lo, hi = pts.min(0), pts.max(0)
        c = 0.5 * (lo + hi)
        half = 0.5 * float(np.max(hi - lo)) + ring
        m, t0, dvec = first
        hs = max(1.0, 0.04 * 2 * half / std['L'])
        ax.add_patch(hull_patch(0, 0, 0, std['L'], std['W'], hs, OWN_C, z=8))
        ax.add_patch(hull_patch(t0[0], t0[1], math.atan2(dvec[1], dvec[0]), std['L'], std['W'], hs, TGT_C, z=8))
        ax.text(0, 0, '   A', fontsize=9, color=OWN_C, va='top', weight='bold', zorder=9)
        ax.text(t0[0], t0[1], '   T', fontsize=9, color=TGT_C, va='top', weight='bold', zorder=9)
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect('equal')
        step = nice_step(2 * half, 5)
        xs = np.arange(np.ceil((c[0] - half) / step) * step, c[0] + half, step)
        ys = np.arange(np.ceil((c[1] - half) / step) * step, c[1] + half, step)
        ax.set_xticks(xs); ax.set_yticks(ys)
        ax.set_xticklabels([_fmt_km(x, step) for x in xs], fontsize=7)
        ax.set_yticklabels([_fmt_km(y, step) for y in ys], fontsize=7)
        ax.grid(True, color=GRID, lw=0.6)
        tc = m['initial_tcpa_s']
        ax.set_title('%s  (%s)\ninitial range %.0f m, DCPA %.0f m, TCPA %s' % (
            name, m['rule'], m['initial_range_m'], m['initial_dcpa_m'], ('%.0f s' % tc) if tc and tc > 0 else 'none'),
            fontsize=9)
    draw_standard_panel(axs.flat[len(names)], std)
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    fig.suptitle(title or ('attack scenarios, initial DCPA band "%s": attacker A (blue) and fixed-track target T (red) '
                           'both holding course for 1.5 x TCPA\nbold = one draw, faint = other draws; at the CPA: target '
                           'domain (red dashed), d_safe ring (orange), DCPA (purple); parallel lanes: zones at the start; '
                           'axes in km' % band), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


__all__ = ['record_episode', 'animate_episode', 'storyboard', 'animate_grid', 'catalogue', 'draw_map', 'draw_full',
           'draw_panels', 'key_frames', 'SEVERITY_NAMES']
