"""
Through-wall WiFi imaging feasibility simulator (ESP32-class hardware).

Scenario: one fixed ESP32 transmitter on the far side of a partition wall.
A moving ESP32 receiver sweeps a straight path on the near side (robot / rail),
forming a synthetic aperture. We try to reconstruct a top-down image of what is
behind the wall from received signal strength (RSSI) only.

Deliberate design choice: the SIMULATOR uses ray-based multipath physics, while
the RECONSTRUCTION uses the simple "attenuation along a fat line" model that a
real implementation would use. The model mismatch is the point.

Throwaway feasibility code - not hackathon project code.
"""

import numpy as np

rng_global = np.random.default_rng(7)

C = 299_792_458.0
WIFI_CH_FREQ = {ch: (2407 + 5 * ch) * 1e6 for ch in range(1, 14)}  # 2.4 GHz channels

# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------
ROOM_X, ROOM_Y = 3.2, 4.0
CELL = 0.05
NX, NY = int(ROOM_X / CELL), int(ROOM_Y / CELL)

TX = np.array([0.5, 2.0])          # transmitter, far side of the wall
RX_X = 2.8                          # receiver sweep line (corridor side)
WALL_X, WALL_T = 2.0, 0.10          # partition wall
WALL_ATTEN = 40.0                   # dB/m -> 4 dB per pass (drywall-ish)
PERSON_R = 0.20
PERSON_ATTEN = 15.0                 # dB/m -> ~6 dB through a body at 2.4 GHz
CLUTTER_ATTEN = 8.0                 # dB/m, furniture

ROI = (0.8, 1.9, 0.8, 3.2)          # x0, x1, y0, y1 : region we try to image


def cell_centers():
    xs = (np.arange(NX) + 0.5) * CELL
    ys = (np.arange(NY) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return gx, gy


GX, GY = cell_centers()


def build_scene(person_xy, clutter=True, include_person=True):
    """Attenuation map in dB per metre."""
    a = np.zeros((NX, NY))
    wall = (GX >= WALL_X) & (GX < WALL_X + WALL_T)
    a[wall] = WALL_ATTEN
    if clutter:
        # a desk and a metal cabinet inside the room (static clutter)
        desk = (GX > 1.0) & (GX < 1.7) & (GY > 0.9) & (GY < 1.2)
        cab = (GX > 1.5) & (GX < 1.8) & (GY > 2.9) & (GY < 3.3)
        a[desk] = CLUTTER_ATTEN
        a[cab] = 25.0
    if include_person:
        d = np.hypot(GX - person_xy[0], GY - person_xy[1])
        a[d < PERSON_R] = PERSON_ATTEN
    return a


# ----------------------------------------------------------------------------
# Ray tracing: attenuation integral along segments (vectorised)
# ----------------------------------------------------------------------------
def segment_loss(atten, p0, p1, n_samples=96):
    """p0, p1: (M,2). Returns (M,) loss in dB along each segment."""
    t = np.linspace(0.0, 1.0, n_samples)
    pts = p0[:, None, :] + (p1 - p0)[:, None, :] * t[None, :, None]   # (M,T,2)
    ix = np.clip((pts[..., 0] / CELL).astype(int), 0, NX - 1)
    iy = np.clip((pts[..., 1] / CELL).astype(int), 0, NY - 1)
    vals = atten[ix, iy]                                              # dB/m
    seg_len = np.linalg.norm(p1 - p0, axis=1)                         # (M,)
    return vals.mean(axis=1) * seg_len


# ----------------------------------------------------------------------------
# Channel model: direct ray + specular-ish scatterers, complex sum
# ----------------------------------------------------------------------------
class Environment:
    """Fixed multipath geometry for one 'room' realisation."""

    def __init__(self, n_scatter=24, rng=None, mean_db=-6.0):
        rng = rng or rng_global
        self.sx = rng.uniform(0.05, ROOM_X - 0.05, n_scatter)
        self.sy = rng.uniform(0.05, ROOM_Y - 0.05, n_scatter)
        # reflection strengths: a few strong (walls/floor), many weak
        self.gain = 10 ** (rng.normal(mean_db, 4.0, n_scatter) / 20.0)
        self.phase = rng.uniform(0, 2 * np.pi, n_scatter)


def received_power_dbm(atten, rx_pts, env, freq, tx=TX, pos_jitter=0.0, rng=None, coherent=True):
    """Full multipath RSSI in dBm for each receiver position."""
    rng = rng or rng_global
    lam = C / freq
    M = rx_pts.shape[0]
    if pos_jitter > 0:
        rx_pts = rx_pts + rng.normal(0, pos_jitter, rx_pts.shape)

    txp = np.repeat(tx[None, :], M, axis=0)

    # --- direct ray -----------------------------------------------------
    d_dir = np.linalg.norm(rx_pts - txp, axis=1)
    L_dir = segment_loss(atten, txp, rx_pts)
    amp_dir = (lam / (4 * np.pi * d_dir)) * 10 ** (-L_dir / 20.0)
    h = amp_dir * np.exp(-2j * np.pi * d_dir / lam)
    pow_incoh = amp_dir ** 2

    # --- scattered rays -------------------------------------------------
    for k in range(len(env.sx)):
        s = np.array([env.sx[k], env.sy[k]])
        sp = np.repeat(s[None, :], M, axis=0)
        d1 = np.linalg.norm(sp - txp, axis=1)
        d2 = np.linalg.norm(rx_pts - sp, axis=1)
        L1 = segment_loss(atten, txp, sp)
        L2 = segment_loss(atten, sp, rx_pts)
        amp = env.gain[k] * (lam / (4 * np.pi * (d1 + d2))) * 10 ** (-(L1 + L2) / 20.0)
        h = h + amp * np.exp(-2j * np.pi * (d1 + d2) / lam + 1j * env.phase[k])
        pow_incoh = pow_incoh + amp ** 2

    if coherent:
        p_lin = np.abs(h) ** 2
    else:
        p_lin = pow_incoh            # sum of ray powers, no phase cancellation
    return 10 * np.log10(p_lin + 1e-30) + 20.0   # +20 dBm TX power


def measure(atten, rx_pts, env, channels, n_packets=32, rssi_sigma=1.2,
            quantise=True, pos_jitter=0.01, rng=None, multipath=True, tx=TX, coherent=True):
    """
    Average received power across channels and packets, as an ESP32 would report.
    Returns dBm per receiver position.
    """
    rng = rng or rng_global
    acc = np.zeros(rx_pts.shape[0])
    for ch in channels:
        if multipath:
            p = received_power_dbm(atten, rx_pts, env, WIFI_CH_FREQ[ch],
                                   tx=tx, pos_jitter=pos_jitter, rng=rng,
                                   coherent=coherent)
        else:
            # no multipath: direct ray only (idealised)
            lam = C / WIFI_CH_FREQ[ch]
            txp = np.repeat(tx[None, :], rx_pts.shape[0], axis=0)
            d = np.linalg.norm(rx_pts - txp, axis=1)
            L = segment_loss(atten, txp, rx_pts)
            p = 20 * np.log10(lam / (4 * np.pi * d)) - L + 20.0
        # per-packet noise, averaged in the linear domain like real AGC readings
        noise = rng.normal(0, rssi_sigma, (n_packets, p.size)) / np.sqrt(1)
        pk = p[None, :] + noise
        if quantise:
            pk = np.round(pk)          # ESP32 reports integer dBm
        acc += 10 * np.log10(np.mean(10 ** (pk / 10.0), axis=0))
    return acc / len(channels)


# ----------------------------------------------------------------------------
# Reconstruction: the naive model a real implementation would use
# ----------------------------------------------------------------------------
def build_weight_matrix(tx_pts, rx_pts, ellipse_width=0.15):
    """Fresnel-ellipse weights per TX/RX pair: w = 1/sqrt(d) inside first zone."""
    M = rx_pts.shape[0]
    cells = np.stack([GX.ravel(), GY.ravel()], axis=1)       # (N,2)
    W = np.zeros((M, cells.shape[0]))
    for i in range(M):
        t, r = tx_pts[i], rx_pts[i]
        d = np.linalg.norm(r - t)
        d1 = np.linalg.norm(cells - t, axis=1)
        d2 = np.linalg.norm(cells - r, axis=1)
        inside = (d1 + d2 - d) < ellipse_width
        W[i, inside] = 1.0 / np.sqrt(d)
    return W


def reconstruct(W, y, lam_reg=1.0):
    """Ridge regression solved in measurement space (Wilson & Patwari style)."""
    A = W @ W.T + lam_reg * np.eye(W.shape[0])
    return W.T @ np.linalg.solve(A, y)


def roi_mask():
    return ((GX > ROI[0]) & (GX < ROI[1]) & (GY > ROI[2]) & (GY < ROI[3])).ravel()


def score(img, person_xy):
    """Localisation error (m) and contrast (peak / 95th pct of background)."""
    img = np.asarray(img).ravel()
    m = roi_mask()
    vals = np.where(m, img, -np.inf)
    idx = int(np.argmax(vals))
    px, py = GX.ravel()[idx], GY.ravel()[idx]
    err = float(np.hypot(px - person_xy[0], py - person_xy[1]))
    inside = img[m]
    peak = inside.max()
    bg = np.percentile(inside, 95)
    contrast = float(peak / bg) if bg > 0 else float("inf")
    return err, contrast, (px, py)


# ----------------------------------------------------------------------------
# Experiment harness
# ----------------------------------------------------------------------------
def run_trial(mode="differential", multipath=True, channels=(6,), n_packets=32,
              aperture=3.0, n_pos=200, pos_jitter=0.01, rssi_sigma=1.2,
              seed=0, return_image=False, n_tx=1, tx_span=2.0, coherent=True, scatter_db=-6.0):
    rng = np.random.default_rng(seed)
    env = Environment(rng=rng, mean_db=scatter_db)

    # receiver sweep line (the robot / rail)
    n_per_tx = max(20, n_pos // n_tx)
    ys = np.linspace(2.0 - aperture / 2, 2.0 + aperture / 2, n_per_tx)
    rx_line = np.stack([np.full(n_per_tx, RX_X), ys], axis=1)

    # transmitter positions on the far side (extra ESP32s, or a moving TX)
    if n_tx == 1:
        tx_ys = np.array([TX[1]])
    else:
        tx_ys = np.linspace(TX[1] - tx_span / 2, TX[1] + tx_span / 2, n_tx)
    tx_list = [np.array([TX[0], ty]) for ty in tx_ys]

    person = np.array([rng.uniform(1.0, 1.75), rng.uniform(1.3, 2.9)])
    scene_person = build_scene(person, include_person=True)
    scene_empty = build_scene(person, include_person=False)

    kw = dict(env=env, channels=list(channels), n_packets=n_packets,
              rssi_sigma=rssi_sigma, pos_jitter=pos_jitter, rng=rng,
              multipath=multipath, coherent=coherent)

    ys_meas, tx_pts, rx_pts = [], [], []
    for t in tx_list:
        if mode == "variance":
            # person fidgets/sways while the scan happens; static multipath is
            # identical across snapshots, so only the person's effect varies.
            K = 6
            snaps_p, snaps_e = [], []
            for _ in range(K):
                jig = person + rng.normal(0, 0.08, 2)
                snaps_p.append(measure(build_scene(jig, include_person=True),
                                       rx_line, tx=t, **kw))
                snaps_e.append(measure(scene_empty, rx_line, tx=t, **kw))
            v_p = np.var(np.stack(snaps_p), axis=0)
            v_e = np.var(np.stack(snaps_e), axis=0)
            ys_meas.append(np.maximum(v_p - v_e, 0.0))
        elif mode == "differential":
            y_e = measure(scene_empty, rx_line, tx=t, **kw)
            y_p = measure(scene_person, rx_line, tx=t, **kw)
            ys_meas.append(y_e - y_p)
        else:
            lam = C / WIFI_CH_FREQ[list(channels)[0]]
            d = np.linalg.norm(rx_line - t[None, :], axis=1)
            fspl = 20 * np.log10(lam / (4 * np.pi * d)) + 20.0
            ys_meas.append(fspl - measure(scene_person, rx_line, tx=t, **kw))
        tx_pts.append(np.repeat(t[None, :], n_per_tx, axis=0))
        rx_pts.append(rx_line)

    y = np.concatenate(ys_meas)
    tx_pts = np.concatenate(tx_pts)
    rx_pts = np.concatenate(rx_pts)
    truth = person

    W = build_weight_matrix(tx_pts, rx_pts)
    img = reconstruct(W, y, lam_reg=max(1e-3, 0.05 * np.var(y) * len(y)))
    img = img.reshape(NX, NY)
    err, contrast, peak_xy = score(img, truth)
    if return_image:
        return err, contrast, img, truth, peak_xy
    return err, contrast


def sweep(label, n_trials=12, **kw):
    errs, cons = [], []
    for s in range(n_trials):
        e, c = run_trial(seed=100 + s, **kw)
        errs.append(e)
        cons.append(c)
    errs, cons = np.array(errs), np.array(cons)
    hit = float(np.mean(errs < 0.5))
    print(f"{label:<52s} median err {np.median(errs):5.2f} m | "
          f"contrast {np.median(cons):5.2f} | within 0.5 m: {hit*100:3.0f}%")
    return errs, cons


