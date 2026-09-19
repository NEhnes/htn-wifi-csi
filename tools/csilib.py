"""Shared CSI parsing + feature extraction for the multi-link sensing build."""
import numpy as np

N_GROUPS = 8          # collapse 64 subcarriers into 8 bands
FEATS_PER_LINK = N_GROUPS + 7


def parse_line(s):
    """'CSIX,<t_us>,<mac>,<rssi>,<ch>,<len>,<hex>' -> (t_s, rssi, amp[64])."""
    p = s.split(",")
    if len(p) < 7 or p[0] not in ("CSI", "CSIX"):
        return None
    try:
        t = int(p[1]); rssi = int(p[3]); n = int(p[5])
    except ValueError:
        return None
    if p[0] == "CSIX":
        h = p[6].strip()
        if len(h) < 2 * n:
            return None
        try:
            arr = np.frombuffer(bytes.fromhex(h[:2 * n]), dtype=np.int8)
        except ValueError:
            return None
    else:
        vals = [v for v in p[6:] if v != ""]
        if len(vals) < n:
            return None
        try:
            arr = np.array([int(v) for v in vals[:n]], dtype=np.int8)
        except ValueError:
            return None
    im = arr[0::2].astype(np.float64)
    re = arr[1::2].astype(np.float64)
    return t / 1e6, float(rssi), np.sqrt(re ** 2 + im ** 2)


def link_features(rows, dur):
    """rows = [(t_pc, rssi, amp[64]), ...] within one window.

    Returns a fixed-length vector describing this link during the window.
    Deliberately compact: with only minutes of training data, a few robust
    features beat a high-dimensional representation.
    """
    if len(rows) < 6:
        return None
    amp = np.stack([r[2] for r in rows])            # (T, 64)
    rssi = np.array([r[1] for r in rows])
    t = np.array([r[0] for r in rows])

    prof = amp.mean(axis=0)                          # per-subcarrier mean
    groups = prof.reshape(N_GROUPS, -1).mean(axis=1)  # 8 band means

    # temporal variability: the key discriminator between STILL and MOVING
    tstd = amp.std(axis=0).mean()
    series = amp.mean(axis=1)                        # overall amplitude vs time
    dif = np.abs(np.diff(series)).mean() if series.size > 1 else 0.0

    # energy in a motion band (0.5-3 Hz) of the amplitude time series
    motion = 0.0
    if series.size >= 16 and t[-1] > t[0]:
        fs = len(t) / (t[-1] - t[0])
        x = series - series.mean()
        sp = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        fr = np.fft.rfftfreq(x.size, 1.0 / max(fs, 1e-6))
        band = (fr >= 0.5) & (fr <= 3.0)
        tot = sp.sum()
        motion = float(sp[band].sum() / tot) if tot > 0 else 0.0

    return np.array([
        prof.mean(),            # overall amplitude
        prof.std(),             # spread across subcarriers (fading shape)
        rssi.mean(),
        rssi.std(),
        tstd,                   # temporal jitter
        dif,                    # frame-to-frame change
        motion,                 # fraction of energy in the motion band
        *groups,
    ], dtype=np.float64)


def feature_names():
    base = ["amp_mean", "amp_spread", "rssi_mean", "rssi_std",
            "t_std", "frame_diff", "motion_frac"]
    return base + [f"band{i}" for i in range(N_GROUPS)]


def windowize(rows, wsec):
    """Split [(t_pc, rssi, amp)] into consecutive windows of wsec seconds."""
    if not rows:
        return []
    t0 = rows[0][0]
    out, cur, idx = [], [], 0
    for r in rows:
        k = int((r[0] - t0) // wsec)
        if k != idx:
            if cur:
                out.append((idx, cur))
            cur, idx = [], k
        cur.append(r)
    if cur:
        out.append((idx, cur))
    return out
