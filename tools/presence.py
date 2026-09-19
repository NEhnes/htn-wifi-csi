"""Presence detection: can we tell 'person in the path' from 'empty room'?

This is a far weaker claim than breathing - no periodic signal required, just a
statistical difference between the two conditions. Uses the captures we already
have: paced.csv (person present) vs control.csv (absent).

Reports AUC per feature. AUC 0.5 = coin flip, 1.0 = perfect separation.

IMPORTANT CAVEAT printed at the end: one capture per class means 'presence' is
confounded with 'time of capture'. Alternating captures are needed to confirm.

Usage: python presence.py <mac> [window_seconds]
"""
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from analyze import load  # noqa: E402


def windows(frames, wsec):
    """Split a capture into fixed-duration windows of complex CSI."""
    t = np.array([f[0] for f in frames], dtype=np.float64) / 1e6
    t -= t[0]
    rssi = np.array([f[1] for f in frames], dtype=np.float64)
    raw = np.stack([f[2] for f in frames])
    im = raw[:, 0::2].astype(np.float64)
    re = raw[:, 1::2].astype(np.float64)
    amp = np.sqrt(re ** 2 + im ** 2)          # (T, 64)

    out = []
    n = int(t[-1] // wsec)
    for i in range(n):
        m = (t >= i * wsec) & (t < (i + 1) * wsec)
        if m.sum() < 10:
            continue
        a = amp[m]
        out.append(dict(
            mean_amp=a.mean(),
            std_amp=a.std(axis=0).mean(),      # temporal variability
            mean_rssi=rssi[m].mean(),
            profile=a.mean(axis=0),            # 64-dim fading profile
            spread=a.mean(axis=0).std(),       # how uneven across subcarriers
        ))
    return out


def auc(pos, neg):
    """Rank-based AUC; symmetric so 0.5 means no separation either way."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, allv.size + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    r = ranks[:pos.size].sum()
    a = (r - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return max(a, 1 - a), a


def main():
    mac = sys.argv[1]
    wsec = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    data = {}
    for name, label in (("paced", "PRESENT"), ("control", "ABSENT")):
        per = load(os.path.join(D, name + ".csv"))
        if mac not in per:
            print(f"{name}: transmitter not found"); return
        data[label] = windows(per[mac], wsec)
        print(f"{label:<8} {len(data[label]):3d} windows of {wsec:.0f}s")

    P, A = data["PRESENT"], data["ABSENT"]
    if not P or not A:
        print("not enough windows"); return

    print(f"\n{'feature':<12}{'present':>11}{'absent':>11}{'delta':>10}{'AUC':>8}")
    for key in ("mean_rssi", "mean_amp", "std_amp", "spread"):
        p = [w[key] for w in P]
        a = [w[key] for w in A]
        sep, _ = auc(p, a)
        print(f"{key:<12}{np.mean(p):>11.2f}{np.mean(a):>11.2f}"
              f"{np.mean(p)-np.mean(a):>10.2f}{sep:>8.3f}")

    # Multivariate: nearest-centroid on the 64-subcarrier fading profile,
    # with a held-out split so it is not just memorising.
    Pp = np.stack([w["profile"] for w in P])
    Ap = np.stack([w["profile"] for w in A])
    rng = np.random.default_rng(0)
    accs = []
    for _ in range(200):
        ip = rng.permutation(len(Pp)); ia = rng.permutation(len(Ap))
        hp, ha = len(Pp) // 2, len(Ap) // 2
        cP = Pp[ip[:hp]].mean(axis=0)
        cA = Ap[ia[:ha]].mean(axis=0)
        teP, teA = Pp[ip[hp:]], Ap[ia[ha:]]
        okP = np.linalg.norm(teP - cP, axis=1) < np.linalg.norm(teP - cA, axis=1)
        okA = np.linalg.norm(teA - cA, axis=1) < np.linalg.norm(teA - cP, axis=1)
        accs.append((okP.sum() + okA.sum()) / (len(teP) + len(teA)))
    acc = float(np.mean(accs))
    print(f"\nnearest-centroid on 64-subcarrier profile: "
          f"held-out accuracy = {acc*100:.1f}%  (chance = 50%)")

    print("\n" + "=" * 62)
    if acc > 0.9:
        print("STRONG separation between present and absent.")
    elif acc > 0.7:
        print("MODERATE separation.")
    else:
        print("WEAK/NO separation - presence not detectable this way.")
    print("CAVEAT: one capture per condition means 'presence' is confounded")
    print("with 'when it was recorded'. Confirm with ALTERNATING captures")
    print("before believing any of this.")


if __name__ == "__main__":
    main()
