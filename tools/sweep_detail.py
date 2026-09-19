"""Characterise a sweep log without assuming constant walking speed."""
import sys

import numpy as np

path = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

r = np.genfromtxt(path, delimiter=",", names=True)
t, s, sh, p = r["t"], r["score"], r["shape"], r["power_db"]

print(f"duration {t.max():.1f}s, {len(t)} samples "
      f"({len(t)/t.max():.1f} Hz display rate)\n")

# ASCII timeline: bucket into ~70 columns, show max score per bucket
NB = 70
edges = np.linspace(t.min(), t.max(), NB + 1)
buck = [s[(t >= edges[i]) & (t < edges[i + 1])] for i in range(NB)]
peaks = np.array([b.max() if b.size else np.nan for b in buck])
levels = " .:-=+*#%@"
row = "".join(levels[int(np.clip((v + 2) / 10 * (len(levels) - 1), 0,
                                 len(levels) - 1))] if np.isfinite(v) else " "
              for v in peaks)
print("score timeline (each col ~%.1fs, '@'=high):" % ((t.max() - t.min()) / NB))
print("  |" + row + "|")
print(f"  peak={np.nanmax(peaks):.1f}  median={np.median(s):.1f}\n")

for th in (2.0, 3.0, 4.0, 5.0):
    above = s > th
    if not above.any():
        print(f"  thr {th:.1f}: never crossed")
        continue
    idx = np.where(above)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 2)[0] + 1)
    runs = [q for q in runs if len(q) >= 2]
    durs = [t[q[-1]] - t[q[0]] for q in runs]
    print(f"  thr {th:.1f}: {above.mean()*100:5.1f}% of time, "
          f"{len(runs):2d} crossings, longest {max(durs) if durs else 0:.2f}s, "
          f"total {sum(durs):.1f}s")

print(f"\nshape channel: median={np.median(sh):.2f} max={sh.max():.2f}")
print(f"power channel: median={np.median(p):+.2f}dB  "
      f"min={p.min():+.2f}  max={p.max():+.2f}")

# Correlation tells us whether detections are body-like (shape+power together)
m = s > thr
if m.sum() > 5:
    print(f"\nduring detections: shape={sh[m].mean():.2f} "
          f"power={p[m].mean():+.2f}dB")
    print(f"during quiet     : shape={sh[~m].mean():.2f} "
          f"power={p[~m].mean():+.2f}dB")
