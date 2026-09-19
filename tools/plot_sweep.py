"""Plot a logged walk-through so you can SEE the detection field.

Do a transect: start well outside the link, walk slowly and steadily straight
across the TX-RX line at the midpoint, continue out the far side. If you know
roughly how far you walked and how long it took, this converts the time axis to
approximate distance and measures the width of the sensitive zone.

Usage:
  python plot_sweep.py sweep.csv [--span 3.0] [--thresh 4]
    --span = total metres walked during the log (default: time axis only)
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--span", type=float, default=None)
ap.add_argument("--thresh", type=float, default=4.0)
a = ap.parse_args()

rows = np.genfromtxt(a.csv, delimiter=",", names=True)
t = rows["t"]
s = rows["score"]
p = rows["power_db"]

x = None
if a.span:
    # assume constant walking speed across the whole log
    x = (t - t.min()) / (t.max() - t.min()) * a.span - a.span / 2.0

above = s > a.thresh
print(f"samples={len(t)}  duration={t.max()-t.min():.1f}s")
print(f"score: min={s.min():.1f}  max={s.max():.1f}  median={np.median(s):.1f}")
print(f"fraction above threshold {a.thresh}: {above.mean()*100:.1f}%")

if above.any():
    # widest contiguous run above threshold = the detection zone crossing
    idx = np.where(above)[0]
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    run = max(splits, key=len)
    dt = t[run[-1]] - t[run[0]]
    print(f"widest continuous detection: {dt:.2f}s "
          f"({len(run)} samples), peak score {s[run].max():.1f}")
    if x is not None:
        dx = abs(x[run[-1]] - x[run[0]])
        print(f"  -> approx {dx*100:.0f} cm wide at {a.span:.1f} m walked")
        print(f"  (first Fresnel zone theory for a 1.5 m link: ~43 cm)")
else:
    print("never crossed the threshold - walk closer to the midpoint")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(x if x is not None else t, s, lw=1.2)
    ax[0].axhline(a.thresh, color="r", ls="--", label=f"threshold {a.thresh}")
    ax[0].fill_between(x if x is not None else t, 0, s,
                       where=above, color="r", alpha=0.15)
    ax[0].set_ylabel("presence score (sigma)")
    ax[0].set_title("Detection field transect")
    ax[0].legend()
    ax[1].plot(x if x is not None else t, p, lw=1.2, color="purple")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_ylabel("power change (dB)")
    ax[1].set_xlabel("position across link (m)" if x is not None else "time (s)")
    fig.tight_layout()
    out = a.csv.replace(".csv", ".png")
    fig.savefig(out, dpi=130)
    print(f"plot -> {out}")
except Exception as e:
    print(f"(plot skipped: {e})")
