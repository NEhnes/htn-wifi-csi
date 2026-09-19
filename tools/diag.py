"""Which transmitters did we actually hear, and which support a FAIR A/B?"""
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from analyze import load  # noqa: E402

runs = {}
for name in ("paced", "control"):
    per = load(os.path.join(D, name + ".csv"))
    runs[name] = per
    print(f"=== {name}: transmitters by RSSI (closest first) ===")
    rows = []
    for mac, fr in per.items():
        span = (fr[-1][0] - fr[0][0]) / 1e6 if len(fr) > 1 else 0.0
        rows.append((float(np.median([x[1] for x in fr])), mac, len(fr), span))
    rows.sort(reverse=True)
    for rssi, mac, n, span in rows[:8]:
        rate = n / span if span else 0.0
        print(f"  rssi={rssi:6.1f}  {mac}  n={n:5d}  rate={rate:5.1f} Hz  span={span:5.1f}s")
    print()

pa, co = runs["paced"], runs["control"]
common = set(pa) & set(co)
print(f"MACs present in BOTH runs: {len(common)}")
best = []
for m in common:
    if len(pa[m]) >= 100 and len(co[m]) >= 100:
        best.append((min(len(pa[m]), len(co[m])), m,
                     float(np.median([x[1] for x in pa[m]]))))
best.sort(reverse=True)
print("usable for a fair A/B (>=100 frames in BOTH):")
if not best:
    print("  NONE - no transmitter gave enough frames in both runs")
for n, m, r in best[:6]:
    print(f"  {m}  min_frames={n}  rssi={r:.0f}  "
          f"paced={len(pa[m])}  control={len(co[m])}")
