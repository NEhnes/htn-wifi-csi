"""Why does multipath break it? Higher trial counts + confidence intervals."""
import numpy as np
from wifi_sar_sim import run_trial

CH1 = (6,)
CH8 = (1, 3, 5, 7, 9, 11, 13, 2)
N = 40


def study(label, **kw):
    errs = []
    for s in range(N):
        e, _ = run_trial(seed=1000 + s, **kw)
        errs.append(e)
    errs = np.array(errs)
    hit = float(np.mean(errs < 0.5))
    se = np.sqrt(hit * (1 - hit) / N)
    print(f"{label:<54s} median {np.median(errs):4.2f} m | "
          f"hit {hit*100:3.0f}% +/- {se*200:2.0f}%")
    return errs


print("=" * 104)
print(f"DIAGNOSIS: why multipath breaks RSSI imaging  ({N} trials, +/- is 2 standard errors)")
print("=" * 104)

print()
print("--- baseline: no multipath at all ---")
study("no multipath, noise + quantisation, 4 TX",
      mode="differential", multipath=False, channels=CH1, n_tx=4)

print()
print("--- multipath present, but phases IGNORED (powers added, no fading) ---")
study("incoherent multipath, 1 ch, 4 TX",
      mode="differential", multipath=True, coherent=False, channels=CH1, n_tx=4)
study("incoherent multipath, 8 ch, 8 TX",
      mode="differential", multipath=True, coherent=False, channels=CH8, n_tx=8)

print()
print("--- real coherent multipath (fading) ---")
study("coherent multipath, 1 ch, 4 TX",
      mode="differential", multipath=True, channels=CH1, n_tx=4)
study("coherent multipath, 8 ch, 4 TX",
      mode="differential", multipath=True, channels=CH8, n_tx=4)
study("coherent multipath, 8 ch, 8 TX",
      mode="differential", multipath=True, channels=CH8, n_tx=8)
study("coherent multipath, 8 ch, 8 TX, 200 pkts",
      mode="differential", multipath=True, channels=CH8, n_tx=8, n_packets=200)

print()
print("--- how much does the richness of the multipath matter? ---")
for db, name in ((-16.0, "sparse room (weak reflections)"),
                 (-10.0, "moderate"),
                 (-6.0, "cluttered room (default)"),
                 (-2.0, "very reflective (metal/concrete)")):
    study(f"{name:<34s} 8 ch, 8 TX", mode="differential", multipath=True,
          channels=CH8, n_tx=8, scatter_db=db)

print()
print("--- static-scene imaging (no baseline), realistic ---")
study("absolute mode, coherent multipath, 8 ch, 8 TX",
      mode="absolute", multipath=True, channels=CH8, n_tx=8)
