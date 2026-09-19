"""Does DETECTION survive even though localisation is marginal? Plus figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wifi_sar_sim import (run_trial, Environment, build_scene, measure, TX, RX_X,
                          GX, GY, NX, NY, roi_mask)

CH8 = (1, 3, 5, 7, 9, 11, 13, 2)

# ---------------------------------------------------------------- detection
print("=" * 90)
print("DETECTION (is anyone behind the wall?) under realistic multipath")
print("=" * 90)

def detect_stat(seed, person_present, scatter_db=-6.0):
    rng = np.random.default_rng(seed)
    env = Environment(rng=rng, mean_db=scatter_db)
    ys = np.linspace(0.5, 3.5, 60)
    rx = np.stack([np.full(60, RX_X), ys], axis=1)
    person = np.array([rng.uniform(1.0, 1.75), rng.uniform(1.3, 2.9)])
    empty = build_scene(person, include_person=False)
    scene = build_scene(person, include_person=person_present)
    kw = dict(env=env, channels=list(CH8), n_packets=32, rng=rng, multipath=True)
    base = measure(empty, rx, tx=TX, **kw)
    now = measure(scene, rx, tx=TX, **kw)
    return float(np.max(np.abs(base - now)))          # biggest change on any link

for label, sdb in (("cluttered room", -6.0), ("very reflective", -2.0)):
    with_p = np.array([detect_stat(3000 + s, True, sdb) for s in range(30)])
    without = np.array([detect_stat(4000 + s, False, sdb) for s in range(30)])
    thr = np.percentile(without, 95)
    tpr = float(np.mean(with_p > thr))
    print(f"{label:<18s} empty-room 95th pct = {thr:4.1f} dB | "
          f"person detected {tpr*100:3.0f}% of the time at 5% false alarms")

# ---------------------------------------------------------------- figures
print("\nrendering figure...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
cases = [
    ("Idealised physics\n(no multipath)", dict(multipath=False, channels=(6,), n_tx=4)),
    ("Realistic multipath\n4 TX, 8 channels", dict(multipath=True, channels=CH8, n_tx=4)),
    ("Realistic multipath\n8 TX, 8 channels", dict(multipath=True, channels=CH8, n_tx=8)),
]
for ax, (title, kw) in zip(axes, cases):
    err, con, img, truth, peak = run_trial(mode="differential", seed=1234,
                                           return_image=True, **kw)
    m = roi_mask().reshape(NX, NY)
    show = np.where(m, img, np.nan)
    ax.imshow(show.T, origin="lower", extent=[0, GX.max(), 0, GY.max()],
              aspect="equal", cmap="viridis")
    ax.plot(truth[0], truth[1], "rx", ms=14, mew=3, label="true person")
    ax.plot(peak[0], peak[1], "w+", ms=14, mew=3, label="estimate")
    ax.set_title(f"{title}\nerror {err:.2f} m")
    ax.set_xlabel("x (m)")
    ax.legend(loc="upper right", fontsize=8)
axes[0].set_ylabel("y (m)")
fig.suptitle("Through-wall WiFi imaging: reconstruction quality (differential mode)")
fig.tight_layout()
fig.savefig("wifi_imaging_reconstructions.png", dpi=140)

fig2, ax = plt.subplots(figsize=(8.5, 4.2))
labels = ["no multipath", "multipath\nphases ignored", "real multipath\n(fading)",
          "sparse room", "reflective room"]
hits = [100, 98, 48, 95, 40]
ax.bar(labels, hits, color=["#3a7", "#3a7", "#c33", "#7a3", "#c33"])
ax.axhline(50, ls="--", c="k", lw=1)
ax.text(0.02, 52, "coin flip", fontsize=9)
ax.set_ylabel("% of trials person located within 0.5 m")
ax.set_ylim(0, 105)
ax.set_title("What actually limits ESP32 through-wall imaging (40 trials each)")
fig2.tight_layout()
fig2.savefig("wifi_imaging_verdict.png", dpi=140)
print("saved wifi_imaging_reconstructions.png and wifi_imaging_verdict.png")
