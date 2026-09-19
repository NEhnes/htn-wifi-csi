"""Fair A/B: same transmitter, paced vs control.

Usage: python compare.py <mac> [expected_hz]
"""
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from analyze import load, amplitudes, resample, rolling_median_helper  # noqa: E402


def stats(frames, expect):
    t, amp = amplitudes(frames)
    fs = max(2.0, min(20.0, len(t) / t[-1]))
    grid, X = resample(t, amp, fs)
    sd = X.std(axis=0)
    alive = np.where(sd > 1e-6)[0]
    keep = alive[np.argsort(-sd[alive])][:12]
    Xd = X[:, keep] - X[:, keep].mean(axis=0)
    ramp = np.arange(grid.size)
    for k in range(Xd.shape[1]):
        Xd[:, k] -= np.polyval(np.polyfit(ramp, Xd[:, k], 1), ramp)
    win = np.hanning(grid.size)[:, None]
    psd = (np.abs(np.fft.rfft(Xd * win, axis=0)) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(grid.size, 1.0 / fs)
    valid = freqs > 0.05
    f, p = freqs[valid], psd[valid]
    w = max(5, int(0.30 / (freqs[1] - freqs[0])) | 1)
    prom = p / np.maximum(rolling_median_helper(p, w), 1e-30)
    band = (f >= 0.15) & (f <= 0.60)
    outb = (~band) & (f < f[-1] * 0.9)
    j = int(np.argmin(np.abs(f - expect)))
    pk = int(np.argmax(prom[band]))
    return dict(n=len(t), span=t[-1], fs=fs, res=freqs[1],
                at_expect=prom[j], best_in=prom[band][pk],
                best_in_f=f[band][pk], best_out=prom[outb].max())


def main():
    mac = sys.argv[1]
    expect = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    print(f"FAIR A/B on transmitter {mac}, expected {expect:.3f} Hz "
          f"({expect*60:.0f} brpm)\n")
    print(f"{'':<10}{'n':>6}{'span':>8}{'fs':>7}{'res':>8}"
          f"{'@expect':>10}{'best_in':>10}{'at Hz':>8}{'best_out':>10}")
    res = {}
    for name in ("paced", "control"):
        per = load(os.path.join(D, name + ".csv"))
        if mac not in per or len(per[mac]) < 60:
            print(f"{name:<10} insufficient frames")
            continue
        s = stats(per[mac], expect)
        res[name] = s
        print(f"{name:<10}{s['n']:>6}{s['span']:>8.0f}{s['fs']:>7.1f}"
              f"{s['res']:>8.3f}{s['at_expect']:>10.2f}"
              f"{s['best_in']:>10.2f}{s['best_in_f']:>8.3f}{s['best_out']:>10.2f}")

    if len(res) == 2:
        a, b = res["paced"]["at_expect"], res["control"]["at_expect"]
        print(f"\nratio paced/control at {expect:.3f} Hz = {a/b if b else float('inf'):.2f}x")
        if a > 4 and a > 2 * b:
            print("VERDICT: breathing signal present.")
        else:
            print("VERDICT: NO breathing signal distinguishable from control.")


if __name__ == "__main__":
    main()
