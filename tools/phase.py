"""Phase-based re-analysis of EXISTING captures (no new data needed).

Amplitude throws away most of the information in CSI. Raw ESP32 phase is
unusable directly: every packet carries a random carrier/sampling frequency
offset. The standard fix is to conjugate-multiply ADJACENT subcarriers -
the common offset cancels, leaving a phase difference that tracks path length,
which is exactly what a moving chest modulates.

Usage: python phase.py <mac> [expected_hz]
"""
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from analyze import load, resample, rolling_median_helper  # noqa: E402


def features(frames):
    """Return time vector and several phase/amplitude derived feature sets."""
    t = np.array([f[0] for f in frames], dtype=np.float64) / 1e6
    raw = np.stack([f[2] for f in frames])
    im = raw[:, 0::2].astype(np.float64)
    re = raw[:, 1::2].astype(np.float64)
    c = re + 1j * im                        # (T, 64) complex CSI

    # Conjugate product of adjacent subcarriers cancels the per-packet offset.
    d = c[:, :-1] * np.conj(c[:, 1:])       # (T, 63)

    feats = {
        "amplitude":      np.abs(c),
        "phase_diff":     np.angle(d),
        "conj_real":      np.real(d),
        "conj_imag":      np.imag(d),
        "amp_ratio":      np.abs(c[:, :-1]) / np.maximum(np.abs(c[:, 1:]), 1e-9),
    }
    return t - t[0], feats


def prominence_at(t, X, expect):
    fs = max(2.0, min(25.0, len(t) / t[-1]))
    grid, R = resample(t, X, fs)
    if grid.size < 64:
        return None
    sd = R.std(axis=0)
    alive = np.where(sd > 1e-9)[0]
    if alive.size == 0:
        return None
    keep = alive[np.argsort(-sd[alive])][:12]
    Xd = R[:, keep] - R[:, keep].mean(axis=0)
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
    j = int(np.argmin(np.abs(f - expect)))
    pk = int(np.argmax(prom[band]))
    return dict(at=prom[j], best=prom[band][pk], at_hz=f[band][pk])


def main():
    mac = sys.argv[1]
    expect = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    runs = {}
    for name in ("paced", "control"):
        per = load(os.path.join(D, name + ".csv"))
        if mac not in per:
            print(f"{name}: transmitter {mac} not present")
            return
        runs[name] = features(per[mac])

    print(f"Phase-based A/B, expected {expect:.3f} Hz ({expect*60:.0f} brpm)\n")
    print(f"{'feature':<14}{'paced@exp':>11}{'ctrl@exp':>11}{'ratio':>8}"
          f"{'paced_best':>12}{'at Hz':>8}")
    verdicts = []
    for fname in ("amplitude", "phase_diff", "conj_real", "conj_imag", "amp_ratio"):
        a = prominence_at(runs["paced"][0], runs["paced"][1][fname], expect)
        b = prominence_at(runs["control"][0], runs["control"][1][fname], expect)
        if a is None or b is None:
            continue
        ratio = a["at"] / b["at"] if b["at"] else float("inf")
        verdicts.append((fname, a["at"], ratio))
        print(f"{fname:<14}{a['at']:>11.2f}{b['at']:>11.2f}{ratio:>8.2f}"
              f"{a['best']:>12.2f}{a['at_hz']:>8.3f}")

    print()
    win = [v for v in verdicts if v[1] > 4 and v[2] > 2]
    if win:
        print("SIGNAL FOUND in:", ", ".join(v[0] for v in win))
    else:
        print("No feature shows breathing at the predicted rate above control.")


if __name__ == "__main__":
    main()
