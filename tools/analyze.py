"""Analyse captured ESP32 CSI: per-transmitter time series -> breathing band.

Usage: python analyze.py <capfile> [label]

ESP32 LLTF CSI layout: 64 subcarriers, each stored as an (imag, real) int8 pair.
Amplitude = sqrt(re^2 + im^2).

Frames from different source MACs traverse different paths, so they are NOT
comparable. We group by MAC and analyse the busiest transmitter only.
"""
import sys
from collections import defaultdict

import numpy as np

BREATH_LO, BREATH_HI = 0.15, 0.60   # Hz  (9-36 breaths/min)


def rolling_median_helper(y, w):
    """Rolling median, used as a LOCAL spectral background.

    CSI noise is strongly 1/f, so comparing an in-band peak to the
    high-frequency floor makes slow drift masquerade as signal.
    """
    half = w // 2
    pad = np.pad(y, half, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(y.size)])


def load(path):
    per_mac = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7 or p[0] not in ("CSI", "CSIX"):
                continue
            try:
                t_us = int(p[1]); mac = p[2]; rssi = int(p[3]); n = int(p[5])
            except ValueError:
                continue

            if p[0] == "CSIX":
                # v2 receiver: hex-encoded I/Q (compact enough for 30 Hz @115200)
                h = p[6].strip()
                if len(h) < 2 * n:
                    continue
                try:
                    raw = bytes.fromhex(h[:2 * n])
                except ValueError:
                    continue
                arr = np.frombuffer(raw, dtype=np.int8)
            else:
                # v1 receiver: decimal CSV
                vals = [v for v in p[6:] if v != ""]
                if len(vals) < n:
                    continue
                try:
                    arr = np.array([int(v) for v in vals[:n]], dtype=np.int8)
                except ValueError:
                    continue
            per_mac[mac].append((t_us, rssi, arr))
    return per_mac


def amplitudes(frames):
    """(T, n_sub) amplitude matrix + time vector in seconds."""
    t = np.array([f[0] for f in frames], dtype=np.float64) / 1e6
    raw = np.stack([f[2] for f in frames])          # (T, 128)
    im = raw[:, 0::2].astype(np.float64)
    re = raw[:, 1::2].astype(np.float64)
    amp = np.sqrt(re ** 2 + im ** 2)                # (T, 64)
    return t - t[0], amp


def resample(t, x, fs):
    grid = np.arange(0, t[-1], 1.0 / fs)
    out = np.empty((grid.size, x.shape[1]))
    for k in range(x.shape[1]):
        out[:, k] = np.interp(grid, t, x[:, k])
    return grid, out


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path

    per_mac = load(path)
    if not per_mac:
        print("no CSI frames parsed"); return

    print(f"=== {label} ===")
    print(f"transmitters seen: {len(per_mac)}")
    top = sorted(per_mac.items(), key=lambda kv: -len(kv[1]))[:5]
    for mac, fr in top:
        span = (fr[-1][0] - fr[0][0]) / 1e6 if len(fr) > 1 else 0
        rate = len(fr) / span if span > 0 else 0
        print(f"  {mac}  frames={len(fr):5d}  span={span:6.1f}s  rate={rate:5.1f} Hz")

    mac, frames = top[0]
    if len(frames) < 40:
        print("\ntoo few frames from the busiest transmitter to analyse"); return

    t, amp = amplitudes(frames)
    span = t[-1]
    rate = len(t) / span
    print(f"\nanalysing {mac}: {len(t)} frames over {span:.1f}s ({rate:.1f} Hz)")

    # Uniform grid: needed because frames arrive irregularly.
    fs = max(2.0, min(20.0, rate))
    grid, X = resample(t, amp, fs)
    if grid.size < 32:
        print("capture too short"); return

    # Drop dead/constant subcarriers, keep the most variable ones.
    sd = X.std(axis=0)
    alive = np.where(sd > 1e-6)[0]
    if alive.size == 0:
        print("all subcarriers flat - no signal"); return
    order = alive[np.argsort(-sd[alive])]
    keep = order[: min(12, order.size)]
    print(f"live subcarriers: {alive.size}/64, using top {keep.size} by variance")

    Xd = X[:, keep] - X[:, keep].mean(axis=0)
    # Detrend (remove slow drift that would smear into the breathing band)
    ramp = np.arange(grid.size)
    for k in range(Xd.shape[1]):
        c = np.polyfit(ramp, Xd[:, k], 1)
        Xd[:, k] -= np.polyval(c, ramp)

    win = np.hanning(grid.size)[:, None]
    spec = np.abs(np.fft.rfft(Xd * win, axis=0)) ** 2
    freqs = np.fft.rfftfreq(grid.size, 1.0 / fs)
    psd = spec.mean(axis=1)

    # CSI noise is strongly 1/f: comparing an in-band peak to the HIGH-frequency
    # floor makes any slow drift look like signal. Instead measure each bin
    # against a LOCAL background (rolling median of the spectrum), so only a
    # genuinely narrow peak standing above its own neighbourhood counts.
    def rolling_median(y, w):
        half = w // 2
        pad = np.pad(y, half, mode="edge")
        return np.array([np.median(pad[i:i + w]) for i in range(y.size)])

    valid = freqs > 0.05
    f = freqs[valid]
    p = psd[valid]
    w = max(5, int(0.30 / (freqs[1] - freqs[0])) | 1)   # ~0.3 Hz window, odd
    bg = rolling_median(p, w)
    prom = p / np.maximum(bg, 1e-30)                    # prominence over local bg

    band = (f >= BREATH_LO) & (f <= BREATH_HI)
    if not band.any():
        print("frequency grid too coarse - capture longer"); return

    pk_i = np.argmax(prom[band])
    pk_f = f[band][pk_i]
    pk_prom = prom[band][pk_i]

    print(f"\nfreq resolution: {freqs[1]:.3f} Hz   Nyquist: {freqs[-1]:.2f} Hz")
    print(f"strongest NARROW peak in {BREATH_LO}-{BREATH_HI} Hz: "
          f"{pk_f:.3f} Hz = {pk_f*60:.1f} brpm")
    print(f"prominence over local background = {pk_prom:.1f}x")

    # How special is the breathing band compared with everywhere else? If the
    # band is not more peaky than the rest of the spectrum, we have nothing.
    out = (~band) & (f < freqs[-1] * 0.9)
    if out.any():
        print(f"best prominence OUTSIDE the band = {prom[out].max():.1f}x "
              f"(at {f[out][np.argmax(prom[out])]:.3f} Hz)")

    if pk_prom > 6 and pk_prom > 1.5 * (prom[out].max() if out.any() else 0):
        print("VERDICT: narrow peak clearly above local background")
    elif pk_prom > 3:
        print("VERDICT: suggestive only - REQUIRES a paced-breathing A/B test")
    else:
        print("VERDICT: no narrow periodic component; consistent with 1/f drift")

    if len(sys.argv) > 3:
        want = float(sys.argv[3])                      # expected Hz
        j = int(np.argmin(np.abs(f - want)))
        print(f"\nAT EXPECTED RATE {want:.3f} Hz ({want*60:.0f} brpm): "
              f"prominence = {prom[j]:.1f}x")
        print(f"  (peak found {abs(pk_f-want):.3f} Hz away from expected)")

    print("\ntop 5 narrow peaks in band:")
    for i in np.argsort(-prom[band])[:5]:
        print(f"  {f[band][i]:.3f} Hz ({f[band][i]*60:5.1f} brpm)  "
              f"prominence={prom[band][i]:6.1f}x")


if __name__ == "__main__":
    main()
