"""Paced-breathing CSI test: metronome + live capture + analysis + plot.

Usage:
    python paced.py                 # 96s at 15 brpm (0.25 Hz)
    python paced.py --brpm 12       # pace at 12 breaths/min
    python paced.py --secs 120
    python paced.py --mac aabbcc112233   # lock to one transmitter
    python paced.py --control       # no metronome: control run, sit away

Why paced: we PREDICT the frequency, then test whether energy shows up there.
Far harder to fool yourself than hunting for any peak in a wide band.
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, amplitudes, resample, rolling_median_helper  # noqa: E402

PORT, BAUD = "COM4", 115200


def reader(ser, lines, stop):
    while not stop.is_set():
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("utf-8", "ignore").strip()
        # Accept BOTH formats: "CSI," (v1 decimal) and "CSIX," (v2 hex).
        # Note "CSIX,".startswith("CSI,") is False - that mismatch silently
        # discarded every frame once the firmware switched to hex.
        if s.startswith(("CSI,", "CSIX,")):
            lines.append(s)


def bar(frac, width=40):
    n = int(max(0.0, min(1.0, frac)) * width)
    return "#" * n + "-" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brpm", type=float, default=15.0)
    ap.add_argument("--secs", type=float, default=96.0)
    ap.add_argument("--mac", default=None)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--reset", action="store_true",
                    help="reboot the board first so it re-scans for the channel")
    ap.add_argument("--port", default=PORT, help="receiver serial port")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cycle = 60.0 / a.brpm
    expect_hz = a.brpm / 60.0
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "control.csv" if a.control else "paced.csv")

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, BAUD, 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.dtr = False
    ser.rts = False
    time.sleep(0.3)

    if a.reset:
        # The board chooses its channel during the boot scan. If the hotspot was
        # switched on after the last boot, we must re-scan or we sit on the wrong
        # channel and hear nothing from it.
        print("  resetting board so it re-scans (hotspot must be ON now)...")
        ser.dtr = False
        ser.rts = True
        time.sleep(0.15)
        ser.rts = False
        boot = ""
        t_end = time.time() + 8
        while time.time() < t_end:
            boot += ser.read(4096).decode("utf-8", "ignore")
            if "CSI capture running" in boot:
                break
        for ln in boot.splitlines():
            if "camping on channel" in ln or "CSI capture running" in ln:
                print("   ", ln.strip())
        aps = [l for l in boot.splitlines() if "csi: AP " in l]
        if aps:
            print(f"    ({len(aps)} APs seen; strongest listed first)")
            for ln in aps[:3]:
                print("   ", ln.strip())

    ser.reset_input_buffer()

    lines, stop = [], threading.Event()
    th = threading.Thread(target=reader, args=(ser, lines, stop), daemon=True)

    print("=" * 62)
    if a.control:
        print("  CONTROL RUN - move AWAY from the board, stay out of the path")
    else:
        print(f"  PACED BREATHING - {a.brpm:.0f} breaths/min -> expect {expect_hz:.3f} Hz")
        print(f"  Sit between phone and ESP32. Chest still. {cycle:.1f}s per cycle.")
    print("=" * 62)
    for k in range(5, 0, -1):
        print(f"  starting in {k}...", end="\r", flush=True)
        time.sleep(1)
    print(" " * 40, end="\r")

    th.start()
    t0 = time.time()
    last = 0
    while True:
        el = time.time() - t0
        if el >= a.secs:
            break
        phase = (el % cycle) / cycle
        word = "INHALE" if phase < 0.5 else "EXHALE"
        sub = (phase % 0.5) * 2
        n = len(lines)
        rate = (n - last) if el >= 1 else 0
        if a.control:
            print(f"  [{bar(el/a.secs)}] {el:5.1f}/{a.secs:.0f}s  "
                  f"frames={n:5d}", end="\r", flush=True)
        else:
            print(f"  {word}  [{bar(sub, 22)}]  {el:5.1f}/{a.secs:.0f}s  "
                  f"frames={n:5d}  ", end="\r", flush=True)
        time.sleep(0.1)
        if int(el) != int(el - 0.1):
            last = n
    stop.set()
    time.sleep(0.4)
    ser.close()
    print("\n")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"captured {len(lines)} frames -> {out}")

    analyse(out, expect_hz, a.mac, a.control)


def analyse(path, expect_hz, want_mac, is_control):
    per_mac = load(path)
    if not per_mac:
        print("no frames parsed"); return

    print("\n--- transmitters ---")
    ranked = sorted(per_mac.items(), key=lambda kv: -len(kv[1]))
    for mac, fr in ranked[:8]:
        span = (fr[-1][0] - fr[0][0]) / 1e6 if len(fr) > 1 else 0
        rssi = np.median([x[1] for x in fr])
        star = " <-- selected" if (want_mac and mac == want_mac) else ""
        print(f"  {mac}  n={len(fr):5d}  {len(fr)/span if span else 0:5.1f} Hz  "
              f"rssi={rssi:6.1f}{star}")

    mac, frames = (want_mac, per_mac[want_mac]) if want_mac in per_mac else ranked[0]
    if len(frames) < 60:
        print("too few frames"); return

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
    out_b = (~band) & (f < f[-1] * 0.9)
    j = int(np.argmin(np.abs(f - expect_hz)))
    pk = int(np.argmax(prom[band]))

    print(f"\n--- {mac}: {len(t)} frames, {t[-1]:.0f}s, {fs:.1f} Hz, "
          f"res {freqs[1]:.4f} Hz ---")
    print(f"live subcarriers {alive.size}/64")
    print(f"\nPROMINENCE AT EXPECTED {expect_hz:.3f} Hz "
          f"({expect_hz*60:.0f} brpm) = {prom[j]:.1f}x")
    print(f"best in band  : {prom[band][pk]:.1f}x at {f[band][pk]:.3f} Hz "
          f"({f[band][pk]*60:.1f} brpm)")
    print(f"best out band : {prom[out_b].max():.1f}x at "
          f"{f[out_b][np.argmax(prom[out_b])]:.3f} Hz")

    if is_control:
        print("\nCONTROL: any in-band prominence here is the false-positive floor.")
    else:
        hit = abs(f[band][pk] - expect_hz) <= 3 * freqs[1]
        if prom[j] > 6 and hit:
            print("\nVERDICT: energy at the PREDICTED rate. Real signal.")
        elif prom[j] > 4:
            print("\nVERDICT: elevated at predicted rate; compare against --control.")
        else:
            print("\nVERDICT: nothing at the predicted rate.")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(11, 10))
        ax[0].imshow(X.T, aspect="auto", origin="lower",
                     extent=[0, grid[-1], 0, X.shape[1]], cmap="viridis")
        ax[0].set_title(f"CSI amplitude waterfall - {mac}")
        ax[0].set_xlabel("s"); ax[0].set_ylabel("subcarrier")

        ax[1].plot(grid, Xd[:, 0], lw=0.9)
        if not is_control:
            for c in np.arange(0, grid[-1], 1.0 / expect_hz):
                ax[1].axvline(c, color="r", alpha=0.25, lw=0.8)
        ax[1].set_title("strongest subcarrier (detrended); red = expected breath cycles")
        ax[1].set_xlabel("s")

        ax[2].semilogy(f, prom, lw=1.0)
        ax[2].axvspan(0.15, 0.60, color="g", alpha=0.12, label="breathing band")
        ax[2].axvline(expect_hz, color="r", ls="--", label=f"expected {expect_hz:.3f} Hz")
        ax[2].axhline(1.0, color="k", lw=0.6)
        ax[2].set_xlim(0, min(2.0, f[-1]))
        ax[2].set_xlabel("Hz"); ax[2].set_ylabel("prominence over local bg")
        ax[2].legend()
        fig.tight_layout()
        png = path.replace(".csv", ".png")
        fig.savefig(png, dpi=130)
        print(f"\nplot -> {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
