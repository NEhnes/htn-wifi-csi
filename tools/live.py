"""LIVE presence monitor v2 - no training, no prompts. Walk in and out.

Improvements over v1, in order of how much they matter:

1. PER-SUBCARRIER NORMALISATION (Mahalanobis-ish instead of Euclidean).
   Subcarriers have very different noise levels; plain Euclidean distance let
   the noisiest ones dominate. Now each is divided by its OWN calibration
   standard deviation, so a quiet subcarrier moving 2 units counts more than a
   noisy one moving 2 units.
2. HYSTERESIS. Separate enter/exit thresholds stop the flicker at the boundary.
3. TEMPORAL SMOOTHING (EMA) so single bad packets cannot flip the state.
4. ADAPTIVE BASELINE. While confidently ABSENT the baseline creeps toward the
   current channel, cancelling slow drift - the main false-positive source.
5. SHAPE vs POWER split. Reports both how much the fading SHAPE changed and how
   much total power changed; a body does both, AGC wander mostly does the latter.

Usage:
  python live.py [--port COM4] [--calib 20] [--hi 4] [--lo 2.5] [--log out.csv]
Ctrl+C to stop.
"""
import argparse
import time
from collections import deque

import numpy as np
import serial


def parse(s):
    p = s.split(",")
    if len(p) < 7 or p[0] not in ("CSI", "CSIX"):
        return None
    try:
        t = int(p[1]); rssi = int(p[3]); n = int(p[5])
    except ValueError:
        return None
    if p[0] == "CSIX":
        h = p[6].strip()
        if len(h) < 2 * n:
            return None
        try:
            arr = np.frombuffer(bytes.fromhex(h[:2 * n]), dtype=np.int8)
        except ValueError:
            return None
    else:
        vals = [v for v in p[6:] if v != ""]
        if len(vals) < n:
            return None
        try:
            arr = np.array([int(v) for v in vals[:n]], dtype=np.int8)
        except ValueError:
            return None
    im = arr[0::2].astype(np.float64)
    re = arr[1::2].astype(np.float64)
    return t / 1e6, rssi, np.sqrt(re ** 2 + im ** 2)


def bar(v, hi, width=30):
    n = int(np.clip(v / max(hi, 1e-9), 0, 1) * width)
    return "#" * n + "." * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--calib", type=float, default=20.0)
    ap.add_argument("--hi", type=float, default=4.0, help="enter PRESENT above")
    ap.add_argument("--lo", type=float, default=2.5, help="back to ABSENT below")
    ap.add_argument("--window", type=float, default=0.8)
    ap.add_argument("--log", default=None, help="CSV of time,score,rssi")
    a = ap.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, 115200, 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.dtr = False
    ser.rts = False
    time.sleep(0.3)
    ser.reset_input_buffer()

    print("=" * 72)
    print(f"  CALIBRATING {a.calib:.0f}s - STAY OUT OF THE PATH")
    print("=" * 72)
    for k in range(5, 0, -1):
        print(f"  starting in {k}...", end="\r", flush=True)
        time.sleep(1)

    chunk, profiles = [], []
    t0 = time.time()
    last = t0
    while time.time() - t0 < a.calib:
        raw = ser.readline()
        if not raw:
            continue
        r = parse(raw.decode("utf-8", "ignore").strip())
        if r is None:
            continue
        chunk.append(r[2])
        now = time.time()
        if now - last >= 0.4 and chunk:
            profiles.append(np.mean(chunk, axis=0))
            chunk = []
            last = now
        print(f"  calibrating {now-t0:4.1f}/{a.calib:.0f}s  "
              f"chunks={len(profiles):3d}", end="\r", flush=True)

    if len(profiles) < 10:
        print("\n  NOT ENOUGH DATA - transmitter powered? in range?")
        ser.close(); return

    P = np.stack(profiles)
    baseline = P.mean(axis=0)
    base_sd = np.maximum(P.std(axis=0), 1e-3)   # per-subcarrier noise
    base_pow = baseline.sum()

    # Calibration score distribution (should sit near 1 by construction).
    zc = (P - baseline) / base_sd
    sc = np.sqrt((zc ** 2).mean(axis=1))
    mu_s, sd_s = float(sc.mean()), max(float(sc.std()), 1e-6)

    print(f"\n  baseline from {len(profiles)} chunks | "
          f"calib score {mu_s:.2f} +/- {sd_s:.2f}\n")
    print("=" * 72)
    print(f"  LIVE - enter>{a.hi:.1f}  exit<{a.lo:.1f}. Walk in and out. Ctrl+C to stop.")
    print("=" * 72)

    logf = open(a.log, "w", encoding="utf-8") if a.log else None
    if logf:
        logf.write("t,score,shape,power_db,rssi,state\n")

    win = deque()
    ema = 0.0
    present = False
    absent_since = time.time()
    last_draw = 0.0
    tstart = time.time()
    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            r = parse(raw.decode("utf-8", "ignore").strip())
            if r is None:
                continue
            win.append((time.time(), r[1], r[2]))
            cut = time.time() - a.window
            while win and win[0][0] < cut:
                win.popleft()
            if len(win) < 5:
                continue
            now = time.time()
            if now - last_draw < 0.12:
                continue
            last_draw = now

            prof = np.mean([x[2] for x in win], axis=0)
            rssi = float(np.mean([x[1] for x in win]))

            z = (prof - baseline) / base_sd
            raw_score = float(np.sqrt((z ** 2).mean()))
            score = (raw_score - mu_s) / sd_s          # in calibration sigmas
            ema = 0.45 * score + 0.55 * ema

            # split: shape change vs overall power change
            power_db = 10 * np.log10(max(prof.sum(), 1e-9) / max(base_pow, 1e-9))
            shape = float(np.sqrt(((z - z.mean()) ** 2).mean()))

            if present and ema < a.lo:
                present = False
                absent_since = now
            elif (not present) and ema > a.hi:
                present = True

            # Adaptive baseline: only while confidently empty for a while.
            if (not present) and ema < a.lo * 0.6 and now - absent_since > 4.0:
                baseline = 0.995 * baseline + 0.005 * prof
                base_pow = baseline.sum()

            state = "PRESENT" if present else " ABSENT"
            mark = "##" if present else "  "
            print(f"  {mark}[{state}]{mark} {ema:6.1f}s "
                  f"|{bar(ema, max(a.hi * 3, 12))}| "
                  f"shape={shape:5.1f} pow={power_db:+5.2f}dB rssi={rssi:6.1f}",
                  end="\r", flush=True)
            if logf:
                logf.write(f"{now-tstart:.3f},{ema:.3f},{shape:.3f},"
                           f"{power_db:.3f},{rssi:.1f},{int(present)}\n")
                logf.flush()
    except KeyboardInterrupt:
        print("\n\n  stopped.")
    finally:
        if logf:
            logf.close()
            print(f"  log -> {a.log}")
        ser.close()


if __name__ == "__main__":
    main()
