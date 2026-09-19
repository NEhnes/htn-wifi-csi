"""Alternating presence test - defeats the time confound.

Records IN / OUT / IN / OUT / ... blocks so that slow drift cannot line up with
the labels. Then validates BLOCK-WISE: train on some blocks, test on a held-out
IN/OUT pair the model has never seen. If accuracy stays high across held-out
blocks, the signal is presence, not drift.

Usage: python alternate.py [--block 25] [--cycles 3] [--port COM4]
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import serial

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from analyze import load  # noqa: E402
from presence import windows  # noqa: E402

TX_MAC = "024353490001"


def reader(ser, sink, stop):
    while not stop.is_set():
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("utf-8", "ignore").strip()
        if s.startswith(("CSI,", "CSIX,")):
            sink.append(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=float, default=25.0)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--port", default="COM4")
    ap.add_argument("--move", type=float, default=8.0, help="seconds to move")
    a = ap.parse_args()

    ser = serial.Serial()
    ser.port, ser.baudrate, ser.timeout = a.port, 115200, 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.dtr = False
    ser.rts = False
    time.sleep(0.3)

    total = a.cycles * 2 * (a.block + a.move) / 60.0
    print("=" * 62)
    print(f"  ALTERNATING PRESENCE TEST  ~{total:.1f} minutes")
    print(f"  {a.cycles} cycles of: {a.block:.0f}s IN the path, {a.block:.0f}s OUT")
    print("  Follow the prompts. Sit still during IN blocks.")
    print("=" * 62)

    blocks = []
    for cyc in range(a.cycles):
        for label in ("IN", "OUT"):
            word = ("SIT BETWEEN THE BOARDS" if label == "IN"
                    else "MOVE AWAY, OUT OF THE PATH")
            for k in range(int(a.move), 0, -1):
                print(f"  cycle {cyc+1}/{a.cycles}  >>> {word} <<<  "
                      f"starting in {k:2d}s ", end="\r", flush=True)
                time.sleep(1)
            sink, stop = [], threading.Event()
            th = threading.Thread(target=reader, args=(ser, sink, stop), daemon=True)
            ser.reset_input_buffer()
            th.start()
            t0 = time.time()
            while time.time() - t0 < a.block:
                el = time.time() - t0
                print(f"  cycle {cyc+1}/{a.cycles}  [{label:3s}] recording "
                      f"{el:4.1f}/{a.block:.0f}s   frames={len(sink):5d}      ",
                      end="\r", flush=True)
                time.sleep(0.1)
            stop.set()
            time.sleep(0.3)
            path = os.path.join(D, f"alt_{cyc}_{label}.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(sink) + "\n")
            blocks.append((cyc, label, path, len(sink)))
            print(f"  cycle {cyc+1}/{a.cycles}  [{label:3s}] done, "
                  f"{len(sink)} frames                      ")
    ser.close()
    print()
    analyse(blocks)


def analyse(blocks):
    feats = {}
    for cyc, label, path, n in blocks:
        per = load(path)
        if TX_MAC not in per or len(per[TX_MAC]) < 30:
            print(f"cycle {cyc} {label}: too few frames ({n})")
            continue
        w = windows(per[TX_MAC], 2.0)
        feats[(cyc, label)] = np.stack([x["profile"] for x in w])

    cycles = sorted({c for c, _ in feats})
    if len(cycles) < 2:
        print("need at least 2 complete cycles"); return

    print("\n--- block-wise validation (train on other cycles, test held-out) ---")
    accs = []
    for held in cycles:
        tr_in = np.concatenate([feats[(c, "IN")] for c in cycles
                                if c != held and (c, "IN") in feats])
        tr_out = np.concatenate([feats[(c, "OUT")] for c in cycles
                                 if c != held and (c, "OUT") in feats])
        cIn, cOut = tr_in.mean(axis=0), tr_out.mean(axis=0)
        teIn = feats.get((held, "IN"))
        teOut = feats.get((held, "OUT"))
        if teIn is None or teOut is None:
            continue
        okI = np.linalg.norm(teIn - cIn, axis=1) < np.linalg.norm(teIn - cOut, axis=1)
        okO = np.linalg.norm(teOut - cOut, axis=1) < np.linalg.norm(teOut - cIn, axis=1)
        acc = (okI.sum() + okO.sum()) / (len(teIn) + len(teOut))
        accs.append(acc)
        print(f"  held-out cycle {held}: {acc*100:5.1f}%  "
              f"(IN {okI.mean()*100:.0f}%, OUT {okO.mean()*100:.0f}%)")

    if accs:
        m = float(np.mean(accs))
        print(f"\nMEAN held-out accuracy = {m*100:.1f}%  (chance 50%)")
        if m > 0.85:
            print("VERDICT: presence detection is REAL and generalises across time.")
        elif m > 0.70:
            print("VERDICT: usable but noisy - would need more data//features.")
        else:
            print("VERDICT: does not generalise; earlier result was time drift.")


if __name__ == "__main__":
    main()
