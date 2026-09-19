"""Labelled data collection from one or more receivers.

Records several REPS of each class, interleaved, so that a classifier can later
be validated on a whole held-out rep. That matters: validating on random windows
from the same recording massively overstates accuracy, because windows seconds
apart are nearly identical.

Usage:
  python collect.py --ports COM4,COM6 --classes empty,still,moving --secs 25 --reps 3
"""
import argparse
import os
import threading
import time

import serial

D = os.path.dirname(os.path.abspath(__file__))


def reader(ser, sink, stop):
    while not stop.is_set():
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("utf-8", "ignore").strip()
        if s.startswith(("CSI,", "CSIX,")):
            sink.append((time.time(), s))


INSTRUCT = {
    "empty":  "LEAVE THE AREA - nobody in or near the link",
    "still":  "SIT STILL between the boards",
    "moving": "WALK / MOVE AROUND between the boards",
    "near":   "STAND CLOSE TO THE RECEIVER end",
    "far":    "STAND CLOSE TO THE TRANSMITTER end",
    "mid":    "STAND AT THE MIDPOINT between the boards",
}


def instruction(cls):
    """Zone labels (z1, z2, ...) get a generic prompt.

    Mark the zones on the floor with tape BEFORE collecting, and stand on the
    same mark every rep. The model can only be as good as the labels.
    """
    return INSTRUCT.get(cls, f"STAND ON ZONE MARK '{cls.upper()}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="COM4")
    ap.add_argument("--classes", default="empty,still,moving")
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--move", type=float, default=8.0)
    ap.add_argument("--session", default="session1")
    a = ap.parse_args()

    ports = [p.strip() for p in a.ports.split(",") if p.strip()]
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    outdir = os.path.join(D, "data", a.session)
    os.makedirs(outdir, exist_ok=True)

    sers = []
    for p in ports:
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = p, 115200, 0.2
        s.dtr = False
        s.rts = False
        s.open()
        s.dtr = False
        s.rts = False
        sers.append(s)
    time.sleep(0.4)

    total = len(classes) * a.reps * (a.secs + a.move) / 60.0
    print("=" * 70)
    print(f"  COLLECTING  {len(ports)} receiver(s): {', '.join(ports)}")
    print(f"  classes: {', '.join(classes)}   reps: {a.reps}   "
          f"{a.secs:.0f}s each  (~{total:.1f} min)")
    print("=" * 70)

    for rep in range(a.reps):
        for cls in classes:
            msg = instruction(cls)
            for k in range(int(a.move), 0, -1):
                print(f"  rep {rep+1}/{a.reps}  [{cls:6s}]  >>> {msg} <<<  "
                      f"{k:2d}s ", end="\r", flush=True)
                time.sleep(1)

            sinks = [[] for _ in sers]
            stop = threading.Event()
            ths = []
            for s, sink in zip(sers, sinks):
                s.reset_input_buffer()
                th = threading.Thread(target=reader, args=(s, sink, stop),
                                      daemon=True)
                th.start()
                ths.append(th)

            t0 = time.time()
            while time.time() - t0 < a.secs:
                el = time.time() - t0
                counts = "/".join(str(len(x)) for x in sinks)
                print(f"  rep {rep+1}/{a.reps}  [{cls:6s}]  recording "
                      f"{el:4.1f}/{a.secs:.0f}s   frames={counts}        ",
                      end="\r", flush=True)
                time.sleep(0.1)
            stop.set()
            time.sleep(0.3)

            for port, sink in zip(ports, sinks):
                fn = os.path.join(outdir, f"{cls}__rep{rep}__{port}.csv")
                with open(fn, "w", encoding="utf-8") as f:
                    for tp, line in sink:
                        f.write(f"{tp:.4f},{line}\n")
            counts = "/".join(str(len(x)) for x in sinks)
            print(f"  rep {rep+1}/{a.reps}  [{cls:6s}]  saved {counts} frames"
                  + " " * 25)

    for s in sers:
        s.close()
    print(f"\n  data -> {outdir}")
    print("  next:  python train.py --session " + a.session)


if __name__ == "__main__":
    main()
