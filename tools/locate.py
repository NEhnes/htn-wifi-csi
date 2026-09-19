"""Live zone localisation from CSI fingerprints, with temporal smoothing.

This is FINGERPRINTING, not geometric localisation. We do not solve for
position from signal strength - multipath makes that hopeless. Instead every
location scatters the signal into a distinctive 64-subcarrier pattern, and we
learn what each zone looks like. Multipath stops being the enemy and becomes
the fingerprint.

Raw per-window predictions flicker, because two adjacent zones look similar and
a single noisy window can flip the answer. A person cannot teleport, so we run
a forward (Bayes) filter over a transition model: the posterior for a zone
depends on the observation AND on where you plausibly were a moment ago. This
is a one-line change that makes the demo look dramatically better, and it is
honest estimation rather than cosmetics.

Usage:
  python locate.py --session zones1 --ports COM4,COM6 [--grid 2x2] [--stay 0.85]
"""
import argparse
import os
import pickle
import sys
import threading
import time
from collections import deque

import numpy as np
import serial

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from csilib import parse_line, link_features  # noqa: E402


def build_transition(classes, grid, stay):
    """Transition matrix. With a grid we only allow moves to neighbours."""
    n = len(classes)
    A = np.full((n, n), (1.0 - stay) / max(n - 1, 1))
    np.fill_diagonal(A, stay)
    if not grid:
        return A

    rows, cols = grid
    pos = {}
    for i, c in enumerate(classes):
        pos[i] = (i // cols, i % cols)
    A = np.zeros((n, n))
    for i in range(n):
        ri, ci = pos[i]
        nbrs = [j for j in range(n)
                if abs(pos[j][0] - ri) + abs(pos[j][1] - ci) == 1]
        A[i, i] = stay
        if nbrs:
            for j in nbrs:
                A[i, j] = (1.0 - stay) / len(nbrs)
        else:
            A[i, i] = 1.0
    return A


def reader(ser, sink, stop):
    while not stop.is_set():
        raw = ser.readline()
        if not raw:
            continue
        s = raw.decode("utf-8", "ignore").strip()
        if s.startswith(("CSI,", "CSIX,")):
            r = parse_line(s)
            if r is not None:
                sink.append((time.time(), r[1], r[2]))


def bar(p, w=18):
    return "#" * int(round(p * w)) + "." * (w - int(round(p * w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="zones1")
    ap.add_argument("--ports", default=None)
    ap.add_argument("--grid", default=None, help="e.g. 2x2 for neighbour-only moves")
    ap.add_argument("--stay", type=float, default=0.85,
                    help="probability of staying put between updates")
    ap.add_argument("--window", type=float, default=None)
    a = ap.parse_args()

    mp = os.path.join(D, "data", a.session, "model.pkl")
    if not os.path.exists(mp):
        raise SystemExit(f"no model at {mp} - run collect.py then train.py first")
    with open(mp, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    classes = list(bundle["classes"])
    ports = [p.strip() for p in a.ports.split(",")] if a.ports else bundle["ports"]
    wsec = a.window or bundle["window"]

    grid = None
    if a.grid:
        r, c = a.grid.lower().split("x")
        grid = (int(r), int(c))
        if grid[0] * grid[1] != len(classes):
            print(f"warning: grid {a.grid} does not match {len(classes)} zones;"
                  f" ignoring grid")
            grid = None

    A = build_transition(classes, grid, a.stay)
    prior = np.full(len(classes), 1.0 / len(classes))

    sers, sinks, ths = [], [], []
    stop = threading.Event()
    for p in ports:
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = p, 115200, 0.2
        s.dtr = False
        s.rts = False
        s.open()
        s.dtr = False
        s.rts = False
        sers.append(s)
        sink = deque(maxlen=4000)
        sinks.append(sink)
    time.sleep(0.4)
    for s, sink in zip(sers, sinks):
        s.reset_input_buffer()
        th = threading.Thread(target=reader, args=(s, sink, stop), daemon=True)
        th.start()
        ths.append(th)

    print("=" * 70)
    print(f"  LIVE ZONE TRACKING  model={bundle.get('model','?')} "
          f"zones={len(classes)}  links={len(ports)}")
    print(f"  smoothing: stay={a.stay}"
          + (f", grid {grid[0]}x{grid[1]} (neighbour moves only)" if grid else ""))
    print("  Ctrl+C to stop")
    print("=" * 70)

    try:
        while True:
            time.sleep(wsec * 0.5)
            now = time.time()
            vec, ok = [], True
            for sink in sinks:
                rows = [r for r in list(sink) if r[0] >= now - wsec]
                f = link_features(rows, wsec) if rows else None
                if f is None:
                    ok = False
                    break
                vec.append(f)
            if not ok:
                print("  waiting for frames on all links...        ",
                      end="\r", flush=True)
                continue

            x = np.concatenate(vec).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                like = model.predict_proba(x)[0]
                like = np.array([like[list(model.classes_).index(c)]
                                 for c in classes])
            else:
                pred = model.predict(x)[0]
                like = np.array([1.0 if c == pred else 0.02 for c in classes])

            # Bayes forward filter: predict through the transition model,
            # then correct with the new observation.
            prior = prior @ A
            post = prior * np.maximum(like, 1e-9)
            post /= post.sum()
            prior = post

            k = int(np.argmax(post))
            line = "  ".join(f"{c}:{post[i]*100:3.0f}%"
                             for i, c in enumerate(classes))
            print(f"  >> {classes[k]:<8s} [{bar(post[k])}] {post[k]*100:5.1f}%   "
                  f"{line}      ", end="\r", flush=True)
    except KeyboardInterrupt:
        print("\n\n  stopped.")
    finally:
        stop.set()
        time.sleep(0.3)
        for s in sers:
            s.close()


if __name__ == "__main__":
    main()
