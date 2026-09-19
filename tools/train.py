"""Train and HONESTLY validate a classifier on collected CSI.

The validation is leave-one-rep-out. Windows from the same recording are nearly
identical, so a random split would report ~99% and mean nothing. Holding out a
whole rep asks the real question: does this work on a recording it has never
seen?

Usage: python train.py --session session1 [--window 1.0]
"""
import argparse
import glob
import os
import pickle
from collections import defaultdict

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
sys.path.insert(0, D)
from csilib import parse_line, link_features, windowize, feature_names  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402


def load_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            i = line.find(",")
            if i < 0:
                continue
            try:
                tp = float(line[:i])
            except ValueError:
                continue
            r = parse_line(line[i + 1:].strip())
            if r is not None:
                rows.append((tp, r[1], r[2]))   # PC time, rssi, amp
    return rows


def build(session, wsec):
    files = glob.glob(os.path.join(D, "data", session, "*.csv"))
    if not files:
        raise SystemExit(f"no data in data/{session}")

    # group by (class, rep) -> {port: rows}
    groups = defaultdict(dict)
    ports = set()
    for fp in files:
        base = os.path.basename(fp)[:-4]
        cls, rep, port = base.split("__")
        ports.add(port)
        groups[(cls, int(rep[3:]))][port] = load_file(fp)
    ports = sorted(ports)

    X, y, g = [], [], []
    for (cls, rep), byport in sorted(groups.items()):
        per_port_windows = {}
        for p in ports:
            rows = byport.get(p, [])
            per_port_windows[p] = dict(windowize(rows, wsec)) if rows else {}
        keys = set.intersection(*[set(v.keys()) for v in per_port_windows.values()]) \
            if per_port_windows else set()
        for k in sorted(keys):
            vec = []
            ok = True
            for p in ports:
                f = link_features(per_port_windows[p][k], wsec)
                if f is None:
                    ok = False
                    break
                vec.append(f)
            if ok:
                X.append(np.concatenate(vec))
                y.append(cls)
                g.append(rep)
    return np.array(X), np.array(y), np.array(g), ports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session1")
    ap.add_argument("--window", type=float, default=1.0)
    a = ap.parse_args()

    X, y, g, ports = build(a.session, a.window)
    classes = sorted(set(y))
    print(f"receivers : {', '.join(ports)}")
    print(f"windows   : {len(X)}  features: {X.shape[1]}  "
          f"({len(ports)} link(s) x {X.shape[1]//len(ports)})")
    for c in classes:
        print(f"   {c:8s} {int((y==c).sum()):4d} windows")
    reps = sorted(set(g))
    if len(reps) < 2:
        raise SystemExit("need >=2 reps for honest validation")

    models = {
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0)),
        "forest": RandomForestClassifier(n_estimators=300, random_state=0,
                                         min_samples_leaf=2),
    }

    print("\n--- leave-one-rep-out validation ---")
    best, best_acc = None, -1
    for name, mk in models.items():
        accs, preds, trues = [], [], []
        for held in reps:
            tr, te = g != held, g == held
            if te.sum() == 0 or len(set(y[tr])) < 2:
                continue
            from sklearn.base import clone
            m = clone(mk)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            accs.append((p == y[te]).mean())
            preds.extend(p); trues.extend(y[te])
        acc = float(np.mean(accs))
        print(f"  {name:8s} mean acc = {acc*100:5.1f}%   "
              f"per-rep: {', '.join(f'{x*100:.0f}%' for x in accs)}")
        if acc > best_acc:
            best, best_acc, bp, bt = name, acc, preds, trues

    print(f"\nbest: {best} at {best_acc*100:.1f}%  (chance = "
          f"{100/len(classes):.0f}%)")
    cm = confusion_matrix(bt, bp, labels=classes)
    print("\nconfusion (rows=true, cols=pred):")
    print("           " + "".join(f"{c:>9s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:8s} " + "".join(f"{v:9d}" for v in cm[i]))

    # Retrain on everything and save for live inference.
    from sklearn.base import clone
    final = clone(models[best])
    final.fit(X, y)
    out = os.path.join(D, "data", a.session, "model.pkl")
    with open(out, "wb") as f:
        pickle.dump(dict(model=final, ports=ports, window=a.window,
                         classes=classes), f)
    print(f"\nmodel -> {out}")

    if best == "forest":
        names = []
        for p in ports:
            names += [f"{p}:{n}" for n in feature_names()]
        imp = final.feature_importances_
        top = np.argsort(-imp)[:10]
        print("\ntop features:")
        for i in top:
            print(f"   {names[i]:24s} {imp[i]:.3f}")


if __name__ == "__main__":
    main()
