"""Experiment suite for the through-wall WiFi imaging feasibility study."""
import time
import numpy as np
from wifi_sar_sim import sweep

CH1 = (6,)
CH8 = (1, 3, 5, 7, 9, 11, 13, 2)
N = 12

t0 = time.time()
print("=" * 100)
print("THROUGH-WALL WiFi IMAGING FEASIBILITY  -  ESP32-class RSSI hardware")
print("Room behind a drywall partition. Person ~0.4 m wide. Receiver sweeps a 3 m line.")
print(f"'within 0.5 m' = located well enough to be useful. {N} trials each,")
print("randomised person position and multipath geometry per trial.")
print("=" * 100)

print()
print("--- A. How many transmitters? (idealised physics: no multipath, no noise) ---")
for ntx in (1, 2, 4, 8):
    sweep(f"A  {ntx} TX", n_trials=N, mode="differential", multipath=False,
          rssi_sigma=1e-6, pos_jitter=0.0, channels=CH1, n_tx=ntx)

print()
print("--- B. Adding reality one layer at a time (differential mode, 4 TX) ---")
sweep("B1 idealised physics", n_trials=N, mode="differential", multipath=False,
      rssi_sigma=1e-6, pos_jitter=0.0, channels=CH1, n_tx=4)
sweep("B2 + RSSI noise 1.2 dB + 1 dB quantisation", n_trials=N, mode="differential",
      multipath=False, channels=CH1, n_tx=4)
sweep("B3 + realistic multipath, single channel", n_trials=N, mode="differential",
      multipath=True, channels=CH1, n_tx=4)
sweep("B4 + multipath, 8 channels averaged", n_trials=N, mode="differential",
      multipath=True, channels=CH8, n_tx=4)
sweep("B5 + multipath, 8 channels, 8 TX", n_trials=N, mode="differential",
      multipath=True, channels=CH8, n_tx=8)

print()
print("--- C. Absolute mode: image an unknown STATIC scene, no baseline (8 TX) ---")
sweep("C1 idealised physics", n_trials=N, mode="absolute", multipath=False,
      rssi_sigma=1e-6, pos_jitter=0.0, channels=CH1, n_tx=8)
sweep("C2 realistic multipath, 8 channels", n_trials=N, mode="absolute",
      multipath=True, channels=CH8, n_tx=8)

print()
print("--- D. Aperture length (differential, multipath, 8 ch, 4 TX) ---")
for ap in (0.5, 1.0, 2.0, 3.0):
    sweep(f"D aperture {ap:.1f} m", n_trials=N, mode="differential", multipath=True,
          aperture=ap, channels=CH8, n_tx=4)

print()
print("--- E. How well must you know the receiver's position? ---")
for j in (0.0, 0.01, 0.03, 0.10):
    sweep(f"E position error {j*100:4.1f} cm", n_trials=N, mode="differential",
          multipath=True, pos_jitter=j, channels=CH8, n_tx=4)

print()
print("--- F. Packets averaged per position ---")
for npk in (4, 32, 200):
    sweep(f"F {npk:3d} packets/position", n_trials=N, mode="differential", multipath=True,
          n_packets=npk, channels=CH8, n_tx=4)

print()
print(f"total runtime {time.time() - t0:.0f} s")
