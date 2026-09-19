"""Live link check: is the receiver hearing the transmitter, and how fast?

Run this after moving the boards, BEFORE committing to a 96s capture.
Usage: python status.py [seconds] [--port COMx]
"""
import sys
import time

import serial

port = "COM4"
if "--port" in sys.argv:
    port = sys.argv[sys.argv.index("--port") + 1]
secs = 15.0
for a in sys.argv[1:]:
    if a.replace(".", "").isdigit():
        secs = float(a)
        break

ser = serial.Serial()
ser.port, ser.baudrate, ser.timeout = port, 115200, 0.2
ser.dtr = False
ser.rts = False
ser.open()
ser.dtr = False
ser.rts = False
time.sleep(0.3)
ser.reset_input_buffer()

n = 0
rssi = []
t0 = time.time()
print(f"listening on {port} for {secs:.0f}s ...\n")
while time.time() - t0 < secs:
    raw = ser.readline()
    if not raw:
        continue
    s = raw.decode("utf-8", "ignore").strip()
    if s.startswith("CSIX,"):
        n += 1
        try:
            rssi.append(int(s.split(",")[3]))
        except (IndexError, ValueError):
            pass
        el = time.time() - t0
        print(f"  frames={n:5d}  rate={n/el:5.1f} Hz  "
              f"rssi={rssi[-1] if rssi else 0:4d} dBm   ", end="\r", flush=True)
    elif "STATUS" in s or "no packets" in s:
        print(f"\n  {s.strip()}")
ser.close()

el = time.time() - t0
print("\n" + "=" * 56)
if n == 0:
    print("  NO FRAMES from the transmitter.")
    print("  -> is the TX board powered? on channel 6? within range?")
else:
    avg = sum(rssi) / len(rssi) if rssi else 0
    print(f"  {n} frames in {el:.0f}s = {n/el:.1f} Hz, mean RSSI {avg:.0f} dBm")
    if n / el < 15:
        print("  WARNING: rate is low - move the boards closer or check power.")
    else:
        print("  GOOD: link is healthy, ready to capture.")
