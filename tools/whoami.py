"""Identify which COM port is the TRANSMITTER and which is the RECEIVER,
by listening to what each board actually says.

Usage: python whoami.py
Tip: to find out which PHYSICAL board is which, unplug one and run this again.
     The port that disappears belongs to the board in your hand.
"""
import time

import serial
import serial.tools.list_ports


def classify(port, secs=6.0):
    try:
        ser = serial.Serial()
        ser.port, ser.baudrate, ser.timeout = port, 115200, 0.2
        ser.dtr = False
        ser.rts = False
        ser.open()
        ser.dtr = False
        ser.rts = False
        time.sleep(0.2)
        ser.reset_input_buffer()
        buf = ""
        t0 = time.time()
        while time.time() - t0 < secs:
            buf += ser.read(2048).decode("utf-8", "ignore")
            if "csitx" in buf or "CSIX," in buf:
                break
        ser.close()
    except Exception as e:
        return "ERROR", str(e)

    if "csitx" in buf or "rate pinned" in buf:
        return "TRANSMITTER", "power it from a charger, place ~1.5 m away"
    if "CSIX," in buf or "STATUS heard" in buf:
        return "RECEIVER", "keep this one plugged into the laptop"
    if buf.strip():
        return "UNKNOWN", buf.strip().splitlines()[-1][:60]
    return "SILENT", "no output (unpowered, or needs a reset)"


ports = sorted(p.device for p in serial.tools.list_ports.comports())
if not ports:
    print("No COM ports found - is anything plugged in?")
else:
    print(f"Found {len(ports)} port(s). Listening to each...\n")
    for p in ports:
        role, note = classify(p)
        print(f"  {p:<6} = {role:<12}  {note}")
    print("\nTo find which PHYSICAL board is which: unplug one, run this again,")
    print("and whichever port vanished is the board you are holding.")
