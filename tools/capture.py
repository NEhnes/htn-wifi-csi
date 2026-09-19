"""Record raw CSI lines from the ESP32 to a file.

Usage: python capture.py <seconds> <outfile>
Opens COM4 without asserting DTR/RTS so the board is not reset mid-capture.
"""
import sys
import time

import serial

PORT = "COM4"
BAUD = 115200


def main():
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    out = sys.argv[2] if len(sys.argv) > 2 else "cap.csv"

    ser = serial.Serial()
    ser.port = PORT
    ser.baudrate = BAUD
    ser.timeout = 0.2
    # Critical: do not reset the ESP32 when opening.
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.dtr = False
    ser.rts = False

    time.sleep(0.3)
    ser.reset_input_buffer()

    n_csi = 0
    deadline = time.time() + secs
    with open(out, "w", encoding="utf-8") as f:
        while time.time() < deadline:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", "ignore").strip()
            except Exception:
                continue
            if line.startswith(("CSI,", "CSIX,")):
                f.write(line + "\n")
                n_csi += 1
    ser.close()
    print(f"captured {n_csi} CSI frames in {secs:.0f}s -> {out}")
    print(f"effective rate: {n_csi / secs:.1f} frames/sec")


if __name__ == "__main__":
    main()
