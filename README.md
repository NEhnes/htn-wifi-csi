# htn-wifi-csi

Contactless human sensing from WiFi Channel State Information (CSI) on ESP32,
built at Hack the North 2026.

Two cheap ESP32 boards form a radio link. A body in the path perturbs the
channel, and we detect it — **no camera, no wearable, works in darkness.**

---

## What we measured (and what we did not)

We tested two claims. One survived.

### Breathing detection — NULL RESULT

Paced breathing at exactly 15 breaths/min (0.25 Hz), with a matched control
recording of an empty room, on the same radio link:

| | prominence @ 0.25 Hz | best in-band | best out-of-band |
|---|---|---|---|
| paced (breathing) | **0.93x** | 2.93x @ 0.534 Hz | 6.89x |
| control (empty) | **0.76x** | 3.90x @ 0.554 Hz | 5.84x |

At the frequency predicted *in advance*, both runs sit **below their own local
background**, and the strongest structure is outside the breathing band
entirely. Ratio paced/control = 1.22x. There is no breathing signal here.

Why: chest displacement is ~5 mm against a 12.3 cm wavelength (~0.04 lambda),
giving a sub-1% amplitude effect — below 8-bit I/Q quantisation and per-packet
AGC noise, in the worst part of a strongly 1/f spectrum.

We report this rather than hiding it. An earlier, naive statistic (in-band peak
vs the *high-frequency* noise floor) reported "24.9x — strong periodic
component" on an **empty room**. Measuring peaks against a *local* background
instead collapsed it to nothing. See `tools/analyze.py`.

### Presence detection — WORKS

Same hardware, weaker and far more defensible claim:

| feature | present | absent | AUC |
|---|---|---|---|
| mean RSSI | -62.93 dBm | -66.85 dBm | 0.886 |
| mean amplitude | 20.30 | 17.71 | 0.907 |
| subcarrier spread | 12.35 | 14.66 | 0.924 |

Nearest-centroid on the 64-subcarrier fading profile: **87.6%** held-out
accuracy. A walk-through transect peaks at **6.6 sigma** against a noise floor
of ~1.4 sigma, with the power channel independently dipping 0.7 dB at the same
instant.

The effect size is roughly **500x larger** than breathing: a body is ~2.5x
change in received power, versus <1% for a moving chest.

### Detection field geometry

Sensitivity follows the first Fresnel zone — an ellipsoid around the TX-RX line,
radius `0.5*sqrt(lambda*d)` at the midpoint:

| link length | radius at midpoint | full width |
|---|---|---|
| 1.5 m | 21 cm | ~43 cm |
| 3 m | 30 cm | ~61 cm |
| 5 m | 39 cm | ~78 cm |

It is **not a tripwire** — it is a volume, thickest at the midpoint and
tapering to nothing at each board. Measured crossing duration matched the
~43 cm prediction.

---

## Hardware

- **2x ESP32** (ESP32-D0WD-V3, classic — needed for `esp_wifi` CSI support)
- TX runs on any USB charger; RX connects to the host over serial

The transmitter broadcasts ESP-NOW frames at a fixed 30 Hz from a fixed MAC
(`02:43:53:49:00:01`) on channel 6. The receiver runs promiscuous, filters to
that one MAC, and streams CSI as hex.

### Two findings that cost us hours

1. **ESP-NOW defaults to a 1 Mbps 802.11b rate. 802.11b is DSSS and has no
   OFDM Long Training Field — so it produces NO CSI at all.** The transmitter
   must pin the rate to OFDM (11g 6 Mbps). Symptom: receiver hears tens of
   thousands of frames from ambient APs, and exactly zero from your own board.
2. **Ambient beacon sniffing is not enough.** Any single access point gives only
   2-4 packets/sec, irregular, from a position you do not control. A dedicated
   transmitter took us from 3.4 Hz to ~31 Hz on a link with known geometry.

---

## Layout

```
firmware/tx/     ESP-NOW transmitter, fixed rate + fixed MAC, OFDM-pinned
firmware/rx/     promiscuous CSI receiver, channel-pinned, MAC-filtered
tools/           capture, analysis, live detection, ML pipeline
data/            our own captures (single TX MAC) and plots
sims/            earlier feasibility simulations
```

### Tools

| script | purpose |
|---|---|
| `status.py` | live link check — frames/sec and RSSI |
| `whoami.py` | identify which COM port is TX vs RX |
| `live.py` | **live presence monitor** — calibrate, then walk in and out |
| `paced.py` | paced-breathing capture with metronome + control mode |
| `analyze.py` | spectral analysis with local-background prominence |
| `compare.py` | fair A/B on the same transmitter |
| `presence.py` | present-vs-absent separability (AUC) |
| `alternate.py` | alternating capture — defeats the time confound |
| `collect.py` / `train.py` | labelled collection + leave-one-rep-out validation |

Build firmware with PlatformIO:

```
pio run -d firmware/tx -t upload
pio run -d firmware/rx -t upload
```

Live presence detection:

```
python tools/live.py --hi 2.5 --lo 1.5
```

---

## Methodology notes

Every negative result here was checked for the boring explanation first:

- **Predict the frequency before looking.** Hunting for any peak in a wide band
  produced a confident false positive on an empty room.
- **Compare against a local spectral background**, not the high-frequency floor.
  CSI noise is strongly 1/f.
- **Use the same radio link on both sides of an A/B.** Our first comparison
  accidentally used two different transmitters.
- **Validate on held-out recordings**, not random windows. Windows seconds apart
  are nearly identical, so a random split reports ~99% and means nothing.
- **Instrument the "am I receiving anything" path first.** Two separate bugs
  (an 11b/OFDM mistake, and a `CSI,` vs `CSIX,` prefix mismatch) each looked
  exactly like "WiFi sensing doesn't work."

The single-capture-per-condition presence result above is still **confounded
with time** — `alternate.py` exists to settle that properly.
