# Getting the QNX Pi 5 onto WiFi — what was wrong

Diagnosed from the SD card our team had been trying to get online. The image
is QNX 8.0.5 "QNX Everywhere" quickstart for Raspberry Pi 5 (confirmed from the
build options embedded in `qnx_sdp.ifs`: default hostname `qnxpi`, IP `dhcp`).

## Status

**Diagnosed, not yet verified on hardware.** Nothing here counts as fixed until
the Pi actually gets an IP address and we can SSH in.

## What we found on the card

### 1. `qnx_config.txt` had Windows (CRLF) line endings — most likely culprit

Checked byte-by-byte: 3 of the 4 lines ended in `\r\n` instead of `\n`. QNX's
shell reads `IP_ADDR=dhcp\r` as the value `dhcp` **plus a hidden carriage
return**. So:

- `IP_ADDR` became `dhcp\r`, which is not the word `dhcp`, so DHCP may never
  have been requested. The Pi could join WiFi and still never get an IP.
- `WIFI_SSID` became `Burger\r`, which matches no network.
- `HOSTNAME` got the same stray character.

Only the last line (the password) was clean, because it had no line ending
at all. This is the classic result of editing the file in Windows Notepad or
similar.

### 2. The three config files disagreed with each other

| file | hostname |
|---|---|
| `qnx_config.txt` | `qnxpi-JJ` |
| `network` | `qnxpi71` |
| image default | `qnxpi` |

Different guides were followed at different times. That matters for two
reasons: we don't know which file the boot scripts read (see below), and
anyone trying `ssh qnxuser@qnxpi.local` was looking for a name the Pi wasn't
using.

### 3. WPA3 vs WPA2 (if the network is an iPhone hotspot)

Every iPhone hotspot visible from our laptop advertised **WPA3-Personal**.
The card's `wpa_supplicant.conf` uses `key_mgmt=WPA-PSK`, which is **WPA2
only**. A WPA2-only client cannot join a WPA3-only network. Fix on the phone:
Settings → Personal Hotspot → **Maximize Compatibility ON**.

### 4. Harmless, but worth knowing

The card had been edited on a Mac too (`.Spotlight-V100`, `.fseventsd`,
`._recovery.bin`). These don't affect WiFi.

## What we could not check

The scripts that actually bring WiFi up are on the QNX6 system partition,
which Windows cannot read. So we cannot prove whether the Pi reads
`qnx_config.txt`, `wpa_supplicant.conf`, or both. **The fix is to make all
three files agree**, which `sd_wifi.ps1` does.

## The fix

`qnx/sd_wifi.ps1` rewrites `qnx_config.txt`, `wpa_supplicant.conf`, and
`network` so that they:

- use Unix `\n` line endings only
- are UTF-8 **without** a BOM
- agree on hostname (`qnxpi-abhi`), SSID, and password
- take the SSID exactly as it appears in a live scan (iPhone names contain a
  curly apostrophe `’`, not `'`)
- prompt for the password (hidden), so it is never saved anywhere else

Originals were backed up before anything was changed.

```
powershell -ExecutionPolicy Bypass -File qnx\sd_wifi.ps1 -Drive D -Ssid "QNX"
```

## Picking a network

| network | notes |
|---|---|
| `QNX` | 2.4 GHz, strong signal at the makerspace. Probably set up for these Pi kits. **Try this first.** Ask the QNX booth for the password. |
| `HackTheNorth` | 5 GHz only near us. Event networks often **block device-to-device traffic**, so the Pi can be online but unreachable from a laptop. |
| iPhone hotspot | Works if Maximize Compatibility is on. Uses phone data. |

Whichever you choose, **the laptop must be on the same network** to SSH in.

## After booting

1. Put the card in the Pi and power it on. Give it about a minute.
2. Put the laptop on the same network.
3. `ssh qnxuser@qnxpi-abhi.local`. If `.local` names don't resolve, find the
   Pi's IP in the router or hotspot's list of connected devices.

If it still won't connect, the next step is the Pi's serial console
(`enable_uart=1` is already set in `config.txt`), which shows the boot log
and any WiFi errors directly.
