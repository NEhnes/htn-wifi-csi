# Configure a QNX Everywhere Raspberry Pi 5 SD card to join a WiFi network.
#
# Run on Windows with the SD card's boot partition mounted:
#   powershell -ExecutionPolicy Bypass -File qnx\sd_wifi.ps1 -Drive D
#
# The password is prompted (masked) and written straight to the card; it is
# never stored anywhere else.
#
# Gotchas this handles, each of which silently breaks WiFi on the Pi:
#   - iPhone hotspot names use a CURLY apostrophe (U+2019), not ASCII '.
#     The script picks the exact SSID from a live scan.
#   - Files must use Unix LF line endings. A CRLF file makes every value end
#     in '\r', so the SSID no longer matches.
#   - Files must be UTF-8 WITHOUT a BOM.
#   - iPhones default to WPA3; this config is WPA2-PSK. Turn ON
#     "Maximize Compatibility" on the phone.

param(
    [string]$Drive = "D",
    [string]$Hostname = "qnxpi-abhi",
    [string]$Ssid = "",
    # Copy the SSID of the network this laptop is connected to right now.
    # Best way to get an iPhone hotspot name exactly right.
    [switch]$UseCurrent
)

# netsh output must be decoded as UTF-8, or the curly apostrophe in iPhone
# hotspot names gets mangled and the Pi looks for a network that doesn't exist.
[Console]::OutputEncoding = [Text.Encoding]::UTF8

if ($UseCurrent) {
    $m = netsh wlan show interfaces | Select-String "^\s+SSID\s+:\s(.+)$" | Select-Object -First 1
    if (-not $m) {
        Write-Host "This laptop is not connected to any WiFi network." -ForegroundColor Red
        exit 1
    }
    $Ssid = $m.Matches[0].Groups[1].Value.Trim()
}

$root = "${Drive}:\"
if (-not (Test-Path (Join-Path $root "qnx_sdp.ifs"))) {
    Write-Host "No qnx_sdp.ifs on ${Drive}: - is this the QNX boot partition?" -ForegroundColor Red
    exit 1
}

if (-not $Ssid) {
    Write-Host "Scanning for WiFi networks..."
    $seen = netsh wlan show networks | Select-String "^SSID \d+ : (.+)$" |
        ForEach-Object { $_.Matches[0].Groups[1].Value.Trim() } |
        Where-Object { $_ } | Sort-Object -Unique
    $phones = @($seen | Where-Object { $_ -match "iPhone|Abhi" })
    if ($phones.Count -eq 1) {
        $Ssid = $phones[0]
        Write-Host "Found: '$Ssid'"
    } else {
        Write-Host "Visible networks:"
        $seen | ForEach-Object { Write-Host "   $_" }
        if ($phones.Count -eq 0) {
            Write-Host "Your hotspot is not visible. Open Settings > Personal Hotspot on the" -ForegroundColor Yellow
            Write-Host "phone (keeps it broadcasting) and turn ON Maximize Compatibility." -ForegroundColor Yellow
        }
        $Ssid = Read-Host "Type the exact SSID (copy it from the list above)"
    }
}

$sec = Read-Host "WiFi password for '$Ssid'" -AsSecureString
$pw = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
if ($pw.Length -lt 8 -or $pw.Length -gt 63) {
    Write-Host "WPA2 passwords must be 8-63 characters." -ForegroundColor Red
    exit 1
}
if ($pw.Contains('"') -or $Ssid.Contains('"')) {
    Write-Host 'Double quotes in the SSID/password are not supported by this script.' -ForegroundColor Red
    exit 1
}

$utf8 = New-Object System.Text.UTF8Encoding($false)   # no BOM
function Write-Unix($name, $lines) {
    $text = ($lines -join "`n") + "`n"                 # LF only
    [IO.File]::WriteAllText((Join-Path $root $name), $text, $utf8)
}

Write-Unix "qnx_config.txt" @(
    "HOSTNAME=$Hostname",
    "IP_ADDR=dhcp",
    "WIFI_SSID=`"$Ssid`"",
    "WIFI_PASS=`"$pw`""
)
# wpa_supplicant accepts the SSID as unquoted hex bytes. That sidesteps every
# encoding question for names containing characters like U+2019.
$ssidHex = ([Text.Encoding]::UTF8.GetBytes($Ssid) | ForEach-Object { $_.ToString("x2") }) -join ""
Write-Unix "wpa_supplicant.conf" @(
    "ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev",
    "update_config=1",
    "country=CA",
    "",
    "network={",
    "    ssid=$ssidHex",
    "    psk=`"$pw`"",
    "    key_mgmt=WPA-PSK",
    "}"
)
Write-Unix "network" @("HOSTNAME=$Hostname")
$pw = $null

Write-Host ""
Write-Host "Written to ${Drive}: - hostname '$Hostname', SSID '$Ssid'." -ForegroundColor Green
Write-Host ("SSID characters: " + (($Ssid.ToCharArray() | ForEach-Object { "U+{0:X4}" -f [int]$_ }) -join " "))
Write-Host "(an iPhone hotspot name should contain U+2019, the curly apostrophe)"
Write-Host "Eject the card safely, put it in the Pi, and power it on."
