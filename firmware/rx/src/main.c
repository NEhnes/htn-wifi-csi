/* CSI receiver locked to our own dedicated ESP-NOW transmitter.
 *
 * v1 sniffed ambient beacons: only 2-4 frames/sec per transmitter, irregular,
 * and from APs whose position we did not control. Paced-breathing vs control
 * came out null because of it.
 *
 * v2 listens to exactly one transmitter (fixed MAC, fixed channel, fixed rate).
 * Output is hex so 30 Hz of full I/Q fits inside 115200 baud:
 *   CSIX,<t_us>,<mac>,<rssi>,<channel>,<len>,<hex I/Q>
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_netif.h"
#include "nvs_flash.h"

#define RX_CHANNEL 6                      /* must match the transmitter */

static const char *TAG = "csi";

/* The transmitter's fixed locally-administered MAC. */
static const uint8_t kTxMac[6] = {0x02, 0x43, 0x53, 0x49, 0x00, 0x01};

static volatile uint32_t g_total;         /* every frame heard */
static volatile uint32_t g_matched;       /* frames from our transmitter */

static const char kHex[] = "0123456789abcdef";

static void csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }
    g_total++;

    /* Only our transmitter: everything else wastes UART bandwidth and is not
     * comparable anyway (different path geometry). */
    if (memcmp(info->mac, kTxMac, 6) != 0) {
        return;
    }
    g_matched++;

    int len = info->len;
    if (len > 256) {
        len = 256;
    }
    const int8_t *buf = info->buf;

    /* Hex is ~1.8x smaller than decimal CSV here, which is what lets 30 Hz fit. */
    char hex[513];
    for (int i = 0; i < len; i++) {
        uint8_t b = (uint8_t)buf[i];
        hex[2 * i]     = kHex[b >> 4];
        hex[2 * i + 1] = kHex[b & 0x0F];
    }
    hex[2 * len] = '\0';

    const uint8_t *m = info->mac;
    printf("CSIX,%lld,%02x%02x%02x%02x%02x%02x,%d,%u,%d,%s\n",
           (long long)esp_timer_get_time(),
           m[0], m[1], m[2], m[3], m[4], m[5],
           info->rx_ctrl.rssi, info->rx_ctrl.channel, len, hex);
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    /* Pinned, not scanned: v1 picked a different channel on each boot, which
     * silently made runs non-comparable. */
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    ESP_ERROR_CHECK(esp_wifi_set_channel(RX_CHANNEL, WIFI_SECOND_CHAN_NONE));

    wifi_csi_config_t csi_cfg = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = false,
        .manu_scale        = false,
        .shift             = 0,
    };

    esp_err_t e;
    e = esp_wifi_set_csi_rx_cb(csi_rx_cb, NULL);
    if (e != ESP_OK) ESP_LOGE(TAG, "set_csi_rx_cb: %s", esp_err_to_name(e));
    e = esp_wifi_set_csi_config(&csi_cfg);
    if (e != ESP_OK) ESP_LOGE(TAG, "set_csi_config: %s", esp_err_to_name(e));
    e = esp_wifi_set_csi(true);
    if (e != ESP_OK) ESP_LOGE(TAG, "set_csi: %s", esp_err_to_name(e));

    ESP_LOGI(TAG, "RX on channel %d, locked to TX %02x:%02x:%02x:%02x:%02x:%02x",
             RX_CHANNEL, kTxMac[0], kTxMac[1], kTxMac[2],
             kTxMac[3], kTxMac[4], kTxMac[5]);

    uint32_t prev = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        uint32_t m = g_matched;
        /* Status line tells you instantly whether the TX is being heard,
         * without having to run the analysis first. */
        ESP_LOGI(TAG, "STATUS heard=%lu from_TX=%lu (%.1f Hz)",
                 (unsigned long)g_total, (unsigned long)m, (m - prev) / 5.0);
        if (m == prev) {
            ESP_LOGW(TAG, "no packets from the TX board - is it powered, "
                          "on channel %d, and within range?", RX_CHANNEL);
        }
        prev = m;
    }
}
