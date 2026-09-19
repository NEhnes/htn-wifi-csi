/* Dedicated CSI transmitter: broadcasts ESP-NOW frames at a fixed rate.
 *
 * Why this exists: sniffing ambient beacons gave only 2-4 packets/sec from any
 * one transmitter, far too slow and too irregular to see breathing. This sends
 * a steady, regular stream we fully control, on a known channel, from a known
 * MAC, so the receiver can lock onto exactly one link.
 *
 * No WiFi credentials and no association: ESP-NOW is connectionless.
 */
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "nvs_flash.h"

#define TX_CHANNEL   6
#define TX_HZ        30          /* packets per second */

static const char *TAG = "csitx";

/* Fixed locally-administered MAC ("CSI" in hex) so the receiver can filter
 * on a constant and we never have to discover it. */
static uint8_t kTxMac[6] = {0x02, 0x43, 0x53, 0x49, 0x00, 0x01};
static const uint8_t kBroadcast[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

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

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    /* Must be set before start() to take effect reliably. */
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, kTxMac));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_channel(TX_CHANNEL, WIFI_SECOND_CHAN_NONE));

    /* CSI is computed from the OFDM Long Training Field. 802.11b is DSSS and
     * has no L-LTF at all, so 11b frames yield NO CSI on the receiver. Make
     * sure the OFDM modes are available. */
    ESP_ERROR_CHECK(esp_wifi_set_protocol(WIFI_IF_STA,
        WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N));

    ESP_ERROR_CHECK(esp_now_init());

    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, kBroadcast, 6);
    peer.channel = TX_CHANNEL;
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    /* ESP-NOW defaults to a 1 Mbps 802.11b rate, which is DSSS and produces no
     * CSI. Pin it to an OFDM 11g rate so every frame carries an L-LTF. */
    esp_now_rate_config_t rate_cfg = {
        .phymode = WIFI_PHY_MODE_11G,
        .rate    = WIFI_PHY_RATE_6M,
        .ersu    = false,
        .dcm     = false,
    };
    esp_err_t rerr = esp_now_set_peer_rate_config(kBroadcast, &rate_cfg);
    if (rerr != ESP_OK) {
        ESP_LOGE(TAG, "set_peer_rate_config: %s - frames may carry no CSI",
                 esp_err_to_name(rerr));
    } else {
        ESP_LOGI(TAG, "rate pinned to OFDM 11g 6 Mbps (L-LTF present)");
    }

    uint8_t mac[6];
    esp_wifi_get_mac(WIFI_IF_STA, mac);
    ESP_LOGI(TAG, "TX MAC %02x:%02x:%02x:%02x:%02x:%02x  channel %d  %d Hz",
             mac[0], mac[1], mac[2], mac[3], mac[4], mac[5], TX_CHANNEL, TX_HZ);
    ESP_LOGI(TAG, "broadcasting - leave this board powered and placed");

    const TickType_t period = pdMS_TO_TICKS(1000 / TX_HZ);
    TickType_t last = xTaskGetTickCount();
    uint32_t seq = 0;
    uint32_t errs = 0;
    uint8_t payload[24];

    while (1) {
        memset(payload, 0xA5, sizeof(payload));
        memcpy(payload, &seq, sizeof(seq));
        esp_err_t se = esp_now_send(kBroadcast, payload, sizeof(payload));
        if (se != ESP_OK) {
            errs++;
        }
        seq++;
        /* Every 2s, not 10s: we want fast confirmation that sends succeed. */
        if ((seq % (TX_HZ * 2)) == 0) {
            ESP_LOGI(TAG, "sent %lu  errors %lu",
                     (unsigned long)seq, (unsigned long)errs);
        }
        /* Fixed cadence: regular sampling is what the analysis needs. */
        vTaskDelayUntil(&last, period);
    }
}
