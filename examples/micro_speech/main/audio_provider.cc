/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "audio_provider.h"

#include <cstdlib>
#include <cstring>

// FreeRTOS.h must be included before some of the following dependencies.
// Solves b/150260343.
// clang-format off
#include "freertos/FreeRTOS.h"
// clang-format on

#include "driver/i2s.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/task.h"
#include "ringbuf.h"
#include "micro_model_settings.h"



#include <string.h>
#include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
// #include "esp_log.h"
#include "nvs_flash.h"

#include "lwip/err.h"
#include "lwip/sys.h"

#include "esp_netif.h"

#include "lwip/sockets.h"

using namespace std;

static const char* TAG = "TF_LITE_AUDIO_PROVIDER";
/* ringbuffer to hold the incoming audio data */
ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;
/* model requires 20ms new data from g_audio_capture_buffer and 10ms old data
 * each time , storing old data in the histrory buffer , {
 * history_samples_to_keep = 10 * 16 } */
constexpr int32_t history_samples_to_keep =
    ((kFeatureSliceDurationMs - kFeatureSliceStrideMs) *
     (kAudioSampleFrequency / 1000));
/* new samples to get each time from ringbuffer, { new_samples_to_get =  20 * 16
 * } */
constexpr int32_t new_samples_to_get =
    (kFeatureSliceStrideMs * (kAudioSampleFrequency / 1000));

namespace {
// int16_t g_audio_sample_buffer[16000];
int16_t g_audio_sample_buffer[320];
int32_t sample_counter = 0;
int32_t audio_counter = 0;
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
}  // namespace

const int32_t kAudioCaptureBufferSize = 80000;
const int32_t i2s_bytes_to_read = 3200;

static void i2s_init(void) {
  // Start listening for audio: MONO @ 16KHz
  i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_TX),
      .sample_rate = 16000,
      .bits_per_sample = (i2s_bits_per_sample_t)16,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_I2S,
      .intr_alloc_flags = 0,
      .dma_buf_count = 3,
      .dma_buf_len = 300,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = -1,
  };
  i2s_pin_config_t pin_config = {
      .bck_io_num = 5,    // IIS_SCLK
      .ws_io_num = 25,     // IIS_LCLK
      .data_out_num = 26,  // IIS_DSIN
      .data_in_num = 35,   // IIS_DOUT
  };

  // i2s_pin_config_t pin_config = {
  //     .bck_io_num = 26,    // IIS_SCLK
  //     .ws_io_num = 32,     // IIS_LCLK
  //     .data_out_num = -1,  // IIS_DSIN
  //     .data_in_num = 33,   // IIS_DOUT
  // };
  esp_err_t ret = 0;
  ret = i2s_driver_install((i2s_port_t)1, &i2s_config, 0, NULL);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_driver_install");
  }
  ret = i2s_set_pin((i2s_port_t)1, &pin_config);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in i2s_set_pin");
  }

  ret = i2s_zero_dma_buffer((i2s_port_t)1);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Error in initializing dma buffer with 0");
  }
}

static void CaptureSamples(void* arg) {
  ESP_LOGE(TAG, "CaptureSamples read : %d", i2s_bytes_to_read);
  size_t bytes_read = i2s_bytes_to_read;
  uint8_t i2s_read_buffer[i2s_bytes_to_read] = {};
  i2s_init();
  while (1) {
    /* read 100ms data at once from i2s */
    i2s_read((i2s_port_t)1, (void*)i2s_read_buffer, i2s_bytes_to_read,
             &bytes_read, 10);
    if (bytes_read <= 0) {
      ESP_LOGE(TAG, "Error in I2S read : %d", bytes_read);
    } else {
      if (bytes_read < i2s_bytes_to_read) {
        ESP_LOGW(TAG, "Partial I2S read");
      }
      /* write bytes read by i2s into ring buffer */
      int bytes_written = rb_write(g_audio_capture_buffer,
                                   (uint8_t*)i2s_read_buffer, bytes_read, 10);
      /* update the timestamp (in ms) to let the model know that new data has
       * arrived */
      g_latest_audio_timestamp +=
          ((1000 * (bytes_written / 2)) / kAudioSampleFrequency);
      if (bytes_written <= 0) {
        ESP_LOGE(TAG, "Could Not Write in Ring Buffer: %d ", bytes_written);
      } else if (bytes_written < bytes_read) {
        ESP_LOGW(TAG, "Partial Write");
      }
    }
  }
  vTaskDelete(NULL);
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  ESP_LOGE(TAG, "InitAudioRecording kAudioCaptureBufferSize=%d g_latest_audio_timestamp=%d", kAudioCaptureBufferSize, g_latest_audio_timestamp);
  g_audio_capture_buffer = rb_init("tf_ringbuffer", kAudioCaptureBufferSize);
  if (!g_audio_capture_buffer) {
    ESP_LOGE(TAG, "Error creating ring buffer");
    return kTfLiteError;
  }
  /* create CaptureSamples Task which will get the i2s_data from mic and fill it
   * in the ring buffer */
  ESP_LOGE(TAG, "InitAudioRecording CaptureSamples");
  
  xTaskCreate(CaptureSamples, "CaptureSamples", 1024 * 32, NULL, 10, NULL);
  while (!g_latest_audio_timestamp) {
  }
  ESP_LOGI(TAG, "Audio Recording started");
  return kTfLiteOk;
}

void send_audio_packets(int sock, char *buff, int snd_len)
{
  // ESP_LOGE(TAG, "send_audio_packets: sock %d snd_len=%d", sock, snd_len);
  int to_write = snd_len;
  while (to_write > 0) {
      int written = send(sock, buff + (snd_len - to_write), to_write, 0);
      if (written < 0) {
          ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
      }
      to_write -= written;
  }

  int len;
  char rx_buffer[128];

  len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
  if (len < 0) {
      ESP_LOGE(TAG, "Error occurred during receiving: errno %d", errno);
  } else if (len == 0) {
      ESP_LOGW(TAG, "Connection closed");
  } else {
      rx_buffer[len] = 0; // Null-terminate whatever is received and treat it like a string
      // ESP_LOGI(TAG, "Received %d bytes: %s", len, rx_buffer);
      // send() can return less bytes than supplied length.
      // Walk-around for robust implementation.       
  }
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples, int tcp_sock) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  /* copy 160 samples (320 bytes) into output_buff from history */
  memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
         history_samples_to_keep * sizeof(int16_t));

  /* copy 320 samples (640 bytes) from rb at ( int16_t*(g_audio_output_buffer) +
   * 160 ), first 160 samples (320 bytes) will be from history */
  int32_t bytes_read =
      rb_read(g_audio_capture_buffer,
              ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
              new_samples_to_get * sizeof(int16_t), 10);
  if (bytes_read < 0) {
    ESP_LOGE(TAG, " Model Could not read data from Ring Buffer");
  } else if (bytes_read < new_samples_to_get * sizeof(int16_t)) {
    ESP_LOGD(TAG, "RB FILLED RIGHT NOW IS %d",
             rb_filled(g_audio_capture_buffer));
    ESP_LOGD(TAG, " Partial Read of Data by Model ");
    ESP_LOGV(TAG, " Could only read %d bytes when required %d bytes ",
             bytes_read, new_samples_to_get * sizeof(int16_t));
  }

  /* copy 320 bytes from output_buff into history */
  memcpy((void*)(g_history_buffer),
         (void*)(g_audio_output_buffer + new_samples_to_get),
         history_samples_to_keep * sizeof(int16_t));


  //
  memcpy((void*)(g_audio_sample_buffer+sample_counter * new_samples_to_get),
         (void*)(g_audio_output_buffer + history_samples_to_keep),
         new_samples_to_get * sizeof(int16_t));

  send_audio_packets(tcp_sock, (char*)(g_audio_sample_buffer+sample_counter * new_samples_to_get), new_samples_to_get*2);
  
  // sample_counter += 1;
  // if (sample_counter < 50){
    
  // }
  // else{
  //   sample_counter = 0;
    // int32_t idx = 0;
  //   audio_counter += 1;

  //   if (audio_counter % 10 == 8){
      // for (idx=0; idx<1; idx++){ //for (idx=0; idx<1600; idx++){
      //   ESP_LOGE(TAG, "%4d: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, ", idx, g_audio_sample_buffer[idx*10], g_audio_sample_buffer[idx*10+1], g_audio_sample_buffer[idx*10+2], g_audio_sample_buffer[idx*10+3], g_audio_sample_buffer[idx*10+4], g_audio_sample_buffer[idx*10+5], g_audio_sample_buffer[idx*10+6], g_audio_sample_buffer[idx*10+7], g_audio_sample_buffer[idx*10+8], g_audio_sample_buffer[idx*10+9]);
      // }
  //   }        
  // }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return g_latest_audio_timestamp; }
