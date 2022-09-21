/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "lwip/err.h"
#include "lwip/sys.h"

#include "esp_netif.h"

#include "lwip/sockets.h"
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace




#define PORT 8088 //CONFIG_EXAMPLE_PORT

#define EXAMPLE_ESP_WIFI_SSID      "ESP32_TEST"
#define EXAMPLE_ESP_WIFI_PASS      "12345678"
#define EXAMPLE_ESP_WIFI_CHANNEL   1
#define EXAMPLE_MAX_STA_CONN       4

static const char *TAG = "example";
int tcp_sock = -1;
static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                                    int32_t event_id, void* event_data)
{
    if (event_id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t* event = (wifi_event_ap_staconnected_t*) event_data;
        ESP_LOGI(TAG, "station " MACSTR " join, AID=%d",
                 MAC2STR(event->mac), event->aid);
    } else if (event_id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t* event = (wifi_event_ap_stadisconnected_t*) event_data;
        ESP_LOGI(TAG, "station " MACSTR " leave, AID=%d",
                 MAC2STR(event->mac), event->aid);
    }
}

void wifi_init_softap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        NULL));

    // wifi_config_t wifi_config = {
    //     .ap = {
    //         .ssid = EXAMPLE_ESP_WIFI_SSID,
    //         .ssid_len = strlen(EXAMPLE_ESP_WIFI_SSID),
    //         .channel = EXAMPLE_ESP_WIFI_CHANNEL,
    //         .password = EXAMPLE_ESP_WIFI_PASS,
    //         .max_connection = EXAMPLE_MAX_STA_CONN,
    //         .authmode = WIFI_AUTH_WPA_WPA2_PSK
    //     },
    // };

    wifi_config_t wifi_config;
    strcpy((char*)wifi_config.ap.ssid, EXAMPLE_ESP_WIFI_SSID);
    wifi_config.ap.ssid_len = strlen(EXAMPLE_ESP_WIFI_SSID);
    wifi_config.ap.channel = EXAMPLE_ESP_WIFI_CHANNEL;
    strcpy((char*)wifi_config.ap.password, EXAMPLE_ESP_WIFI_PASS);
    wifi_config.ap.max_connection = EXAMPLE_MAX_STA_CONN;
    wifi_config.ap.authmode = WIFI_AUTH_WPA_WPA2_PSK;
    
    if (strlen(EXAMPLE_ESP_WIFI_PASS) == 0) {
        wifi_config.ap.authmode = WIFI_AUTH_OPEN;
    }

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config((wifi_interface_t)ESP_IF_WIFI_AP, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "wifi_init_softap finished. SSID:%s password:%s channel:%d",
             EXAMPLE_ESP_WIFI_SSID, EXAMPLE_ESP_WIFI_PASS, EXAMPLE_ESP_WIFI_CHANNEL);
}


// static void do_retransmit(const int sock)
// {
//     int len;
//     char rx_buffer[128];

//     do {
//         len = recv(sock, rx_buffer, sizeof(rx_buffer) - 1, 0);
//         if (len < 0) {
//             ESP_LOGE(TAG, "Error occurred during receiving: errno %d", errno);
//         } else if (len == 0) {
//             ESP_LOGW(TAG, "Connection closed");
//         } else {
//             rx_buffer[len] = 0; // Null-terminate whatever is received and treat it like a string
//             ESP_LOGI(TAG, "Received %d bytes: %s", len, rx_buffer);

//             // send() can return less bytes than supplied length.
//             // Walk-around for robust implementation. 
//             int to_write = len;
//             while (to_write > 0) {
//                 int written = send(sock, rx_buffer + (len - to_write), to_write, 0);
//                 if (written < 0) {
//                     ESP_LOGE(TAG, "Error occurred during sending: errno %d", errno);
//                 }
//                 to_write -= written;
//             }
//         }
//     } while (len > 0);
// }

void tcp_server_task(int addr_family)
{
    char addr_str[128];
    // int addr_family = (int)pvParameters;
    int ip_protocol = 0;
    struct sockaddr_in6 dest_addr;

    if (addr_family == AF_INET) {
        struct sockaddr_in *dest_addr_ip4 = (struct sockaddr_in *)&dest_addr;
        dest_addr_ip4->sin_addr.s_addr = htonl(INADDR_ANY);
        dest_addr_ip4->sin_family = AF_INET;
        dest_addr_ip4->sin_port = htons(PORT);
        ip_protocol = IPPROTO_IP;
    } else if (addr_family == AF_INET6) {
        bzero(&dest_addr.sin6_addr.un, sizeof(dest_addr.sin6_addr.un));
        dest_addr.sin6_family = AF_INET6;
        dest_addr.sin6_port = htons(PORT);
        ip_protocol = IPPROTO_IPV6;
    }

    int listen_sock = socket(addr_family, SOCK_STREAM, ip_protocol);
    if (listen_sock < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }
#if defined(CONFIG_EXAMPLE_IPV4) && defined(CONFIG_EXAMPLE_IPV6)
    // Note that by default IPV6 binds to both protocols, it is must be disabled
    // if both protocols used at the same time (used in CI)
    int opt = 1;
    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(listen_sock, IPPROTO_IPV6, IPV6_V6ONLY, &opt, sizeof(opt));
#endif

    ESP_LOGI(TAG, "Socket created");

    int err = bind(listen_sock, (struct sockaddr *)&dest_addr, sizeof(dest_addr));
    if (err != 0) {
        ESP_LOGE(TAG, "Socket unable to bind: errno %d", errno);
        ESP_LOGE(TAG, "IPPROTO: %d", addr_family);
        // goto CLEAN_UP;
        return;
    }
    ESP_LOGI(TAG, "Socket bound, port %d", PORT);

    err = listen(listen_sock, 1);
    if (err != 0) {
        ESP_LOGE(TAG, "Error occurred during listen: errno %d", errno);
        // goto CLEAN_UP;
        return;
    }

    while (1) {

        ESP_LOGI(TAG, "Socket listening");

        struct sockaddr_in6 source_addr; // Large enough for both IPv4 or IPv6
        uint addr_len = sizeof(source_addr);
        tcp_sock = accept(listen_sock, (struct sockaddr *)&source_addr, &addr_len);
        if (tcp_sock < 0) {
            ESP_LOGE(TAG, "Unable to accept connection: errno %d", errno);
            break;
        }

        // Convert ip address to string
        if (source_addr.sin6_family == PF_INET) {
            inet_ntoa_r(((struct sockaddr_in *)&source_addr)->sin_addr.s_addr, addr_str, sizeof(addr_str) - 1);
        } else if (source_addr.sin6_family == PF_INET6) {
            inet6_ntoa_r(source_addr.sin6_addr, addr_str, sizeof(addr_str) - 1);
        }
        ESP_LOGI(TAG, "Socket accepted ip address: %s", addr_str);

    //     do_retransmit(tcp_sock);

    //     shutdown(tcp_sock, 0);
    //     close(tcp_sock);
        break;
    }

// CLEAN_UP:
    // close(listen_sock);
    // vTaskDelete(NULL);
}

// The name of this function is important for Arduino compatibility.
void setup() {

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] !=
       (kFeatureSliceCount * kFeatureSliceSize)) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;

  //// wifi setup
  ESP_ERROR_CHECK(nvs_flash_init());
  wifi_init_softap();
  // xTaskCreate(tcp_server_task, "tcp_server", 4096, (void*)AF_INET, 5, NULL);

  tcp_server_task(AF_INET);
}

void send_feature_packets(int sock, char *buff, int snd_len)
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

// The name of this function is important for Arduino compatibility.
void loop() {
  ESP_LOGI(TAG, "Loop --------------------");
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices, tcp_sock);
  TF_LITE_REPORT_ERROR(error_reporter, "Feature generation--------");
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  send_feature_packets(tcp_sock, (char*)feature_buffer, kFeatureElementCount/2);
  send_feature_packets(tcp_sock, (char*)feature_buffer + kFeatureElementCount/2, kFeatureElementCount/2);

  // Run the model on the spectrogram input and make sure it succeeds.
  // TfLiteStatus invoke_status = interpreter->Invoke();
  // if (invoke_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  //   return;
  // }

  // // Obtain a pointer to the output tensor
  // TfLiteTensor* output = interpreter->output(0);
  // // Determine whether a command was recognized based on the output of inference
  // const char* found_command = nullptr;
  // uint8_t score = 0;
  // bool is_new_command = false;
  // TfLiteStatus process_status = recognizer->ProcessLatestResults(
  //     output, current_time, &found_command, &score, &is_new_command);
  // if (process_status != kTfLiteOk) {
  //   TF_LITE_REPORT_ERROR(error_reporter,
  //                        "RecognizeCommands::ProcessLatestResults() failed");
  //   return;
  // }
  // // Do something based on the recognized command. The default implementation
  // // just prints to the error console, but you should replace this with your
  // // own function for a real application.
  // RespondToCommand(error_reporter, current_time, found_command, score,
  //                  is_new_command);
}
