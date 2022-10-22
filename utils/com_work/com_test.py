# This example code is in the Public Domain (or CC0 licensed, at your option.)

# Unless required by applicable law or agreed to in writing, this
# software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.

# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals

import socket
import sys
import time

import numpy as np
import tensorflow as tf
import soundfile
import pyaudio
from threading import Thread
import cv2 as cv

# -----------  Config  ----------
PORT = 8088
INTERFACE = 'eth0'
# -------------------------------
audio_data = []
feature_data = []
pred_list = []
started = False

tflite_model_file = "models/audio5_model.tflite"
interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
# print(interpreter.get_tensor_details())
output_details = interpreter.get_output_details()[0]



train_list = ['silence','unknown','air_conditioner','dog_bark','water_sound']

log_file = open(f"log{time.strftime('%Y%m%d_%H%M%S')}.log", 'a')
for i in range(len(train_list)):
    log_file.write(f"category {i}:{train_list[i]} \n")


color_list = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (100, 100, 0),
    (0, 100, 100),
    (100, 0, 100),
    (100, 0, 0),
]
canvas = np.zeros((800, 600, 3))
cv.imshow("predictions", canvas)
cv.waitKey(100)

class AudioThread(Thread):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.running = True

    def run(self):
        pya = pyaudio.PyAudio()
        stream = pya.open(format=pya.get_format_from_width(width=2), channels=1, rate=16000, output=True)
        print("play audio")
        length = 0
        while(self.running):
            
            length = len(audio_data)
            if length == 0:
                break
            wav_audio = np.asarray(audio_data[self.counter:length], dtype=np.int16)
            if wav_audio.shape[0] == 0:
                continue
            self.counter = length
            stream.write(wav_audio.tobytes())
            #print("write sound: ", wav_audio.shape[0]/16000)

        stream.stop_stream()
        stream.close()

        pya.terminate()

    def stop(self):
        self.running = False

class RecThread(Thread):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.running = True

    def run(self):
        print("rec audio")
        length = 0
        time.sleep(3)
        while(self.running):
            
            length = int(len(audio_data) / 16000) * 16000
            #print("rec audio length=", length)
            wav_audio = np.asarray(audio_data[self.counter:length])
            
            if wav_audio.shape[0] > 16000*2:
                waveform = np.array(wav_audio/32768.0, dtype=np.float32)
                self.counter = length
                soundfile.write(f"wav/rec_audio_{time.strftime('%Y%m%d_%H%M%S')}.wav", waveform, 16000)
                print("saved wav file ", waveform.shape[0] / 16000)
                time.sleep(3)


    def stop(self):
        self.running = False

class FeatureThread(Thread):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.running = True

    def run(self):
        print("rec feature")
        length = 0
        time.sleep(0.2)
        while(self.running):
            length = len(feature_data)
            #print("rec feature length=", length)
            if length > self.counter:
                for idx in range(int(self.counter/4), int(length/4)):
                    arr1 = np.frombuffer(feature_data[idx*4+1], dtype=np.int8)
                    arr2 = np.frombuffer(feature_data[idx*4+3], dtype=np.int8)
                    cnt = feature_data[idx*4]
                    # np.save(f"wav/feature_{cnt}_0_{time.strftime('%Y%m%d_%H%M%S')}.npy", arr1)
                    # np.save(f"wav/feature_{cnt}_1_{time.strftime('%Y%m%d_%H%M%S')}.npy", arr2)
                    

                    test_image = np.array(arr1.tolist() + arr2.tolist(), dtype=np.int8)
                    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])

                    #print("test_image shape:", test_image.shape)
                    interpreter.set_tensor(input_details["index"], test_image)
                    interpreter.invoke()
                    output = interpreter.get_tensor(output_details["index"])[0]
                    
                    pred_list.append(output)
                    pred_list.append(cnt)
                    log_file.write(f"time {cnt / 16000}s :{output+128} \n")
                    #print(f"output:{cnt / 16000}s ", output)
                self.counter = length

            time.sleep(0.2)    


    def stop(self):
        self.running = False

def tcp_client(address, payload):
    global audio_data
    
    cur_pred_pos = 0

    for res in socket.getaddrinfo(address, PORT, socket.AF_UNSPEC,
                                  socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        family_addr, socktype, proto, canonname, addr = res
    try:
        sock = socket.socket(family_addr, socket.SOCK_STREAM)
        sock.settimeout(60.0)
    except socket.error as msg:
        print('Could not create socket: ' + str(msg[0]) + ': ' + msg[1])
        raise
    try:
        sock.connect(addr)
    except socket.error as msg:
        print('Could not open socket: ', msg)
        sock.close()
        raise
    started = False
    try:
        while(1):
            try:
                data = sock.recv(2048)
            except:
                print("sock error")
                break
            
            if not data:
                print("not data")
                break
            arr = np.frombuffer(data, dtype=np.int16)

            #print("arr.shape=", arr.shape)
            
            sock.sendall(payload.encode())
            
            if arr.shape[0] == 320:
                audio_data += arr.tolist()
            elif arr.shape[0] == 490:
                feature_data.append(len(audio_data))
                feature_data.append(data)
            else:
                print("truncated packets:", arr.shape[0])

            if started == False:
                print("create thread")
                started = True
                t = AudioThread()
                t.start()
                # t2 = RecThread()
                # t2.start()
                t3 = FeatureThread()
                t3.start()
            canvas.fill(0) #= np.zeros((800, 600, 3))

            cur_cnt = len(pred_list)
            for i in range(cur_pred_pos, int(cur_cnt / 2)):
                output = pred_list[i*2]
                print(f"predictions output({pred_list[i*2+1]/16000}s):", output)
                
                # canvas = np.zeros((800, 600, 3))
                cv.putText(canvas, f"{pred_list[i*2+1]/16000}s", (300, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
                for j in range(len(train_list)):
                    cv.putText(canvas, f"{train_list[j]}({output[j]+128})", (10, 100*j+40), cv.FONT_HERSHEY_SIMPLEX, 1, color_list[j], 1, cv.LINE_AA)
                    cv.rectangle(canvas, (10, 100*j+50), ((output[j]+256), 100*j+90), color_list[j], -1)
                cv.imshow("predictions", canvas)
            
            cv.waitKey(1)    
                
            cur_pred_pos = int(cur_cnt / 2)
    except KeyboardInterrupt:
        print("Exiting") 

    wav_audio = np.asarray(audio_data)
    print(wav_audio.shape)
    waveform = np.array(wav_audio/32768.0, dtype=np.float32)
    soundfile.write(f"wav/audio_{time.strftime('%Y%m%d_%H%M%S')}.wav", waveform, 16000)
    print("saved wav file ", waveform.shape[0] / 16000)
    t.stop()
    # t2.stop()
    t3.stop()

    
if __name__ == '__main__':
    if sys.argv[2:]:    # if two arguments provided:
        # Usage: example_test.py <server_address> <message_to_send_to_server>
        tcp_client(sys.argv[1], sys.argv[2])
    # else:               # otherwise run standard example test as in the CI
    #     test_examples_protocol_socket_tcpserver()
