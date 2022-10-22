ARG USER=nomo
ARG UID=1000

FROM espressif/idf:v4.4.1

ARG UID
ARG USER

# Create Non-Root User
RUN ["dash", "-c", "\
    addgroup \
     --gid ${UID} \
     \"${USER}\" \
 && adduser \
     --disabled-password \
     --gecos \"\" \
     --ingroup \"${USER}\" \
     --uid ${UID} \
     \"${USER}\" \
 && usermod \
     --append \
     --groups \"dialout,plugdev\" \
     \"${USER}\" \
 && git clone --recursive https://github.com/espressif/esp-adf.git /opt/esp/adf \
 && cd /opt/esp/adf \
 && git checkout origin/release/v2.4 --recurse-submodules \
"]

COPY ./idf_patches /opt/esp/idf_patches

# Fix the ADF environment so it doesn't conflict with NimBLE
# https://github.com/espressif/esp-adf/issues/647
#   - Removes bluetooth_service component from IDF
#   - Removes blufi files from compilation

# Fix BLE pairing prompt in IDF 4.4.1
# https://github.com/espressif/esp-idf/issues/8297
#   - Apply patch that prevents pairing message during provisioning
RUN ["dash", "-c", "\
    sed -i '/blufi\\//d' /opt/esp/adf/components/esp_peripherals/CMakeLists.txt \
 && rm -rf /opt/esp/adf/components/bluetooth_service \
 && cd /opt/esp/idf \
 && git apply ../idf_patches/ble_pair_prompt.patch \
 && git apply /opt/esp/adf/idf_patches/idf_v4.4_freertos.patch \
"]

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver
# Dependencies to get TianLong's code to run
RUN ["dash", "-c", "\
    apt update \
 && apt install -y portaudio19-dev python3-pyaudio libsndfile1-dev ffmpeg libsm6 libgtk2.0-dev libopencv-dev python3-opencv qt5-default \
"]

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

RUN echo "export ADF_PATH=/opt/esp/adf" >> /home/nomo/.bashrc && echo "source /opt/esp/idf/export.sh" >> /home/nomo/.bashrc && echo "git config --global --add safe.directory /opt/esp/idf" >> /home/nomo/.bashrc && echo "git config --global --add safe.directory /opt/esp/adf" >> /home/nomo/.bashrc && echo "git config --global --add safe.directory /opt/esp/idf/components/openthread/openthread" >> /home/nomo/.bashrc
ENV ADF_PATH=/opt/esp/adf
ENV LC_ALL=C