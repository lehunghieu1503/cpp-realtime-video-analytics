# Stage 1: Build
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake git wget ca-certificates \
    libopencv-dev libeigen3-dev libmosquitto-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
    && tar xzf onnxruntime-linux-x64-1.20.1.tgz \
    && cp -r onnxruntime-linux-x64-1.20.1/lib/* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-1.20.1/include/* /usr/local/include/ \
    && ldconfig \
    && rm -rf onnxruntime-linux-x64-1.20.1*

# Install OpenVINO (Intel CPU, integrated GPU and NPU backends)
RUN wget -q https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/linux/openvino_toolkit_ubuntu22_2025.4.0.20398.8fdad55727d_x86_64.tgz \
    && mkdir /opt/openvino \
    && tar xzf openvino_toolkit_*.tgz -C /opt/openvino --strip-components=1 \
    && rm openvino_toolkit_*.tgz

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
        -DOpenVINO_DIR=/opt/openvino/runtime/cmake \
    && cmake --build build --parallel $(nproc)

# Stage 2: Runtime
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d \
    libopencv-imgcodecs4.5d \
    libopencv-imgproc4.5d \
    libopencv-dnn4.5d \
    libopencv-videoio4.5d \
    libopencv-highgui4.5d \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-libav \
    libmosquitto1 \
    # User-space driver of Intel integrated GPUs, for openvino:GPU. This
    # Ubuntu 22.04 package predates Arc graphics (Core Ultra), which it
    # does not detect.
    intel-opencl-icd \
    wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# User-space driver of the Intel NPU, for openvino:NPU. The kernel driver and
# its firmware come from the host; pass /dev/accel/accel0 into the container.
# Keep the version close to the host's intel-level-zero-npu package.
RUN wget -q https://github.com/intel/linux-npu-driver/releases/download/v1.26.0/linux-npu-driver-v1.26.0.20251125-19665715237-ubuntu2204.tar.gz \
        https://github.com/oneapi-src/level-zero/releases/download/v1.24.2/level-zero_1.24.2+u22.04_amd64.deb \
    && tar xzf linux-npu-driver-*.tar.gz \
    && apt-get update && apt-get install -y --no-install-recommends \
        ./level-zero_*.deb ./intel-driver-compiler-npu_*.deb \
        ./intel-level-zero-npu_*.deb \
    && rm -rf /var/lib/apt/lists/* ./*.deb ./*.ddeb ./*.asc ./*.tar.gz

# Copy the OpenVINO runtime libraries and their plugins
COPY --from=builder /opt/openvino/runtime/lib/intel64 /opt/openvino/lib
COPY --from=builder /opt/openvino/runtime/3rdparty/tbb/lib /opt/openvino/tbb
ENV LD_LIBRARY_PATH=/opt/openvino/lib:/opt/openvino/tbb

# Copy ONNX Runtime shared lib
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Copy application binary and web assets.
# Models are not in the repo; mount them at /app/models when running.
WORKDIR /app
COPY --from=builder /app/build/src/video_analytics .
COPY web/ web/

# Run as non-root; the video group is needed to open /dev/video*. The NPU
# and GPU nodes belong to the host's render group: add it at run time with
# --group-add (docker-compose.laptop.yml does).
RUN useradd -m -G video appuser
USER appuser

EXPOSE 9001

# No display in the container, and the published port needs 0.0.0.0
ENTRYPOINT ["./video_analytics", "--headless", "--bind", "0.0.0.0"]
CMD ["--device", "0", "--port", "9001"]
