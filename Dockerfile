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

WORKDIR /app
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF \
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
    libmosquitto1 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime shared lib
COPY --from=builder /usr/local/lib/libonnxruntime* /usr/local/lib/
RUN ldconfig

# Copy application binary and web assets.
# Models are not in the repo; mount them at /app/models when running.
WORKDIR /app
COPY --from=builder /app/build/src/video_analytics .
COPY web/ web/

# Run as non-root; the video group is needed to open /dev/video*
RUN useradd -m -G video appuser
USER appuser

EXPOSE 9001

# No display in the container, and the published port needs 0.0.0.0
ENTRYPOINT ["./video_analytics", "--headless", "--bind", "0.0.0.0"]
CMD ["--device", "0", "--port", "9001"]
