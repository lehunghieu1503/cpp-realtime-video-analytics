# Development Guide

## Prerequisites

- **OS:** Linux (Ubuntu 22.04+ recommended)
- **Compiler:** GCC 11+ or Clang 14+ with C++17 support
- **CMake:** 3.20+
- **OpenCV:** 4.6+ (`sudo apt install libopencv-dev`)
- **ONNX Runtime:** 1.17.0 (see installation below)
- **USB Camera:** V4L2 compatible (check with `v4l2-ctl --list-devices`)

## Install ONNX Runtime

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xzf onnxruntime-linux-x64-1.17.0.tgz
sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo ldconfig
```

For GPU support (FP16 inference), install `onnxruntime-gpu` instead:
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
```

## Get YOLOv8 Model

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx
mkdir -p models && mv yolov8n.onnx models/
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Debug | Release for optimized build |
| `ENABLE_ASAN` | OFF | AddressSanitizer (memory errors) |
| `ENABLE_TSAN` | OFF | ThreadSanitizer (data races) |

## Run

```bash
# Live mode (camera 0, localhost only)
build/src/video_analytics --device 0

# Live mode on all interfaces (no authentication: anyone on the network can watch)
build/src/video_analytics --device 0 --bind 0.0.0.0 --port 9001

# Custom port and worker threads
build/src/video_analytics --device 0 --port 8080 --workers 4

# No local window (SSH session, server): dashboard only
build/src/video_analytics --device 0 --headless

# Custom model
build/src/video_analytics --device 0 --model models/yolov8m.onnx

# Benchmark mode (video file, 300 frames)
build/src/video_analytics --benchmark --input video.avi --frames 300

# Benchmark mode with custom model and output report
build/src/video_analytics --benchmark --input video.avi --frames 500 --model models/yolov8m.onnx --output report.md
```

## Testing

```bash
# All tests
build/tests/test_pipeline
build/tests/test_serializer
build/tests/test_latency_tracker
build/tests/test_sse_server
build/tests/test_inference
```

Test structure:
- `test_pipeline` — BoundedQueue operations, thread safety, FrameData structs
- `test_inference` — ONNX model load, tensor shapes, detection output, latency
- `test_serializer` — JSON serialization, rounding, class name mapping
- `test_latency_tracker` — Percentile math, report generation
- `test_sse_server` — SSE event delivery, worker thread release on disconnect

## Docker

```bash
# Build image (multi-stage: builder + runtime, ~400 MB)
docker build -t video-analytics .

# Run with camera access, dashboard reachable from this machine only.
# The model is mounted, it is not part of the image.
docker run --rm --device /dev/video0 -v "$(pwd)/models:/app/models:ro" \
  -p 127.0.0.1:9001:9001 video-analytics

# Custom device and port
docker run --rm --device /dev/video1 -v "$(pwd)/models:/app/models:ro" \
  -p 127.0.0.1:8080:8080 video-analytics --device 1 --port 8080

# Expose to the network (no authentication: anyone on the network can watch)
docker run --rm --device /dev/video0 -v "$(pwd)/models:/app/models:ro" \
  -p 9001:9001 video-analytics
```

The container always runs with `--headless --bind 0.0.0.0`. It runs as `appuser`
in the `video` group; if the host's `video` group has a different GID, add
`--group-add "$(getent group video | cut -d: -f3)"`.

**Production Dockerfile**
- Multi-stage build reduces image size
- Non-root user (`appuser`) for security
- ONNX Runtime 1.17.0 with CPU support
- Ubuntu 22.04 base with minimal OpenCV runtime dependencies

### Jetson (JetPack 6, aarch64)

`Dockerfile` is x86-64 and CPU only. On a Jetson Orin use `Dockerfile.jetson`,
built on the board itself:

```bash
docker build -f Dockerfile.jetson -t video-analytics:jetson .

docker run --rm --runtime nvidia --device /dev/video0 \
  -v "$(pwd)/models:/app/models:ro" -p 9001:9001 video-analytics:jetson
```

- Base `dustynv/onnxruntime:1.20.2-r36.4.0` (~6 GB download): CUDA 12.6,
  TensorRT 10, ONNX Runtime 1.20.2 with the CUDA and TensorRT providers
- `--runtime nvidia` mounts the host GPU driver into the container
- Inference runs on the CPU unless `--provider cuda` or `--provider tensorrt`
  is passed

TensorRT (FP16) is the fastest option on a Jetson. The first start builds the
engine, which takes a few minutes, and caches it in `models/trt-cache/`, so
mount `models` writable (the container user has UID 1000):

```bash
docker run --rm --runtime nvidia --device /dev/video0 \
  -v "$(pwd)/models:/app/models" -p 9001:9001 video-analytics:jetson \
  --provider tensorrt
```

The engine is built from the FP32 model; `scripts/quantize-fp16.py` is not
needed with `--provider tensorrt`.

## Profiling

### ThreadSanitizer (data races)
```bash
cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON
cmake --build build-tsan --parallel $(nproc)
build-tsan/tests/test_pipeline
```

### AddressSanitizer (memory errors)
```bash
cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build-asan --parallel $(nproc)
build-asan/src/video_analytics --device 0
```

### Valgrind (memory leaks)
```bash
valgrind --leak-check=full --track-origins=yes build/src/video_analytics --benchmark --input test.avi --frames 50
```

### perf (CPU profiling)
```bash
perf record -g build/src/video_analytics --benchmark --input test.avi --frames 100
perf report
```

## Camera Troubleshooting

```bash
# List cameras
v4l2-ctl --list-devices

# Check supported formats
v4l2-ctl -d /dev/video0 --list-formats-ext

# Test camera
ffplay /dev/video0

# Permission denied? Add user to video group
sudo usermod -aG video $USER
```

## CI/CD

**GitHub Actions** (`.github/workflows/build.yml`)
- **Triggers:** Push to `main` or `feat/*` branches, pull requests to `main`
- **Jobs:**
  - `build` — Compile release build, run all tests on Ubuntu 22.04
  - `build-tsan` — ThreadSanitizer checks (data race detection)
- **Duration:** ~5 minutes per job

To skip CI on a commit:
```bash
git commit --no-verify  # NOT RECOMMENDED — use only for WIP branches
```

## FP16 Quantization (Optional)

Reduces model size to ~50% with minimal accuracy loss. Requires `onnxruntime-gpu` with CUDA provider for GPU-accelerated inference.

```bash
pip install onnx onnxconverter-common
python3 scripts/quantize-fp16.py models/yolov8n.onnx -o models/yolov8n_fp16.onnx

# Then use the quantized model
build/src/video_analytics --model models/yolov8n_fp16.onnx
```

**Mixed precision:** Resize and GridSample ops remain FP32 for stability; other ops convert to FP16.
