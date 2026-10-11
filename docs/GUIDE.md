# Hướng dẫn build và sử dụng

Tài liệu này đi từng bước từ lúc clone repo đến lúc xem được kết quả nhận diện
trên dashboard, cho hai môi trường: PC x86-64 (chạy CPU) và Jetson Orin
(chạy GPU). Chi tiết dành cho người phát triển nằm ở
[DEVELOPMENT.md](DEVELOPMENT.md), kiến trúc ở [ARCHITECTURE.md](ARCHITECTURE.md).

Chọn đường đi:

| Bạn có | Làm theo |
|--------|----------|
| PC x86-64, muốn chạy nhanh nhất | Bước 1, rồi [2A](#2a-pc-x86-64-bằng-docker) |
| PC x86-64, muốn build từ source | Bước 1, rồi [2B](#2b-pc-x86-64-build-từ-source) |
| Jetson Orin (JetPack 6) | Bước 1, rồi [2C](#2c-jetson-orin-bằng-docker) |

Sau khi build xong, phần [3](#3-chạy-live-mode) trở đi dùng chung cho mọi môi
trường.

## 1. Lấy source và model

```bash
git clone https://github.com/lehunghieu1503/cpp-realtime-video-analytics.git
cd cpp-realtime-video-analytics
```

Model không nằm trong repo. Export YOLOv8n sang ONNX vào thư mục `models/`:

```bash
pip install ultralytics
scripts/download-model.sh          # tạo models/yolov8n.onnx
scripts/download-model.sh yolov8s  # hoặc model khác
```

Bước này cần internet và chỉ làm một lần. Nếu máy đích không có internet (ví
dụ Jetson trong mạng nội bộ), export trên máy khác rồi chép file `.onnx` sang.

## 2. Build

### 2A. PC x86-64 bằng Docker

```bash
docker build -t video-analytics .
```

Image này chỉ chạy CPU. Xong thì sang phần 3, dùng các lệnh `docker run`.

### 2B. PC x86-64 build từ source

Cần Ubuntu 22.04 trở lên, GCC 11+, CMake 3.20+.

1. Cài công cụ và OpenCV:

   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake git wget \
     libopencv-dev libeigen3-dev libmosquitto-dev
   ```

2. Cài ONNX Runtime 1.20.1 (bản CPU):

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
   tar xzf onnxruntime-linux-x64-1.20.1.tgz
   sudo cp -r onnxruntime-linux-x64-1.20.1/lib/* /usr/local/lib/
   sudo cp -r onnxruntime-linux-x64-1.20.1/include/* /usr/local/include/
   sudo ldconfig
   ```

   Không muốn cài vào `/usr/local` thì giải nén ở đâu cũng được và thêm
   `-DONNXRUNTIME_ROOT=/đường/dẫn/onnxruntime-linux-x64-1.20.1` vào lệnh
   `cmake` ở bước 3.

3. Build. Luôn truyền `Release`: không có thì CMake build không tối ưu và chậm
   hơn nhiều.

   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel $(nproc)
   ```

4. Chạy test để chắc bản build đúng:

   ```bash
   build/tests/test_pipeline
   build/tests/test_serializer
   build/tests/test_latency_tracker
   build/tests/test_sse_server
   build/tests/test_inference
   ```

   Mỗi test in `Failed: 0` ở cuối khi qua. `test_inference` in một dòng
   `Failed to load model: ... /nonexistent/path/model.onnx`: đó là ca kiểm tra
   đường dẫn sai, không phải lỗi.

### 2C. Jetson Orin bằng Docker

Yêu cầu: JetPack 6 (L4T r36.x), Docker có runtime `nvidia` (kiểm tra bằng
`docker info | grep -i runtimes`), user thuộc group `docker`.

`Dockerfile.jetson` dựa trên `dustynv/onnxruntime:1.20.2-r36.4.0`, đã có sẵn
CUDA 12.6, TensorRT 10 và ONNX Runtime 1.20.2 kèm provider CUDA và TensorRT.

**Cách 1: board có internet.** Build ngay trên board (tải base khoảng 6 GB):

```bash
docker build -f Dockerfile.jetson -t video-analytics:jetson .
```

**Cách 2 (khuyên dùng): build trên PC, board chỉ chạy.** Source, script và
config nằm trên PC; board không cần internet và không build gì. PC cần Docker
buildx, QEMU (`sudo apt install qemu-user-static binfmt-support`), `rsync` và
Python 3.

```bash
# Trên PC, trong thư mục repo
scripts/download-model.sh yolo11s      # model ghi trong deploy/config/config.yaml
scripts/deploy-jetson.sh USER@BOARD    # build, chuyển, khởi động
```

Script làm lần lượt:

1. Build `Dockerfile.jetson` cho arm64 bằng `docker buildx`.
2. Tải image arm64 của Mosquitto, Prometheus và Grafana.
3. Gửi từng image qua SSH, bỏ những layer board đã có. Lần đầu gửi cả image
   (khoảng 11 GB sau khi load); các lần sau, đổi code chỉ gửi vài MB.
4. Chép thư mục `deploy/` (compose, config) và model sang board.
5. Chạy `docker compose up -d` trên board.

Tham số thứ hai là thư mục trên board (mặc định `~/video-analytics`). Biến
`VIDEO_DIR=/đường/dẫn/trên/board` mount một thư mục video thử vào
`/data/videos`. Muốn đổi camera, zone, line hay model: sửa
`deploy/config/config.yaml` rồi chạy lại script. Chi tiết ở
[phần 7](#7-nhiều-camera-tracking-và-sự-kiện) và
[phần 8](#8-stack-triển-khai-mqtt-prometheus-grafana).

Trên board nên bật chế độ công suất cao nhất trước khi đo hay chạy thật:

```bash
sudo nvpmodel -m 2      # MAXN SUPER trên JetPack 6.2; xem mức hiện tại: nvpmodel -q
sudo jetson_clocks
```

## 3. Chạy live mode

Live mode đọc camera, nhận diện, và phát kết quả lên dashboard web.

### Build từ source

Chạy từ thư mục gốc của repo, vì dashboard được đọc từ thư mục `web/` tương
đối với thư mục hiện tại.

```bash
# Camera USB số 0, có cửa sổ hiển thị, dashboard chỉ mở cho máy này
build/src/video_analytics --device 0

# Không có màn hình (SSH, server): chỉ dashboard
build/src/video_analytics --device 0 --headless

# Cho máy khác trong mạng xem dashboard
build/src/video_analytics --device 0 --headless --bind 0.0.0.0
```

Nhấn `q` trong cửa sổ hoặc `Ctrl+C` để dừng.

### Docker trên PC

```bash
docker run --rm --device /dev/video0 -v "$(pwd)/models:/app/models:ro" \
  -p 127.0.0.1:9001:9001 video-analytics
```

### Docker trên Jetson

```bash
docker run -d --name video-analytics --restart unless-stopped --runtime nvidia \
  --device /dev/video0 -p 9001:9001 \
  -v ~/video-analytics/models:/app/models \
  video-analytics:jetson --backend tensorrt
```

- `--runtime nvidia` đưa driver GPU của board vào container.
- Thư mục `models` phải mount cho ghi (không có `:ro`) khi dùng
  `--backend tensorrt`, vì engine được cache vào `models/trt-cache/`.
- Lần chạy đầu với `tensorrt` mất khoảng 8 phút để build engine và chưa có hình
  trong lúc đó. Các lần sau khởi động trong vài giây. Theo dõi bằng
  `docker logs -f video-analytics`.
- Dừng và xoá: `docker rm -f video-analytics`.

Container luôn chạy với `--headless --bind 0.0.0.0`. Dashboard không có xác
thực: ai vào được cổng 9001 của máy đều xem được camera. Chỉ publish cổng ra
mạng mà bạn tin tưởng, hoặc dùng `-p 127.0.0.1:9001:9001`.

### Xem kết quả

Mở `http://ĐỊA_CHỈ_MÁY:9001` trong trình duyệt. Các đường dẫn của server:

| Đường dẫn | Nội dung |
|-----------|----------|
| `/` | Dashboard: mỗi nguồn một ô, bộ đếm zone/line, nhật ký sự kiện |
| `/video?stream=N` | Luồng MJPEG của nguồn thứ N, có vẽ khung, track id, zone và line |
| `/events` | Server-Sent Events, mỗi frame của mỗi nguồn một JSON |
| `/streams` | Danh sách nguồn kèm tên zone và line |
| `/metrics` | Số đo dạng Prometheus: FPS, độ trễ, sự kiện, công suất, bộ nhớ |
| `/health` | `200 {"status":"ok"}` khi kết quả đang ra đều, `503` khi không |

Mỗi sự kiện ở `/events` có dạng:

```json
{"frame_id":42,"detections":[{"class":"person","confidence":0.94,
 "bbox":{"x":100,"y":50,"w":200,"h":300}}],"latency_ms":35.2,"fps":21.5}
```

`latency_ms` là thời gian từ lúc bắt frame đến lúc nhận diện xong.

## 4. Các tuỳ chọn dòng lệnh

| Cờ | Ý nghĩa | Mặc định |
|----|---------|----------|
| `--config FILE` | File cấu hình YAML (xem phần 7) | không |
| `--device N` | Chỉ số camera V4L2 (USB) | `0` |
| `--rtsp URL` | Luồng RTSP hoặc HTTP, giải mã bằng FFmpeg | không |
| `--gst PIPELINE` | Pipeline GStreamer kết thúc bằng `appsink` | không |
| `--streams N` | Mở nguồn trên dòng lệnh N lần (đo số camera tối đa) | `1` |
| `--model PATH` | Model ONNX, hoặc OpenVINO IR (`.xml`) cho backend OpenVINO | `models/yolov8n.onnx` |
| `--backend NAME` | `cpu`, `cuda`, `tensorrt`, `openvino:CPU`, `openvino:GPU`, `openvino:NPU` | `cpu` |
| `--precision P` | Chỉ cho `tensorrt`: `fp32`, `fp16`, `int8` | `fp16` |
| `--conf X` | Ngưỡng confidence | `0.5` |
| `--workers N` | Số luồng inference, 1 đến 16 | `2`, hoặc `1` khi dùng bộ tăng tốc |
| `--port N` | Cổng dashboard | `9001` |
| `--bind ADDR` | Địa chỉ lắng nghe | `127.0.0.1` |
| `--headless` | Không mở cửa sổ, chỉ dashboard | tắt |
| `--benchmark` | Chạy benchmark thay cho live mode | tắt |
| `--input FILE` | Video đầu vào cho benchmark (bắt buộc) | không |
| `--frames N` | Số frame được đo | `300` |
| `--warmup N` | Số frame đầu bỏ qua, không đo | `30` |
| `--output FILE` | Ghi báo cáo benchmark ra file Markdown | không |
| `--csv FILE` | Ghi thêm một dòng kết quả vào file CSV | không |
| `--eval-coco DIR` | Chạy detector trên một thư mục ảnh `.jpg` để đo mAP | không |
| `--eval-out FILE` | Nơi ghi kết quả của `--eval-coco` | `detections.json` |

`--provider` vẫn dùng được, là tên cũ của `--backend`. Cờ dòng lệnh ghi đè
giá trị trong file config.

Chọn backend:

- `cpu`: chạy ở mọi nơi. Số thread ONNX Runtime tự giới hạn theo số nhân CPU.
- `cuda`: cần ONNX Runtime bản GPU. Không cần bước chuẩn bị nào.
- `tensorrt`: nhanh nhất trên Jetson. Build engine ở lần chạy đầu với mỗi
  model và mỗi precision.
- `openvino:CPU`, `openvino:GPU`, `openvino:NPU`: CPU, iGPU và NPU của Intel.
  Cần build với `-DOpenVINO_DIR=<openvino>/runtime/cmake`.

Mọi backend dùng chung pre-process, post-process và NMS; backend chỉ chạy
tensor. Nếu backend được yêu cầu không có, chương trình báo
`Backend '...' failed to load ...` và thoát, chứ không âm thầm chạy CPU.

## 5. Đo tốc độ bằng benchmark

Benchmark chạy qua một file video, xử lý mọi frame, và in FPS cùng phân bố độ
trễ của riêng bước nhận diện.

```bash
# Build từ source
build/src/video_analytics --benchmark --input video.avi --frames 300
build/src/video_analytics --benchmark --input video.avi --backend tensorrt --output report.md

# Docker trên Jetson
docker run --rm --runtime nvidia \
  -v ~/video-analytics/models:/app/models -v ~/video.avi:/data/video.avi:ro \
  video-analytics:jetson --benchmark --input /data/video.avi --backend tensorrt
```

Báo cáo gồm FPS, độ trễ p50/p95/p99, thời gian từng khâu (decode,
pre-process, inference, post-process), công suất và FPS/W. Công suất trên
Jetson là cả board (cảm biến INA3221). Trên x86 là gói CPU đo qua RAPL, cộng
GPU NVIDIA khi backend chạy trên nó; RAPL chỉ root đọc được, mở bằng
`sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj`.

So sánh nhiều backend và đo mAP:

```bash
scripts/coco-subset.sh 500     # tải 500 ảnh COCO val2017 vào data/coco/
pip install pycocotools
scripts/benchmark.sh video.avi cpu openvino:NPU tensorrt@fp16 tensorrt@int8
```

INT8:

```bash
# TensorRT: sinh bảng calibration, deploy-jetson.sh tự chép sang board
scripts/calibrate-int8.py models/yolo11s.onnx data/coco/val2017
# rồi đặt  precision: int8  trong config, hoặc --precision int8

# OpenVINO: sinh model INT8 bằng NNCF, truyền qua --model
scripts/quantize-openvino.py models/yolo11s.onnx data/coco/val2017
```

Số đo trên Jetson Orin Nano Super 8GB, trong container, chế độ MAXN SUPER,
video 640x480, model YOLO11s 640x640, mAP trên 500 ảnh COCO val2017:

| Backend | Precision | FPS | Độ trễ p50 | Công suất | FPS/W | mAP50-95 |
|---------|-----------|-----|------------|-----------|-------|----------|
| `tensorrt` | INT8 | 107,4 | 9,5 ms | 12,3 W | 8,73 | 0,459 |
| `tensorrt` | FP16 | 68,7 | 14,6 ms | 12,6 W | 5,45 | 0,470 |
| `cuda` | FP32 | 31,9 | 31,4 ms | 15,8 W | 2,03 | 0,470 |
| `cpu` | FP32 | 3,6 | 556,6 ms | 10,8 W | 0,33 | 0,469 |

Video thử chỉ có 250 frame, nên mỗi lần đo 190 frame sau 60 frame warm-up:
đủ để lấy tốc độ, chưa đủ dài để thấy thermal throttling.

## 6. Camera CSI trên Jetson

`--device` chỉ mở được camera USB. Camera CSI (IMX219, IMX477...) cần pipeline
GStreamer qua `--gst`. Với bản build từ source trên board:

```bash
build/src/video_analytics --headless --backend tensorrt --gst \
  "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 \
   ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR \
   ! appsink drop=true max-buffers=1"
```

Pipeline nào cũng dùng được miễn là kết thúc bằng `appsink` và đưa ra frame
BGR. Ví dụ nguồn giả để thử khi chưa có camera:

```bash
--gst "videotestsrc is-live=true ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
```

Image Docker đã có `libEGL` và `libGLESv2` mà các plugin GStreamer của NVIDIA
cần. Decode H.264 bằng phần cứng (`nvv4l2decoder`) đã thử chạy được trong
container; pipeline camera CSI thì chưa thử với cảm biến thật.

## 7. Nhiều camera, tracking và sự kiện

Khai báo trong file YAML, truyền bằng `--config`. File mẫu có chú thích:
[deploy/config/config.yaml](../deploy/config/config.yaml).

```yaml
model: models/yolo11s.onnx
backend: tensorrt
classes: [person]            # chỉ giữ các class COCO này
dwell_seconds: 5

sources:
  - id: cong
    rtsp: rtsp://192.168.1.20:554/stream1
    zones:
      - name: cua
        points: [[300, 200], [660, 200], [660, 440], [300, 440]]
    lines:
      - name: vach
        a: [0, 300]
        b: [960, 300]
  - id: san
    device: 0                # hoặc  file: clip.mp4  hoặc  gst: "... ! appsink"

mqtt:
  host: mosquitto
  topic: video-analytics
snapshots:
  dir: snapshots
  keep: 200
```

- Mỗi nguồn có một luồng capture riêng. Frame được phục vụ xoay vòng giữa các
  nguồn, nên khi số camera vượt sức phần cứng thì mọi camera cùng chậm lại,
  không camera nào bị bỏ đói.
- ByteTrack gán `track_id` cho từng vật thể. Vị trí vật thể tính ở điểm giữa
  cạnh dưới của khung (chân người).
- Sự kiện: `line_cross` (kèm `direction` là `in` khi đi từ bên phải sang bên
  trái của hướng `a → b`, ngược lại là `out`), `zone_enter`, `zone_exit` (kèm
  `dwell_s`) và `dwell_time` (một lần khi ở trong zone quá `dwell_seconds`).
- Mỗi sự kiện được gửi lên MQTT tại `<topic>/<id nguồn>/<loại>` với QoS 1, và
  frame lúc đó được lưu vào thư mục snapshot (giữ `keep` ảnh mới nhất).
- Nguồn live (camera, RTSP, GStreamer) bị mất sẽ được mở lại, thời gian chờ
  tăng gấp đôi từ 1 đến 30 giây. File video thì chạy hết là dừng.

Xem sự kiện:

```bash
mosquitto_sub -h BOARD -t 'video-analytics/#' -v
```

Số camera tối đa đo trên Jetson Orin Nano Super (YOLO11s, mỗi camera là luồng
MJPEG 960x540 decode bằng phần mềm, có tracking):

| Cấu hình | Số camera ở 25 FPS | Số camera ở 15 FPS |
|----------|--------------------|--------------------|
| TensorRT FP16, 1 worker | 3 (13,8 W) | 4 (12,6 W) |
| TensorRT FP16, 2 worker | 4 (16,7 W) | 8 (18,6 W) |
| TensorRT INT8, 2 worker | 5 ở 24,4 FPS (14,3 W) | 10 ở 14,6 FPS (15,4 W) |

Khi chạy nhiều camera trên GPU nên đặt `workers: 2`: pre-process và
post-process khi đó chạy song song.

## 8. Stack triển khai: MQTT, Prometheus, Grafana

`deploy/docker-compose.yml` chạy bốn dịch vụ: ứng dụng, Mosquitto, Prometheus
và Grafana. `scripts/deploy-jetson.sh` đưa cả stack lên board.

| Dịch vụ | Địa chỉ |
|---------|---------|
| Dashboard | `http://BOARD:9001` |
| Grafana | `http://BOARD:3000` (xem không cần đăng nhập; sửa bằng `admin`/`admin`) |
| Prometheus | `http://BOARD:9090` |
| MQTT | `BOARD:1883` |

Không dịch vụ nào có xác thực. Chỉ mở các cổng này trong mạng bạn tin tưởng.

Thao tác trên board, trong thư mục triển khai:

```bash
docker compose ps
docker compose logs -f app
docker compose restart app
docker compose down
```

Chạy liên tục:

- Dịch vụ có `restart: unless-stopped`, nên tự chạy lại sau khi board khởi
  động lại hoặc khi ứng dụng thoát.
- Watchdog: nếu có nguồn đang kết nối mà 60 giây không ra kết quả, ứng dụng tự
  thoát để Docker khởi động lại (`watchdog_seconds` trong config, `0` để tắt).
- Soak test là để stack chạy, rồi lấy tóm tắt từ Prometheus (tăng bộ nhớ, FPS
  thấp nhất, số lần mất kết nối, số lần khởi động lại):

  ```bash
  scripts/soak-report.sh BOARD 24     # 24 giờ gần nhất
  ```

## 9. Lỗi thường gặp

| Triệu chứng | Nguyên nhân và cách xử lý |
|-------------|---------------------------|
| `Backend '...' failed to load ... File doesn't exist` | Chưa có file model. Làm lại bước 1, hoặc kiểm tra `-v ...:/app/models` khi chạy Docker. |
| `Pipeline: cam0 not available, retry in N s` | Nguồn chưa mở được; chương trình thử lại sau 1 đến 30 giây. Với camera USB: không có `/dev/video0`, hoặc thiếu `--device /dev/video0` khi chạy Docker. Liệt kê camera bằng `ls /dev/video*`. Trong Docker, nếu GID của group `video` trên máy khác trong container, thêm `--group-add "$(getent group video \| cut -d: -f3)"`. |
| `Backend 'cuda' failed to load ... libonnxruntime_providers_cuda.so` | ONNX Runtime đang dùng là bản CPU. Trên Jetson hãy dùng image `video-analytics:jetson`; image `video-analytics` (x86) chỉ có CPU. |
| `TensorRT EP could not deserialize engine from cache` | Engine trong `models/trt-cache/` được build bằng phiên bản TensorRT khác (ví dụ đổi giữa chạy native và chạy Docker). Xoá `models/trt-cache/` rồi chạy lại. |
| Chạy `tensorrt` lần đầu đứng yên vài phút | Đang build engine. Chờ, lần sau sẽ nhanh. |
| `permission denied ... docker.sock` | User chưa thuộc group `docker`: `sudo usermod -aG docker $USER` rồi đăng nhập lại. |
| `SSE server: cannot bind 127.0.0.1:9001` | Cổng đang bị chiếm. Đổi bằng `--port`. |
| Mở dashboard báo `Not found` | Chương trình không được chạy từ thư mục gốc repo, nên không thấy `web/`. |
| Dashboard không vào được từ máy khác | Mặc định chỉ lắng nghe `127.0.0.1`. Thêm `--bind 0.0.0.0` (Docker đã bật sẵn) và kiểm tra `-p 9001:9001`. |
| CMake báo `Could not resolve host: github.com` | Máy không có internet để tải `nlohmann/json`, `cpp-httplib`, `yaml-cpp` và `ByteTrack-cpp`. Chép các thư mục source từ `build/_deps/` của máy khác sang, rồi thêm `-DFETCHCONTENT_SOURCE_DIR_JSON=... -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=... -DFETCHCONTENT_SOURCE_DIR_YAML-CPP=... -DFETCHCONTENT_SOURCE_DIR_BYTETRACK=...` vào lệnh `cmake`. |
| `Watchdog: no result for 60 s, exiting` | Có nguồn đang kết nối nhưng inference không ra kết quả. Chương trình tự thoát để Docker khởi động lại; xem các dòng log ngay trước đó. |
| `MQTT: this build has no MQTT support` | Lúc build thiếu `libmosquitto-dev`. |
