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
   sudo apt install -y build-essential cmake git wget libopencv-dev
   ```

2. Cài ONNX Runtime 1.17.0 (bản CPU):

   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
   tar xzf onnxruntime-linux-x64-1.17.0.tgz
   sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
   sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
   sudo ldconfig
   ```

   Không muốn cài vào `/usr/local` thì giải nén ở đâu cũng được và thêm
   `-DONNXRUNTIME_ROOT=/đường/dẫn/onnxruntime-linux-x64-1.17.0` vào lệnh
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

**Cách 2: board không có internet.** Build image arm64 trên PC rồi chuyển qua
SSH. PC cần Docker buildx và QEMU (`sudo apt install qemu-user-static
binfmt-support`).

```bash
# Trên PC, trong thư mục repo
docker build --platform linux/arm64 -f Dockerfile.jetson -t video-analytics:jetson .
docker save video-analytics:jetson | ssh USER@BOARD 'docker load'

# Chép model sang board
ssh USER@BOARD 'mkdir -p ~/video-analytics/models'
scp models/yolov8n.onnx USER@BOARD:~/video-analytics/models/
```

Build dưới giả lập mất khoảng 15 phút, chuyển image 6 GB mất thêm tuỳ tốc độ
mạng. Image sau khi load chiếm khoảng 11 GB trên board.

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
  video-analytics:jetson --provider tensorrt
```

- `--runtime nvidia` đưa driver GPU của board vào container.
- Thư mục `models` phải mount cho ghi (không có `:ro`) khi dùng
  `--provider tensorrt`, vì engine được cache vào `models/trt-cache/`.
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
| `/` | Dashboard |
| `/video` | Luồng MJPEG có vẽ khung nhận diện |
| `/events` | Server-Sent Events, mỗi frame một JSON |
| `/health` | `{"status":"ok"}` |

Mỗi sự kiện ở `/events` có dạng:

```json
{"frame_id":42,"detections":[{"class":"person","confidence":0.94,
 "bbox":{"x":100,"y":50,"w":200,"h":300}}],"latency_ms":35.2,"fps":21.5}
```

`latency_ms` là thời gian từ lúc bắt frame đến lúc nhận diện xong.

## 4. Các tuỳ chọn dòng lệnh

| Cờ | Ý nghĩa | Mặc định |
|----|---------|----------|
| `--device N` | Chỉ số camera V4L2 (USB) | `0` |
| `--gst PIPELINE` | Pipeline GStreamer kết thúc bằng `appsink`, thay cho `--device` | không |
| `--model PATH` | File model ONNX | `models/yolov8n.onnx` |
| `--provider NAME` | Nơi chạy inference: `cpu`, `cuda`, `tensorrt` | `cpu` |
| `--workers N` | Số luồng inference, 1 đến 16 | `2`, hoặc `1` khi dùng GPU |
| `--port N` | Cổng dashboard | `9001` |
| `--bind ADDR` | Địa chỉ lắng nghe | `127.0.0.1` |
| `--headless` | Không mở cửa sổ, chỉ dashboard | tắt |
| `--benchmark` | Chạy benchmark thay cho live mode | tắt |
| `--input FILE` | Video đầu vào cho benchmark (bắt buộc) | không |
| `--frames N` | Số frame benchmark | `300` |
| `--output FILE` | Ghi báo cáo benchmark ra file Markdown | không |

Chọn provider:

- `cpu`: chạy ở mọi nơi. Số thread ONNX Runtime tự giới hạn theo số nhân CPU.
- `cuda`: cần ONNX Runtime bản GPU. Không cần bước chuẩn bị nào.
- `tensorrt`: nhanh nhất trên Jetson (FP16). Build engine ở lần chạy đầu.

Nếu provider được yêu cầu không có trong bản ONNX Runtime đang dùng, chương
trình báo `Execution provider '...' not available` và thoát, chứ không âm thầm
chạy CPU.

## 5. Đo tốc độ bằng benchmark

Benchmark chạy qua một file video, xử lý mọi frame, và in FPS cùng phân bố độ
trễ của riêng bước nhận diện.

```bash
# Build từ source
build/src/video_analytics --benchmark --input video.avi --frames 300
build/src/video_analytics --benchmark --input video.avi --provider tensorrt --output report.md

# Docker trên Jetson
docker run --rm --runtime nvidia \
  -v ~/video-analytics/models:/app/models -v ~/video.avi:/data/video.avi:ro \
  video-analytics:jetson --benchmark --input /data/video.avi --provider tensorrt
```

Số đo tham khảo trên Jetson Orin Nano Super 8GB, trong container, chế độ 25W,
video 640x480, model YOLOv8n 640x640:

| Provider | FPS | Độ trễ p50 |
|----------|-----|------------|
| `cpu` | 6,4 | 314 ms |
| `cuda` | 29,4 | 28 ms |
| `tensorrt` | 46,0 | 21,6 ms |

## 6. Camera CSI trên Jetson

`--device` chỉ mở được camera USB. Camera CSI (IMX219, IMX477...) cần pipeline
GStreamer qua `--gst`. Với bản build từ source trên board:

```bash
build/src/video_analytics --headless --provider tensorrt --gst \
  "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 \
   ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR \
   ! appsink drop=true max-buffers=1"
```

Pipeline nào cũng dùng được miễn là kết thúc bằng `appsink` và đưa ra frame
BGR. Ví dụ nguồn giả để thử khi chưa có camera:

```bash
--gst "videotestsrc is-live=true ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
```

Giới hạn hiện tại: trong image Docker, plugin `nvarguscamerasrc` không load
được vì thiếu `libEGL.so.1`, nên camera CSI chưa chạy được qua Docker. Camera
USB và `videotestsrc` chạy bình thường trong container.

## 7. Lỗi thường gặp

| Triệu chứng | Nguyên nhân và cách xử lý |
|-------------|---------------------------|
| `Failed to load model: ... File doesn't exist` | Chưa có `models/yolov8n.onnx`. Làm lại bước 1, hoặc kiểm tra `-v ...:/app/models` khi chạy Docker. |
| `Failed to open camera 0` | Không có `/dev/video0`, hoặc thiếu `--device /dev/video0` khi chạy Docker. Liệt kê camera bằng `ls /dev/video*`. Trong Docker, nếu GID của group `video` trên máy khác trong container, thêm `--group-add "$(getent group video \| cut -d: -f3)"`. |
| `Execution provider 'cuda' not available` | ONNX Runtime đang dùng là bản CPU. Trên Jetson hãy dùng image `video-analytics:jetson`; image `video-analytics` (x86) chỉ có CPU. |
| `TensorRT EP could not deserialize engine from cache` | Engine trong `models/trt-cache/` được build bằng phiên bản TensorRT khác (ví dụ đổi giữa chạy native và chạy Docker). Xoá `models/trt-cache/` rồi chạy lại. |
| Chạy `tensorrt` lần đầu đứng yên vài phút | Đang build engine. Chờ, lần sau sẽ nhanh. |
| `permission denied ... docker.sock` | User chưa thuộc group `docker`: `sudo usermod -aG docker $USER` rồi đăng nhập lại. |
| `SSE server: cannot bind 127.0.0.1:9001` | Cổng đang bị chiếm. Đổi bằng `--port`. |
| Mở dashboard báo `Not found` | Chương trình không được chạy từ thư mục gốc repo, nên không thấy `web/`. |
| Dashboard không vào được từ máy khác | Mặc định chỉ lắng nghe `127.0.0.1`. Thêm `--bind 0.0.0.0` (Docker đã bật sẵn) và kiểm tra `-p 9001:9001`. |
| CMake báo `Could not resolve host: github.com` | Máy không có internet để tải `nlohmann/json` và `cpp-httplib`. Chép hai thư mục source từ `build/_deps/` của máy khác sang, rồi thêm `-DFETCHCONTENT_SOURCE_DIR_JSON=... -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=...` vào lệnh `cmake`. |
