// A frame's source timestamp must reach the application exactly as the
// sender wrote it: a simulator's frame is identified by it.
//
// The stream is the one a simulation uses: JPEG frames in GStreamer's GDP
// framing, decoded by jpegdec. It is written to a file here and read back
// through GstCapture, so no network is involved.

#include "capture/video-capture.hpp"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>

#ifdef WITH_GSTREAMER
#include "capture/gst-capture.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#endif

static int testsRun = 0;
static int testsFailed = 0;

#define CHECK(condition)                                              \
  do {                                                                \
    testsRun++;                                                       \
    if (!(condition)) {                                               \
      testsFailed++;                                                  \
      std::cerr << "FAIL line " << __LINE__ << ": " #condition "\n";  \
    }                                                                 \
  } while (0)

#ifdef WITH_GSTREAMER
namespace {
const int kWidth = 64;
const int kHeight = 48;

// The frame number is in the pixels, to show a timestamp stays with its frame
cv::Mat frameFor(int index) {
  cv::Mat image(kHeight, kWidth, CV_8UC3);
  image(cv::Rect(0, 0, kWidth / 2, kHeight)) = cv::Scalar::all(index % 200 + 20);
  image(cv::Rect(kWidth / 2, 0, kWidth / 2, kHeight)) =
      cv::Scalar::all(index / 200 * 60 + 20);
  return image;
}

int indexOf(const cv::Mat& frame) {
  double low = cv::mean(frame(cv::Rect(8, 8, 16, 32)))[0] - 20;
  double high = cv::mean(frame(cv::Rect(40, 8, 16, 32)))[0] - 20;
  return static_cast<int>(std::lround(high / 60)) * 200 +
         static_cast<int>(std::lround(low));
}

bool haveElements() {
  for (const char* name : {"gdppay", "gdpdepay", "jpegdec", "filesink"}) {
    GstElementFactory* factory = gst_element_factory_find(name);
    if (!factory) {
      std::cout << "SKIP: GStreamer element " << name << " is not installed\n";
      return false;
    }
    gst_object_unref(factory);
  }
  return true;
}

// Writes the frames as JPEG in GDP framing, each with its timestamp
bool writeStream(const std::string& path, const std::vector<int>& indices,
                 const std::vector<uint64_t>& timestamps) {
  std::string description =
      "appsrc name=src format=time caps=image/jpeg,framerate=30/1 ! gdppay ! "
      "filesink location=" + path;
  GstElement* pipeline = gst_parse_launch(description.c_str(), nullptr);
  if (!pipeline) return false;
  GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "src");
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  for (size_t i = 0; i < indices.size(); i++) {
    std::vector<uchar> jpeg;
    cv::imencode(".jpg", frameFor(indices[i]), jpeg,
                 {cv::IMWRITE_JPEG_QUALITY, 95});
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, jpeg.size(), nullptr);
    gst_buffer_fill(buffer, 0, jpeg.data(), jpeg.size());
    GST_BUFFER_PTS(buffer) = timestamps[i];
    gst_app_src_push_buffer(GST_APP_SRC(source), buffer);
  }
  gst_app_src_end_of_stream(GST_APP_SRC(source));

  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* message = gst_bus_timed_pop_filtered(
      bus, 20 * GST_SECOND,
      static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  bool ok = message && GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS;
  if (message) gst_message_unref(message);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(source);
  gst_object_unref(pipeline);
  return ok;
}

void testGdpJpegKeepsTimestamps() {
  gst_init(nullptr, nullptr);
  if (!haveElements()) return;

  // 300 frames at a simulator's 1/30 s, which is not a whole number of
  // nanoseconds, starting at 0: a valid timestamp, not a missing one. Every
  // seventh frame is dropped before sending, as a full queue would.
  std::vector<int> indices;
  std::vector<uint64_t> timestamps;
  for (int i = 0; i < 360; i++) {
    if (i % 7 == 3) continue;
    indices.push_back(i);
    timestamps.push_back(static_cast<uint64_t>(std::llround(i * 1e9 / 30)));
  }
  // Past 2^53: would not survive a pass through a double
  indices.push_back(360);
  timestamps.push_back(9007199254740993ULL);
  CHECK(indices.size() >= 300);

  std::string path = (std::filesystem::temp_directory_path() / "va-test-timestamps.gdp").string();
  CHECK(writeStream(path, indices, timestamps));

  GstCapture capture;
  CaptureConfig config;
  config.gstPipeline = "filesrc location=" + path +
                       " ! gdpdepay ! jpegdec ! videoconvert ! "
                       "video/x-raw,format=BGR ! appsink sync=false";
  CHECK(!capture.sourceTimestampNs());
  CHECK(capture.open(config));

  size_t received = 0, wrongTime = 0, wrongFrame = 0, missing = 0;
  cv::Mat frame;
  while (received < indices.size() && capture.read(frame)) {
    auto pts = capture.sourceTimestampNs();
    if (!pts) {
      missing++;
    } else if (*pts != timestamps[received]) {
      wrongTime++;
    }
    if (frame.cols != kWidth || indexOf(frame) != indices[received]) {
      wrongFrame++;
    }
    received++;
  }
  std::cout << "GDP/JPEG: " << received << " of " << indices.size()
            << " frames, " << missing << " without timestamp, " << wrongTime
            << " with a changed timestamp, " << wrongFrame
            << " with another frame's pixels\n";
  CHECK(received == indices.size());
  CHECK(missing == 0);
  CHECK(wrongTime == 0);
  CHECK(wrongFrame == 0);

  // The stream has ended: the failed read must not keep the last timestamp
  CHECK(!capture.read(frame));
  CHECK(!capture.sourceTimestampNs());

  capture.release();
  CHECK(!capture.sourceTimestampNs());
  std::remove(path.c_str());
}
}  // namespace
#endif

int main() {
  // A capture that has no source timestamp says so
  CaptureConfig file;
  file.videoPath = "does-not-exist.mp4";
  CHECK(!createCapture(file)->sourceTimestampNs());

#ifdef WITH_GSTREAMER
  testGdpJpegKeepsTimestamps();
#else
  std::cout << "SKIP: built without the native GStreamer capture\n";
#endif

  std::cout << testsRun - testsFailed << "/" << testsRun << " checks passed\n";
  return testsFailed == 0 ? 0 : 1;
}
