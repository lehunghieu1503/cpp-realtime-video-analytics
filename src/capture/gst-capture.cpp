#include "gst-capture.hpp"
#ifdef WITH_NVMM
#include "nvmm-frame.hpp"
#endif

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <iostream>
#include <mutex>
#include <sstream>

namespace {
// How long read() waits for a frame before reporting a failure. The pipeline
// declares a source lost after 30 failures in a row, so about 6 s.
const GstClockTime kPullTimeout = 200 * GST_MSECOND;

GstElement* findAppSink(GstElement* pipeline) {
  GstElement* found = nullptr;
  GstIterator* it = gst_bin_iterate_sinks(GST_BIN(pipeline));
  GValue item = G_VALUE_INIT;
  while (!found && gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
    auto* element = GST_ELEMENT(g_value_get_object(&item));
    if (GST_IS_APP_SINK(element)) {
      found = GST_ELEMENT(gst_object_ref(element));
    }
    g_value_reset(&item);
  }
  g_value_unset(&item);
  gst_iterator_free(it);
  return found;
}
}  // namespace

bool GstCapture::open(const CaptureConfig& config) {
  release();
  config_ = config;

  static std::once_flag initialised;
  std::call_once(initialised, [] { gst_init(nullptr, nullptr); });

  GError* error = nullptr;
  pipeline_ = gst_parse_launch(config.gstPipeline.c_str(), &error);
  if (!pipeline_ || error) {
    std::cerr << "GStreamer: cannot build pipeline: "
              << (error ? error->message : "unknown error") << "\n";
    g_clear_error(&error);
    release();
    return false;
  }
  sink_ = findAppSink(pipeline_);
  if (!sink_) {
    std::cerr << "GStreamer: the pipeline has no appsink\n";
    release();
    return false;
  }

  // Wait for the first data to reach the sink (or for the failure that
  // keeps it from arriving), so that "opened" means frames can be read
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  GstStateChangeReturn result =
      gst_element_get_state(pipeline_, nullptr, nullptr, 10 * GST_SECOND);
  // The bus says which element failed and why; ask it in either case
  bool busError = reportBusError();
  if (result == GST_STATE_CHANGE_FAILURE || busError) {
    if (!busError) std::cerr << "GStreamer: the pipeline did not start\n";
    release();
    return false;
  }
  return true;
}

bool GstCapture::reportBusError() {
  GstBus* bus = gst_element_get_bus(pipeline_);
  GstMessage* message = gst_bus_pop_filtered(bus, GST_MESSAGE_ERROR);
  gst_object_unref(bus);
  if (!message) return false;

  GError* error = nullptr;
  gchar* debug = nullptr;
  gst_message_parse_error(message, &error, &debug);
  std::cerr << "GStreamer: " << GST_OBJECT_NAME(message->src) << ": "
            << error->message << "\n";
  g_clear_error(&error);
  g_free(debug);
  gst_message_unref(message);
  return true;
}

bool GstCapture::read(cv::Mat& frame) {
  if (!pipeline_) return false;

  GstSample* sample =
      gst_app_sink_try_pull_sample(GST_APP_SINK(sink_), kPullTimeout);
  if (!sample) {
    // Nothing within the timeout: the stream ended, failed, or is stalled
    reportBusError();
    return false;
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  if (gst_caps_features_contains(gst_caps_get_features(caps, 0),
                                 "memory:NVMM")) {
#ifdef WITH_NVMM
    try {
      // The frame keeps the sample, and with it the hardware buffer
      auto nvmm = std::make_shared<NvmmFrame>(sample);
      width_ = nvmm->width();
      height_ = nvmm->height();
      format_ = "RGBA (NVMM)";
      deviceFrame_ = std::move(nvmm);
      frame.release();
      return true;
    } catch (const std::exception& e) {
      std::cerr << "GStreamer: " << e.what() << "\n";
      return false;
    }
#else
    std::cerr << "GStreamer: this build cannot read NVMM memory; add "
                 "\"nvvidconv ! video/x-raw,format=BGRx\" before the appsink\n";
    gst_sample_unref(sample);
    return false;
#endif
  }

  GstVideoInfo info;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  bool ok = gst_video_info_from_caps(&info, caps) &&
            gst_buffer_map(buffer, &map, GST_MAP_READ);
  if (!ok) {
    std::cerr << "GStreamer: cannot read the sample\n";
    gst_sample_unref(sample);
    return false;
  }

  width_ = GST_VIDEO_INFO_WIDTH(&info);
  height_ = GST_VIDEO_INFO_HEIGHT(&info);
  if (GST_VIDEO_INFO_FPS_D(&info) > 0) {
    fps_ = static_cast<double>(GST_VIDEO_INFO_FPS_N(&info)) /
           GST_VIDEO_INFO_FPS_D(&info);
  }
  format_ = gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&info));
  size_t stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);

  // The buffer goes back to its pool when the sample is released, so the
  // pixels are copied out here: once, and converted in the same pass when
  // they are not BGR already
  switch (GST_VIDEO_INFO_FORMAT(&info)) {
    case GST_VIDEO_FORMAT_BGR:
      cv::Mat(height_, width_, CV_8UC3, map.data, stride).copyTo(frame);
      break;
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_BGRA:
      cv::cvtColor(cv::Mat(height_, width_, CV_8UC4, map.data, stride), frame,
                   cv::COLOR_BGRA2BGR);
      break;
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_RGBA:
      cv::cvtColor(cv::Mat(height_, width_, CV_8UC4, map.data, stride), frame,
                   cv::COLOR_RGBA2BGR);
      break;
    default:
      std::cerr << "GStreamer: appsink delivers " << format_
                << "; end the pipeline with video/x-raw,format=BGR\n";
      ok = false;
  }

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);
  return ok;
}

void GstCapture::release() {
  if (sink_) {
    gst_object_unref(sink_);
    sink_ = nullptr;
  }
  if (pipeline_) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
}

std::string GstCapture::getInfo() const {
  std::ostringstream ss;
  ss << config_.id << ": GStreamer";
  if (width_ > 0) {
    ss << " " << width_ << "x" << height_ << " @ " << fps_ << " FPS format="
       << format_;
  }
  return ss.str();
}
