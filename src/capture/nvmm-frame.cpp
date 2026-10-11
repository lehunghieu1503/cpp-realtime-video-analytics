#include "nvmm-frame.hpp"

#include <gst/gst.h>
#include <nvbufsurface.h>
#include <opencv2/imgproc.hpp>

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace {
// libnvbufsurface.so belongs to the board's driver stack: the NVIDIA
// container runtime mounts it when the container starts, so it cannot be
// linked while the image is built. Resolve what is needed at first use.
struct SurfaceApi {
  int (*map)(NvBufSurface*, int, int, NvBufSurfaceMemMapFlags) = nullptr;
  int (*unmap)(NvBufSurface*, int, int) = nullptr;
  int (*syncForCpu)(NvBufSurface*, int, int) = nullptr;
  int (*mapEglImage)(NvBufSurface*, int) = nullptr;
  int (*unmapEglImage)(NvBufSurface*, int) = nullptr;

  SurfaceApi() {
    void* lib = dlopen("libnvbufsurface.so", RTLD_NOW | RTLD_GLOBAL);
    if (!lib) {
      throw std::runtime_error(std::string("cannot load libnvbufsurface.so: ") +
                               dlerror());
    }
    auto resolve = [lib](const char* name) {
      void* symbol = dlsym(lib, name);
      if (!symbol) {
        throw std::runtime_error(std::string("libnvbufsurface.so lacks ") +
                                 name);
      }
      return symbol;
    };
    map = reinterpret_cast<decltype(map)>(resolve("NvBufSurfaceMap"));
    unmap = reinterpret_cast<decltype(unmap)>(resolve("NvBufSurfaceUnMap"));
    syncForCpu = reinterpret_cast<decltype(syncForCpu)>(
        resolve("NvBufSurfaceSyncForCpu"));
    mapEglImage = reinterpret_cast<decltype(mapEglImage)>(
        resolve("NvBufSurfaceMapEglImage"));
    unmapEglImage = reinterpret_cast<decltype(unmapEglImage)>(
        resolve("NvBufSurfaceUnMapEglImage"));
  }
};

const SurfaceApi& api() {
  static SurfaceApi instance;
  return instance;
}
}  // namespace

NvmmFrame::NvmmFrame(GstSample* sample) : sample_(sample) {
  auto* map = new GstMapInfo;
  mapInfo_ = map;
  // For NVMM memory the mapped data is the NvBufSurface describing the
  // buffer, not pixels
  if (!gst_buffer_map(gst_sample_get_buffer(sample_), map, GST_MAP_READ)) {
    delete map;
    gst_sample_unref(sample_);
    throw std::runtime_error("cannot map the NVMM buffer");
  }
  surface_ = reinterpret_cast<NvBufSurface*>(map->data);
  bool usable = surface_ && surface_->numFilled >= 1 &&
                surface_->surfaceList[0].colorFormat ==
                    NVBUF_COLOR_FORMAT_RGBA;
  if (!usable) {
    gst_buffer_unmap(gst_sample_get_buffer(sample_), map);
    delete map;
    gst_sample_unref(sample_);
    throw std::runtime_error(
        "NVMM frames must be RGBA: end the pipeline with "
        "nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! appsink");
  }
  width_ = static_cast<int>(surface_->surfaceList[0].width);
  height_ = static_cast<int>(surface_->surfaceList[0].height);
}

NvmmFrame::~NvmmFrame() {
  unmapEglImage();
  auto* map = static_cast<GstMapInfo*>(mapInfo_);
  gst_buffer_unmap(gst_sample_get_buffer(sample_), map);
  delete map;
  gst_sample_unref(sample_);
}

cv::Mat NvmmFrame::toBgr() {
  if (api().map(surface_, 0, 0, NVBUF_MAP_READ) != 0) {
    throw std::runtime_error("cannot map the NVMM buffer for the CPU");
  }
  api().syncForCpu(surface_, 0, 0);
  const auto& params = surface_->surfaceList[0];
  cv::Mat rgba(height_, width_, CV_8UC4, params.mappedAddr.addr[0],
               params.pitch);
  cv::Mat bgr;
  cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  api().unmap(surface_, 0, 0);
  return bgr;
}

void* NvmmFrame::mapEglImage() {
  if (!eglMapped_) {
    if (api().mapEglImage(surface_, 0) != 0) {
      throw std::runtime_error("cannot map the NVMM buffer as an EGLImage");
    }
    eglMapped_ = true;
  }
  return surface_->surfaceList[0].mappedAddr.eglImage;
}

void NvmmFrame::unmapEglImage() {
  if (!eglMapped_) return;
  api().unmapEglImage(surface_, 0);
  eglMapped_ = false;
}
