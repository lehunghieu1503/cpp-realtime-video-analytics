cmake_minimum_required(VERSION 3.20)
project(real_time_video_analytics LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler warnings — catch bugs early
# Only for C++: nvcc hands the host compiler generated code that -Wpedantic
# rejects line by line
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")

# Optional sanitizers for debugging
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable ASAN and TSAN simultaneously")
endif()
if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

# --- Dependencies ---

# nlohmann/json (header-only JSON library)
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(json)

# cpp-httplib (header-only HTTP server library)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.7
)
FetchContent_MakeAvailable(httplib)

# yaml-cpp (config file parser), linked statically
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

# ByteTrack (multi-object tracker). Only downloaded here: its own CMake file
# builds a shared library, and a static one built in src/ needs no install
# step. SOURCE_SUBDIR points at a directory without a CMakeLists.txt.
FetchContent_Declare(
    bytetrack
    GIT_REPOSITORY https://github.com/Vertical-Beach/ByteTrack-cpp.git
    GIT_TAG        d43805d461a714f65da039981bd5f5d21cf5cf59
    SOURCE_SUBDIR  data
)
FetchContent_MakeAvailable(bytetrack)
find_package(Eigen3 REQUIRED)

# OpenCV — installed via system package manager (apt install libopencv-dev)
find_package(OpenCV REQUIRED)

# ONNX Runtime — found via custom cmake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ONNXRuntime REQUIRED)

# OpenVINO backend (Intel CPU / iGPU / NPU). Off when OpenVINO is not
# installed, so the ONNX Runtime backends still build without it.
find_package(OpenVINO QUIET COMPONENTS Runtime)
option(WITH_OPENVINO "Build the OpenVINO backend" ${OpenVINO_FOUND})
if(WITH_OPENVINO AND NOT OpenVINO_FOUND)
    message(FATAL_ERROR "WITH_OPENVINO=ON but OpenVINO was not found; "
                        "set OpenVINO_DIR to <openvino>/runtime/cmake")
endif()
message(STATUS "OpenVINO backend: ${WITH_OPENVINO}")

# Native GStreamer capture (apt install libgstreamer1.0-dev
# libgstreamer-plugins-base1.0-dev). Without it GStreamer pipelines still run
# through OpenCV's wrapper.
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(GST IMPORTED_TARGET
        gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
endif()
option(WITH_GSTREAMER "Build the native GStreamer capture" ${GST_FOUND})
if(WITH_GSTREAMER AND NOT GST_FOUND)
    message(FATAL_ERROR "WITH_GSTREAMER=ON but the GStreamer development "
                        "files were not found")
endif()
message(STATUS "Native GStreamer capture: ${WITH_GSTREAMER}")

# Native TensorRT backend with GPU pre-processing. Needs the TensorRT
# headers and libraries and the CUDA toolkit, as on a Jetson or an NVIDIA
# workstation; without them the other backends still build.
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
          PATH_SUFFIXES aarch64-linux-gnu x86_64-linux-gnu)
find_library(NVINFER_LIBRARY NAMES nvinfer)
find_library(NVONNXPARSER_LIBRARY NAMES nvonnxparser)
include(CheckLanguage)
check_language(CUDA)
if(TENSORRT_INCLUDE_DIR AND NVINFER_LIBRARY AND NVONNXPARSER_LIBRARY
   AND CMAKE_CUDA_COMPILER)
    set(_tensorrt_found ON)
else()
    set(_tensorrt_found OFF)
endif()
option(WITH_TENSORRT "Build the native TensorRT backend" ${_tensorrt_found})
if(WITH_TENSORRT)
    if(NOT _tensorrt_found)
        message(FATAL_ERROR "WITH_TENSORRT=ON but TensorRT or the CUDA "
                            "compiler was not found")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
message(STATUS "Native TensorRT backend: ${WITH_TENSORRT}")

# Zero-copy frames on a Jetson: hardware buffers (NVMM) go from the decoder
# to the CUDA kernel without passing through system memory. Needs the two
# options above, nvbufsurface.h from NVIDIA's multimedia API and the EGL
# headers (apt install libegl-dev).
find_path(NVBUFSURFACE_INCLUDE_DIR NAMES nvbufsurface.h
          PATHS /usr/src/jetson_multimedia_api/include)
find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h)
if(WITH_GSTREAMER AND WITH_TENSORRT AND NVBUFSURFACE_INCLUDE_DIR
   AND EGL_INCLUDE_DIR)
    set(_nvmm_found ON)
else()
    set(_nvmm_found OFF)
endif()
option(WITH_NVMM "Read Jetson NVMM frames without copying" ${_nvmm_found})
if(WITH_NVMM AND NOT _nvmm_found)
    message(FATAL_ERROR "WITH_NVMM=ON needs WITH_GSTREAMER, WITH_TENSORRT, "
                        "nvbufsurface.h and the EGL headers")
endif()
message(STATUS "Zero-copy NVMM frames: ${WITH_NVMM}")

# MQTT event publishing (apt install libmosquitto-dev). Without it the
# application builds and runs, and a configured broker is reported as an
# error at startup.
find_library(MOSQUITTO_LIBRARY NAMES mosquitto)
find_path(MOSQUITTO_INCLUDE_DIR NAMES mosquitto.h)
if(MOSQUITTO_LIBRARY AND MOSQUITTO_INCLUDE_DIR)
    set(_mqtt_found ON)
else()
    set(_mqtt_found OFF)
endif()
option(WITH_MQTT "Publish events over MQTT" ${_mqtt_found})
if(WITH_MQTT AND NOT _mqtt_found)
    message(FATAL_ERROR "WITH_MQTT=ON but libmosquitto was not found")
endif()
message(STATUS "MQTT publishing: ${WITH_MQTT}")

# --- Subdirectories ---
add_subdirectory(src)

# Off in the Docker images: the tests are not shipped
option(BUILD_TESTS "Build the test executables" ON)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
