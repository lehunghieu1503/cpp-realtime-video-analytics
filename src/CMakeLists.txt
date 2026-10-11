# Third-party tracker, built without our warning flags
add_library(bytetrack STATIC
    ${bytetrack_SOURCE_DIR}/src/BYTETracker.cpp
    ${bytetrack_SOURCE_DIR}/src/KalmanFilter.cpp
    ${bytetrack_SOURCE_DIR}/src/lapjv.cpp
    ${bytetrack_SOURCE_DIR}/src/Object.cpp
    ${bytetrack_SOURCE_DIR}/src/Rect.cpp
    ${bytetrack_SOURCE_DIR}/src/STrack.cpp
)
target_include_directories(bytetrack SYSTEM PUBLIC ${bytetrack_SOURCE_DIR}/include)
target_link_libraries(bytetrack PUBLIC Eigen3::Eigen)
target_compile_options(bytetrack PRIVATE -w)

# Everything except main(), shared by the application and the tests
add_library(va_core STATIC
    analytics/rules.cpp
    analytics/tracker.cpp
    capture/opencv-capture.cpp
    config/app-config.cpp
    output/mqtt-publisher.cpp
    inference/inference-engine.cpp
    inference/pre-process.cpp
    inference/post-process.cpp
    inference/backends/backend-factory.cpp
    inference/backends/onnx-backend.cpp
    pipeline/pipeline.cpp
    server/sse-server.cpp
)

# src/ as root so includes like "capture/xxx.hpp" work
target_include_directories(va_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(va_core
    PUBLIC
        ${OpenCV_LIBS}
        ONNXRuntime::ONNXRuntime
        nlohmann_json::nlohmann_json
        httplib::httplib
        pthread
    PRIVATE
        bytetrack
        yaml-cpp::yaml-cpp
)

if(WITH_OPENVINO)
    target_sources(va_core PRIVATE inference/backends/openvino-backend.cpp)
    target_compile_definitions(va_core PUBLIC WITH_OPENVINO)
    target_link_libraries(va_core PUBLIC openvino::runtime)
endif()

if(WITH_GSTREAMER)
    target_sources(va_core PRIVATE capture/gst-capture.cpp)
    target_compile_definitions(va_core PUBLIC WITH_GSTREAMER)
    target_link_libraries(va_core PRIVATE PkgConfig::GST)
endif()

if(WITH_TENSORRT)
    target_sources(va_core PRIVATE
        inference/backends/tensorrt-backend.cpp
        inference/backends/cuda-preprocess.cu)
    target_compile_definitions(va_core PUBLIC WITH_TENSORRT)
    target_include_directories(va_core PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(va_core PRIVATE
        ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY} CUDA::cudart)
    # On a Jetson, TensorRT and CUDA depend on driver libraries that the
    # NVIDIA container runtime mounts when the container starts; they are
    # absent while the image is built
    target_link_options(va_core INTERFACE -Wl,--allow-shlib-undefined)
endif()

if(WITH_NVMM)
    target_sources(va_core PRIVATE capture/nvmm-frame.cpp)
    target_compile_definitions(va_core PUBLIC WITH_NVMM)
    target_include_directories(va_core PRIVATE
        ${NVBUFSURFACE_INCLUDE_DIR} ${EGL_INCLUDE_DIR})
    # libcuda for the EGL interop calls, libdl for libnvbufsurface.so,
    # which exists only on the board and is loaded at run time
    target_link_libraries(va_core PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
endif()

if(WITH_MQTT)
    target_compile_definitions(va_core PRIVATE WITH_MQTT)
    target_include_directories(va_core PRIVATE ${MOSQUITTO_INCLUDE_DIR})
    target_link_libraries(va_core PRIVATE ${MOSQUITTO_LIBRARY})
endif()

add_executable(video_analytics main.cpp)
target_link_libraries(video_analytics PRIVATE va_core)
