# Test executable for inference engine
add_executable(test_inference
    test-inference.cpp
    ../src/inference/inference-engine.cpp
    ../src/inference/pre-process.cpp
    ../src/inference/post-process.cpp
)

# Link dependencies
target_link_libraries(test_inference
    PRIVATE
        ${OpenCV_LIBS}
        ONNXRuntime::ONNXRuntime
        pthread
)

# Include directories
# src/ as root so includes like "inference/xxx.hpp" work
target_include_directories(test_inference
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)

# Test executable for pipeline (BoundedQueue, FrameData, AnalyticsResult)
add_executable(test_pipeline
    test-pipeline.cpp
)

# Link dependencies
target_link_libraries(test_pipeline
    PRIVATE
        ${OpenCV_LIBS}
        pthread
)

# Include directories
target_include_directories(test_pipeline
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)

# Test executable for result serializer (JSON serialization)
add_executable(test_serializer
    test-serializer.cpp
)

# Link dependencies
target_link_libraries(test_serializer
    PRIVATE
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
)

# Include directories
target_include_directories(test_serializer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)

# Test executable for latency tracker and benchmark report
add_executable(test_latency_tracker
    test-latency-tracker.cpp
)

# Link dependencies
target_link_libraries(test_latency_tracker
    PRIVATE
        ${OpenCV_LIBS}
)

# Include directories
target_include_directories(test_latency_tracker
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)

# Test executable for SSE/MJPEG server (event delivery, worker thread release)
add_executable(test_sse_server
    test-sse-server.cpp
    ../src/server/sse-server.cpp
)

# Link dependencies
target_link_libraries(test_sse_server
    PRIVATE
        ${OpenCV_LIBS}
        httplib::httplib
        pthread
)

# Include directories
target_include_directories(test_sse_server
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${OpenCV_INCLUDE_DIRS}
)
