# Each test is one source file linked against the application code
foreach(name inference pipeline serializer latency-tracker sse-server rules
        capture-timestamps lane stall)
    string(REPLACE "-" "_" target "test_${name}")
    add_executable(${target} test-${name}.cpp)
    target_link_libraries(${target} PRIVATE va_core)
endforeach()

# This one writes its own GStreamer stream to read back
if(WITH_GSTREAMER)
    target_link_libraries(test_capture_timestamps PRIVATE PkgConfig::GST)
endif()

# Shared with the Python reference: simulation/test
target_compile_definitions(test_lane PRIVATE
    LANE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/lane-calibration.json")
