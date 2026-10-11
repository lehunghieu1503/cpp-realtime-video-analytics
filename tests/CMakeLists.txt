# Each test is one source file linked against the application code
foreach(name inference pipeline serializer latency-tracker sse-server rules)
    string(REPLACE "-" "_" target "test_${name}")
    add_executable(${target} test-${name}.cpp)
    target_link_libraries(${target} PRIVATE va_core)
endforeach()
