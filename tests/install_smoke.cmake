set(smoke_dir "${project_build}/install-smoke")
file(MAKE_DIRECTORY "${smoke_dir}/source")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${project_build}"
    --prefix "${smoke_dir}/prefix" --config "${config}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Install failed: ${output}\n${error}")
endif()
file(WRITE "${smoke_dir}/source/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.20)
project(consumer LANGUAGES CXX)
find_package(ggufpp 0.1 CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE ggufpp::ggufpp)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/consumer_path_$<CONFIG>.txt" CONTENT "$<TARGET_FILE:consumer>")
]=])
file(WRITE "${smoke_dir}/source/main.cpp" [=[
#include <ggufpp/ggufpp.hpp>
#include <array>
int main() {
    const std::array<std::byte, 24> bytes{};
    try { (void)ggufpp::parse(bytes); }
    catch (const ggufpp::ParseError&) { }
    try { (void)ggufpp::MappedFile::open("ggufpp-install-smoke-missing.gguf"); }
    catch (const ggufpp::ParseError&) { return 0; }
    return 1;
}
]=])
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${smoke_dir}/source" -B "${smoke_dir}/build"
    "-DCMAKE_PREFIX_PATH=${smoke_dir}/prefix" "-DCMAKE_CXX_COMPILER=${compiler}"
    "-DCMAKE_CXX_FLAGS=${consumer_flags}"
    "-DCMAKE_BUILD_TYPE=${config}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer configure failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${smoke_dir}/build" --config "${config}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer build failed: ${output}\n${error}")
endif()
file(READ "${smoke_dir}/build/consumer_path_${config}.txt" consumer_executable)
execute_process(COMMAND "${consumer_executable}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer execution failed: ${output}\n${error}")
endif()
