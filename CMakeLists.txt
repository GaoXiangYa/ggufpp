cmake_minimum_required(VERSION 3.20)
project(ggufpp VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)

option(GGUFPP_BUILD_EXAMPLES "Build the GGUF inspection tool" ON)
option(GGUFPP_BUILD_BENCHMARKS "Build the standalone benchmark" OFF)
option(GGUFPP_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(GGUFPP_BUILD_FUZZER "Build the Clang/libFuzzer parser harness" OFF)

add_library(ggufpp STATIC src/parser.cpp src/file.cpp src/tensor.cpp)
add_library(ggufpp::ggufpp ALIAS ggufpp)
target_compile_features(ggufpp PUBLIC cxx_std_20)
set_target_properties(ggufpp PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_include_directories(ggufpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

function(ggufpp_configure_target target)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
        if(GGUFPP_SANITIZERS)
            target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PRIVATE -fsanitize=address,undefined)
        endif()
    endif()
endfunction()
ggufpp_configure_target(ggufpp)

if(GGUFPP_BUILD_EXAMPLES)
    add_executable(ggufpp_inspect examples/inspect.cpp)
    target_link_libraries(ggufpp_inspect PRIVATE ggufpp::ggufpp)
    ggufpp_configure_target(ggufpp_inspect)
endif()

if(BUILD_TESTING)
    add_executable(ggufpp_tests tests/parser_tests.cpp)
    target_link_libraries(ggufpp_tests PRIVATE ggufpp::ggufpp)
    ggufpp_configure_target(ggufpp_tests)
    add_test(NAME ggufpp_tests COMMAND ggufpp_tests)
    set(consumer_flags "")
    if(GGUFPP_SANITIZERS AND NOT MSVC)
        set(consumer_flags "-fsanitize=address,undefined -fno-omit-frame-pointer")
    endif()
    add_test(NAME ggufpp_install_consumer COMMAND ${CMAKE_COMMAND}
        -Dproject_build=${CMAKE_CURRENT_BINARY_DIR}
        -Dcompiler=${CMAKE_CXX_COMPILER}
        -Dconfig=$<CONFIG>
        "-Dconsumer_flags=${consumer_flags}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/install_smoke.cmake)
    foreach(header document error file ggufpp metadata tensor)
        set(header_source "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${header}.cpp")
        file(GENERATE OUTPUT "${header_source}" CONTENT "#include <ggufpp/${header}.hpp>\n")
        add_library(ggufpp_header_${header} OBJECT "${header_source}")
        target_link_libraries(ggufpp_header_${header} PRIVATE ggufpp::ggufpp)
        ggufpp_configure_target(ggufpp_header_${header})
    endforeach()
endif()

if(GGUFPP_BUILD_BENCHMARKS)
    add_executable(ggufpp_benchmark benchmarks/parse.cpp)
    target_include_directories(ggufpp_benchmark PRIVATE tests)
    target_link_libraries(ggufpp_benchmark PRIVATE ggufpp::ggufpp)
    ggufpp_configure_target(ggufpp_benchmark)
endif()

if(GGUFPP_BUILD_FUZZER)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR MSVC)
        message(FATAL_ERROR "GGUFPP_BUILD_FUZZER requires a Clang compiler with libFuzzer")
    endif()
    add_executable(ggufpp_fuzz tests/fuzz_parse.cpp src/parser.cpp src/tensor.cpp)
    target_include_directories(ggufpp_fuzz PRIVATE include)
    target_compile_features(ggufpp_fuzz PRIVATE cxx_std_20)
    ggufpp_configure_target(ggufpp_fuzz)
    target_compile_options(ggufpp_fuzz PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(ggufpp_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

install(TARGETS ggufpp EXPORT ggufpp_targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ggufpp_targets NAMESPACE ggufpp:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggufpp)
configure_package_config_file(cmake/ggufpp-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ggufpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggufpp)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ggufpp-config-version.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggufpp-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ggufpp-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggufpp)
