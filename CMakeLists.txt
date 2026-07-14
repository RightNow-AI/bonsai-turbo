cmake_minimum_required(VERSION 3.22)
project(bonsai-turbo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(bt_core
    src/gguf.cpp
    src/packs.cpp
    src/retile.cpp
)
target_include_directories(bt_core PUBLIC src)

add_executable(bt-inspect tools/bt_inspect.cpp)
target_link_libraries(bt-inspect PRIVATE bt_core)

enable_testing()
add_executable(test_dequant tests/test_dequant.cpp)
target_link_libraries(test_dequant PRIVATE bt_core)
add_test(NAME dequant COMMAND test_dequant)

add_executable(test_retile tests/test_retile.cpp)
target_link_libraries(test_retile PRIVATE bt_core)
add_test(NAME retile COMMAND test_retile)

# CUDA engine targets (guarded so CPU-only checkouts still build loader/tools).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # must be set BEFORE enable_language(CUDA), which otherwise pins the
    # toolchain default (52 — too old for dp4a)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90)  # H100; use 120 for RTX 5090
    endif()
    enable_language(CUDA)
    # host code is C++20 but CUDA TUs stay on 17: older CMake (<3.24) has no
    # CUDA20 dialect table and nothing in the .cu files needs 20
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA enabled (arch ${CMAKE_CUDA_ARCHITECTURES})")

    add_library(bt_cuda src/cuda/gemv.cu)
    target_include_directories(bt_cuda PUBLIC src)

    add_executable(bt-microbench tools/bt_microbench.cu)
    target_link_libraries(bt-microbench PRIVATE bt_core bt_cuda)
else()
    message(STATUS "CUDA not found: building loader/tools only")
endif()
