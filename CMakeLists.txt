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
)
target_include_directories(bt_core PUBLIC src)

add_executable(bt-inspect tools/bt_inspect.cpp)
target_link_libraries(bt-inspect PRIVATE bt_core)

enable_testing()
add_executable(test_dequant tests/test_dequant.cpp)
target_link_libraries(test_dequant PRIVATE bt_core)
add_test(NAME dequant COMMAND test_dequant)

# CUDA engine targets land here once kernels exist (guarded so CPU-only
# checkouts still build the loader and tools).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA enabled: engine targets will build")
else()
    message(STATUS "CUDA not found: building loader/tools only")
endif()
