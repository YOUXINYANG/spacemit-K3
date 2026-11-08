cmake_minimum_required(VERSION 3.20)
project(k3_tcm_copy LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(spine_tcm CONFIG REQUIRED)
set(K3_LLAMA_BIN "$ENV{HOME}/qwen/builds/ime2-baseline/bin" CACHE PATH
    "Directory containing the existing K3 libggml-cpu.so.0")
if(NOT EXISTS "${K3_LLAMA_BIN}/libggml-cpu.so.0")
    message(FATAL_ERROR "Missing ${K3_LLAMA_BIN}/libggml-cpu.so.0")
endif()
add_library(k3_ggml_cpu SHARED IMPORTED)
set_target_properties(k3_ggml_cpu PROPERTIES
    IMPORTED_LOCATION "${K3_LLAMA_BIN}/libggml-cpu.so.0")

add_executable(tcm-copy main.cpp prefetch_copy.cpp)
set_source_files_properties(prefetch_copy.cpp PROPERTIES
    COMPILE_OPTIONS "-march=rv64gcv_zicbop;-mabi=lp64d")
target_compile_definitions(tcm-copy PRIVATE _GNU_SOURCE SPINE_TCM_DIRECT_LINK=1)
target_compile_options(tcm-copy PRIVATE -Wall -Wextra -fno-builtin-memcpy)
target_link_libraries(tcm-copy PRIVATE k3_ggml_cpu spine_tcm::spine_tcm Threads::Threads)
set_target_properties(tcm-copy PROPERTIES BUILD_RPATH "${K3_LLAMA_BIN}")

enable_testing()
add_executable(test-check-copy test_check_copy.cpp)
target_compile_options(test-check-copy PRIVATE -Wall -Wextra)
add_test(NAME copy-validator COMMAND test-check-copy)

add_executable(test-copy-workload test_copy_workload.cpp)
target_compile_options(test-copy-workload PRIVATE -Wall -Wextra)
add_test(NAME copy-workload COMMAND test-copy-workload)
