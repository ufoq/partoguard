cmake_minimum_required(VERSION 3.22.1)

project("partoguard-llama" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# Android ABI-specific settings
if(DEFINED ANDROID_ABI)
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_CPU_KLEIDIAI ON)
    endif()
endif()

# Pull in upstream llama.cpp as a subdirectory
# LLAMA_BUILD_TOOLS=ON is required for mtmd (set via Gradle cmake arguments)
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/llama.cpp)
add_subdirectory(${LLAMA_SRC} build-llama)

# Our JNI bridge library
add_library(${CMAKE_PROJECT_NAME} SHARED
    partoguard_jni.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/ggml/include
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/tools/mtmd
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    llama
    llama-common
    mtmd
    android
    log
)
