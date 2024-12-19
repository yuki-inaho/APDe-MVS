cmake_minimum_required(VERSION 3.23.0)
project(APD)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

# recommand >=4.6.0
find_package(OpenCV REQUIRED)
# recommand >=1.78.0
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)
# recommand >=11.8
find_package(CUDA REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS})
include_directories(.)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11
        --compiler-options -Wall)

if (CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif ()

cuda_add_executable(
        APD
        main.h
        APD.h
        APD.cpp
        APD.cu
        main.cpp
        ThreadPool.h
)

# find cuda arch
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
message(STATUS "CUDA_ARCH_LIST: ${CUDA_ARCH_LIST}")
set_target_properties(APD PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

target_link_libraries(
        APD
        "${Boost_LIBRARIES}"
        "${OpenCV_LIBS}"
)
