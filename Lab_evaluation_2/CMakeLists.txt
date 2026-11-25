cmake_minimum_required(VERSION 3.24)
project(AI_Tensor_Analytics_CUDA LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
add_executable(cpu_histogram src/cpu_histogram.cpp)
foreach(name global_histogram shared_histogram uint8_distribution_experiment
             uint8_bin_count_experiment distribution_experiment
             quantization_calibration gpu_parallel_cdf div2k_histogram_benchmark)
  add_executable(${name} src/${name}.cu)
  set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()
