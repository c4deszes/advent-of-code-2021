cmake_minimum_required(VERSION 3.20)
project(day4 C CXX CUDA)

add_executable(solution
  src/main.c
  src/cuda.cu
  )
target_include_directories(solution PRIVATE include)
set_property(TARGET solution 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON) 
set_property(TARGET solution PROPERTY CUDA_ARCHITECTURES 50-real)

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(solution PUBLIC cxx_std_11)

