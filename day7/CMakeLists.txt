cmake_minimum_required(VERSION 3.20)
project(day6 C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCL REQUIRED)

add_executable(solution
  src/main.cpp
  )
target_include_directories(solution PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(solution PRIVATE ${OpenCL_LIBRARIES})
