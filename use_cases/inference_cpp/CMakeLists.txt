cmake_minimum_required(VERSION 3.16)
project(inference_tensorrt)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 14)

# To include the "include" folder to the path, and OpenCV
include_directories(include)

# To compile the library of TensorRT
add_subdirectory(../../tensorrt_cpp/tensorrt_cpp_api ${CMAKE_BINARY_DIR}/tensorrt_cpp_api)
add_subdirectory(../../tensorrt_cpp/nn_handler_lib ${CMAKE_BINARY_DIR}/nn_handler_lib)

# To compile the main program
add_executable(single_io_single_batch single_io_single_batch/single_io_single_batch.cpp)
target_link_libraries(single_io_single_batch PRIVATE nn_handler_lib)

add_executable(single_io_multi_batch single_io_multi_batch/single_io_multi_batch.cpp)
target_link_libraries(single_io_multi_batch PRIVATE nn_handler_lib)

add_executable(multi_io_single_batch multi_io_single_batch/multi_io_single_batch.cpp)
target_link_libraries(multi_io_single_batch PRIVATE nn_handler_lib)

add_executable(multi_io_multi_batch multi_io_multi_batch/multi_io_multi_batch.cpp)
target_link_libraries(multi_io_multi_batch PRIVATE nn_handler_lib)

add_executable(single_io_single_batch_cuda single_io_single_batch_cuda/single_io_single_batch_cuda.cpp)
target_link_libraries(single_io_single_batch_cuda PRIVATE nn_handler_lib)

add_executable(multi_io_multi_batch_cuda multi_io_multi_batch_cuda/multi_io_multi_batch_cuda.cpp)
target_link_libraries(multi_io_multi_batch_cuda PRIVATE nn_handler_lib)
