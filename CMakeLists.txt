cmake_minimum_required(VERSION 3.20)
project(eurobot_alphazero_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EUROBOT_ENABLE_WARNINGS "Enable common compiler warnings" ON)

if(EUROBOT_ENABLE_WARNINGS)
  if(MSVC)
    add_compile_options(/W4 /permissive-)
  else()
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

# Expect LibTorch to be discoverable via CMAKE_PREFIX_PATH or Torch_DIR.
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)

message(STATUS "LibTorch CUDA support: ${Torch_CUDA_FOUND}")

add_library(eurobot_alphazero STATIC
  src/alphazero/action_helper.cpp
  src/alphazero/eurobot_world.cpp
  src/alphazero/eurobot_render_cv2.cpp
  src/alphazero/mcts.cpp
  src/alphazero/policy_network.cpp
  src/alphazero/trainer.cpp
  src/alphazero/tensorboard_logger.cpp
)

target_include_directories(eurobot_alphazero
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(eurobot_alphazero
  PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

set_target_properties(eurobot_alphazero PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_executable(eurobot_app
  src/main.cpp
)

target_link_libraries(eurobot_app
  PRIVATE
    eurobot_alphazero
    ${TORCH_LIBRARIES}
)

target_link_libraries(eurobot_app 
  PRIVATE 
    yaml-cpp
)

add_executable(eurobot_eval
  src/eurobot_eval.cpp
)

target_link_libraries(eurobot_eval
  PRIVATE
    eurobot_alphazero
    yaml-cpp
)

if(MSVC)
  target_compile_definitions(eurobot_app PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(eurobot_eval PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
