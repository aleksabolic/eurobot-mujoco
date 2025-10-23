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

message(STATUS "LibTorch CUDA support: ${Torch_CUDA_FOUND}")

add_library(eurobot_alphazero STATIC
  src/alphazero/policy_network.cpp
  src/alphazero/mcts.cpp
  src/alphazero/trainer.cpp
)

target_include_directories(eurobot_alphazero
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(eurobot_alphazero
  PUBLIC
    ${TORCH_LIBRARIES}
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

if(MSVC)
  target_compile_definitions(eurobot_app PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
