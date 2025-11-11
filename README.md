# C++ AlphaZero Eurobot

This directory contains a C++ project that implements AlphaZero algorithm. It is configured to build against LibTorch (the official PyTorch C++ API).

## Layout

```
├── assets
│   ├── arena.xml
│   ├── granary_beta.png
│   ├── table_beta.png
│   ├── table.png
│   └── view_arena.py
├── CMakeLists.txt
├── configs
│   ├── blue_robot.yaml
│   └── yellow_robot.yaml
├── demo.gif
├── include
│   └── alphazero
│       ├── action_helper.hpp
│       ├── eurobot_world.hpp
│       ├── mcts.hpp
│       ├── policy_network.hpp
│       └── trainer.hpp
├── README.md
├── src
│   ├── alphazero
│   │   ├── action_helper.cpp
│   │   ├── eurobot_world.cpp
│   │   ├── mcts.cpp
│   │   ├── policy_network.cpp
│   │   └── trainer.cpp
│   └── main.cpp
└── utils
    ├── click_coords.py
    └── manual_play.py
```

## Prerequisites

- CMake ≥ 3.20
- A C++20 capable compiler (GCC ≥ 10, Clang ≥ 12, or MSVC ≥ 19.29)
- LibTorch (download from https://pytorch.org/get-started/locally/) extracted somewhere on disk
- Optional: CUDA toolkit if you plan to build the CUDA variant of LibTorch
- OpenCV installed, or passed to cmake

## Configure & Build

Use CMake with the standalone project entry point in this directory. Tell CMake where to find LibTorch via `CMAKE_PREFIX_PATH` (or by setting `Torch_DIR`).

```bash
cmake -S src/ -B build/ -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build/
```

## Alternative build (docker)
If you are on windows installation is trickier.
To use docker do:

```bash
docker build -t eurobot:cpu .

docker run --rm -it -v ${PWD}:/work -w /work eurobot:cpu bash -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/libtorch && cmake --build build -j$(nproc)'
```
then run the built app with:

```bash
docker run --rm -it -v ${PWD}:/work -w /work eurobot:cpu bash -lc './build/eurobot_app --config configs/world_test.yaml'
```