# C++ AlphaZero Eurobot

This directory contains a starter C++ project that implements high-level AlphaZero algorithm. It is configured to build against LibTorch (the official PyTorch C++ API) and provides minimal scaffolding for a policy network, Monte-Carlo Tree Search (MCTS) loop, and a training harness.

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

## Checkpointing & Resume

Training configuration is read from `configs/world.yaml`. Set the following keys under the `training` node to enable resume:

- `checkpoint_path`: File that stores the trainer state (policy network, optimizer, replay buffer, and iteration counters). If omitted, it defaults to `<log_dir>/<run_name>_checkpoint.pt`.
- `checkpoint_interval`: How often (in iterations) to write the checkpoint. Set to `0` to disable automatic checkpointing.
- `resume_from_checkpoint`: When `true`, the trainer restores the checkpoint before continuing the loop.

With checkpoints enabled the trainer persistently saves progress after each iteration (or at the requested interval) and you can stop/restart the binary without retraining from scratch. The checkpoint also feeds TensorBoard step counters so the logs remain continuous.
