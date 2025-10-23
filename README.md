# C++ AlphaZero Template

This directory contains a starter C++ project that mirrors the high-level AlphaZero structure used by the Python prototype. It is configured to build against LibTorch (the official PyTorch C++ API) and provides minimal scaffolding for a policy network, Monte-Carlo Tree Search (MCTS) loop, and a training harness.

## Layout

```
src/cpp
├── CMakeLists.txt           # Standalone CMake entry point
├── include/alphazero        # Public headers
│   ├── mcts.hpp
│   ├── policy_network.hpp
│   └── trainer.hpp
├── src                      # Library / executable sources
│   ├── alphazero
│   │   ├── mcts.cpp
│   │   ├── policy_network.cpp
│   │   └── trainer.cpp
│   └── main.cpp             # Example wiring of the components
└── README.md
```

The code is intentionally lightweight and focuses on API shape and compilation setup. Replace the stub logic with your production implementation as you port features from Python.

## Prerequisites

- CMake ≥ 3.20
- A C++20 capable compiler (GCC ≥ 10, Clang ≥ 12, or MSVC ≥ 19.29)
- LibTorch (download from https://pytorch.org/get-started/locally/) extracted somewhere on disk
- Optional: CUDA toolkit if you plan to build the CUDA variant of LibTorch

## Configure & Build

Use CMake with the standalone project entry point in this directory. Tell CMake where to find LibTorch via `CMAKE_PREFIX_PATH` (or by setting `Torch_DIR`).

```bash
cmake -S src/cpp -B build/cpp -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build/cpp
```

The first command configures the project and generates build files in `build/cpp/`. The second command compiles the static library (`eurobot_alphazero`) and the sample executable (`eurobot_app`).

### Running the sample

```bash
./build/cpp/eurobot_app
```

The executable prints dummy metrics and shows how to wire the trainer and search loop together. Replace this with your own entry point once you integrate the Eurobot environment.

## Next Steps

- Replace the placeholder logic in `alphazero::MCTS` and `alphazero::AlphaZeroTrainer` with the real algorithms.
- Port environment interactions from Python one component at a time, exposing them via C++ headers inside `include/`.
- Add tests (e.g., GoogleTest or Catch2) and hook them into the CMake project once the core code solidifies.

Happy hacking!
