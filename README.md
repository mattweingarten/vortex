# Vortex + Rodinia Integration

This repository extends the [Vortex GPGPU Simulator](https://github.com/vortexgpgpu/vortex) with selected [Rodinia GPU benchmarks](https://github.com/yuhc/gpu-rodinia) and custom performance counters.

---

## Repository Overview

This fork includes three additional benchmarks from the Rodinia OpenCL suite, integrated to run on Vortex’s OpenCL runtime and RTL simulator:

### Integrated Benchmarks:

- `hotspot` – fully functional and verified
- `pathfinder` – compiles successfully (see notes)
- `nw` (Needleman-Wunsch) – compiles successfully (kernel fixes may be required for full functionality)

> All benchmarks are compiled with host-side `.cc` wrappers to ensure compatibility with Vortex's C++ runtime environment.

---

## How to Build and Run Benchmarks

### 1. Clone and Set Up Vortex

```bash
git clone https://github.com/vortexgpgpu/vortex.git
cd vortex

Ensure required dependencies are installed: LLVM, PoCL, g++, CMake, Python3.
2. Build the Vortex Runtime Library

cd runtime/rtlsim
make

This will produce the libvortex.a library in:

vortex/build/runtime/lib/

3. Add and Build Benchmarks

Each benchmark is located in:

vortex/tests/opencl/<benchmark_name>

To run a benchmark:

./ci/blackbox.sh --cores=4 --app=<benchmark_name> --driver=rtlsim --perf=1

For example:

./ci/blackbox.sh --cores=4 --app=hotspot --driver=rtlsim --perf=1
./ci/blackbox.sh --cores=4 --app=pathfinder --driver=rtlsim --perf=1
./ci/blackbox.sh --cores=4 --app=nw --driver=rtlsim --perf=1

Notes

    Benchmarks from Rodinia were modified to accept Vortex-compatible arguments and OpenCL setup.

    pathfinder and nw use .cc files as host code wrappers to ensure proper linking with libvortex.a.

 Performance Counters Added

This repository includes custom performance metrics integrated into Vortex:
1. branch_prediction_accuracy

    Tracks correct vs. incorrect branch predictions.

    Implemented inside runtime/utils.cpp.

2. warp_efficiency

    Measures GPU SIMD lane utilization.

    Higher values = better parallelism.

You must run benchmarks with --perf=1 to activate and log these counters:

./ci/blackbox.sh --cores=4 --app=hotspot --driver=rtlsim --perf=1

 Project Structure

vortex/
├── runtime/
│   └── rtlsim/              # Vortex simulator runtime
├── tests/
│   └── opencl/
│       ├── hotspot/         # Rodinia-integrated benchmark
│       ├── pathfinder/      # Rodinia-integrated benchmark
│       └── nw/              # Rodinia-integrated benchmark
├── utils.cpp                # Modified with custom performance metrics



 Acknowledgments

    Vortex GPGPU Simulator

    Rodinia Benchmark Suite
