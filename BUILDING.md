## Dependencies

- CMake 3.3
- OpenCL 1.2
- OpenGL 3.3
- GLFW 3.2
- DevIL 1.8.0
- Nanogui (submodule)

Prebuilt DevIL binaries for Windows included.

## Windows

- Install an OpenCL SDK for libs and headers
	- Intel OpenCL SDK recommended for kernel debugging support on Intel CPUs
	- Alternatives include NVIDIA CUDA Toolkit, AMD APP SDK
- Setup submodules:
    ```
    git submodule update --init --recursive
    ```
- Generate build files:
    ```
    mkdir build
	cd build
	cmake .. -G "Visual Studio 15 2017 Win64"
    ```
- Build using Visual Studio solution (set Fluctus as StartUp project)
- Run in debug mode for CPU kernel debugging, release mode for performance

## Mac

- Install Xcode Command Line Tools (for clang)
- Apple OpenCL framework used for OpenCL support
- Install dependencies with [Homebrew][homebrew]:
	```
    brew install glfw3 devil
    ```
- Setup submodules:
    ```
    git submodule update --init --recursive
    ```
- Compile:
    ```
    mkdir build
    cd build && cmake ..
    make
    ```
- Run (in project root):
    ```
    ./build/fluctus
    ```
- Alternatively, JetBrains CLion can compile the project directly using CMakeLists.txt (working directory has to be changed in IDE settings)
    


## Linux (Debian / Ubuntu)

- Install OpenCL SDK (CUDA Toolkit / Intel OpenCL SDK / AMD APP SDK)
- Install dependencies:
	```
    sudo apt-get install build-essential opencl-headers libdevil-dev libglfw3-dev xorg-dev
    ```
- Setup submodules:
    ```
    git submodule update --init --recursive
    ```
- Compile:
    ```
    mkdir build
    cd build && cmake ..
    make
    ```
- Run (in project root):
    ```
    ./build/fluctus
    ```
- Alternatively, JetBrains CLion can compile the project directly using CMakeLists.txt (working directory has to be changed in IDE settings)


[homebrew]: https://brew.sh/
