## Dependencies

- OpenCL 1.2
- OpenGL
- GLEW 1/2
- GLFW 3.2
- DevIL 1.8.0

Windows dependencies included.


## Windows

- Install an OpenCL SDK for libs and headers
	- Intel OpenCL SDK recommended for kernel debugging support on Intel CPUs
	- Alternatives include NVIDIA CUDA Toolkit, AMD APP SDK
- Open VS2015 solution (or run CMake)
- Copy DevIL.dll, ILU.dll and ILUT.dll (non-unicode) from dependencies/ to binary output folder
	- Also done as a custom build step
- Run in debug mode for CPU kernel debugging, release mode for performance

## Mac

- Install Xcode Command Line Tools (for clang)
- Apple OpenCL framework used for OpenCL support
- Install dependencies with [Homebrew][homebrew]:
	```
    brew install glfw3 glew devil
    ```
- Compile:
    ```
    mkdir build
    cd build && cmake ..
    make
    ```
- Run (in project root):
    ```
    ./fluctus
    ```
- Alternatively, JetBrains CLion can compile the project directly using CMakeLists.txt    
    


## Linux (Debian / Ubuntu)

**Not thoroughly tested (but confirmed working on Ubuntu 16.04)**

- Install OpenCL SDK (CUDA Toolkit / Intel OpenCL SDK / AMD APP SDK)
	- For CUDA Toolkit (cuda_8.0.27_linux.run), driver part can be skipped
- Install dependencies:
	```
    sudo apt-get install build-essential opencl-headers libdevil-dev libglew-dev libglfw3-dev
    ```

- Compile:
    ```
    mkdir build
    cd build && cmake ..
    make
    ```
- Run (in project root):
    ```
    ./fluctus
    ```
- Alternatively, JetBrains CLion can compile the project directly using CMakeLists.txt  


[homebrew]: https://brew.sh/