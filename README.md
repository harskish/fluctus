Fluctus
====================

OpenCL wavefront path tracer
--------------

![Country Kitchen](gallery/kitchen.png)

## Features
- Physically based renderer
    -  OBJ + MTL scenefiles
    -  Lambertian, glossy, ideal specular, and rough specular ([GGX][ggx]) BSDFs
    -  Multiple importance sampled environment maps (alias method)
- [Wavefront path tracing][wavefront] at interactive framerates
    - Optimized structure of arrays data layout
    - Efficient BVH with [spatial splits][sbvh]
- Supports a wide variety of systems
    - Cross-platform (Windows, MacOS, Linux)
    - NVIDIA, AMD and Intel GPUs and CPUs
    - CPU debugging with [Intel's OpenCL SDK][intel_ocl]
- Nanogui-based [user interface](gallery/user_interface.png)
	- Uses only modern OpenGL (3.3+)
	- Drag and drop scene files and environment maps


## Usage

Rename settings_default.json to settings.json. Modify to set default OpenCL device, render scale, window dimensions etc.

### Controls

| Key                     | Action                                                                                |
|-------------------------|---------------------------------------------------------------------------------------|
| **W / A / S / D**       | Move camera (up/down with R/F)                                                        |
| **Mouse left**          | Look around                                                                           |
| **Scroll Up / Down**    | Adjust movement speed                                                                 |
| **Space**               | Place area light                                                                      |
| **F1**                  | Reset camera                                                                          |
| **F2**                  | Save camera/area light state                                                          |
| **F3**                  | Load saved state                                                                      |
| **F5**                  | Export image                                                                          |
| **H**                   | Toggle light sources (environment/area/both)                                          |
| **I / K**               | Adjust max bounces                                                                    |
| **Z / X**               | Adjust environment map emission                                                       |
| **L**                   | Open scene selector                                                                   |
| **M**                   | Switch sampling mode                                                                  |
| **U**                   | Toggle UI                                                                             |
| **Page Up / Down**      | Adjust area light emission                                                            |
| **1-5**                 | Select scene 1-5 (if set in settings.json)                                            |
| **7**                   | Switch metween micro-/megakernel                                                      |
| **8 / 9**               | Change area light size                                                                |
| **, / .**               | Change FOV                                                                            |

## Build

See the [build instructions](./BUILDING.md).

## License

See the [LICENSE](./LICENSE.md) file for license rights and limitations (MIT).


[intel_ocl]: https://software.intel.com/intel-opencl
[wavefront]: http://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus
[sbvh]: http://www.nvidia.com/object/nvidia_research_pub_012.html
[ggx]: https://doi.org/10.2312/EGWR/EGSR07/195-206
