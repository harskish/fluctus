#pragma once

#ifdef GPU
typedef float cl_float;
typedef unsigned int cl_uint;
#else
#include "math/float3.hpp"
using FireRays::float4;
#endif

typedef struct
{
    float4 orig;
    float4 dir;
} Ray;

typedef struct
{
    cl_float R;  // 4B (padded to 16B?)
    float4 pos;  // 16B
    float4 Kd;   // 16B
} Sphere;        // 48B

typedef struct
{
    float4 pos;
    float4 dir;
    cl_float fov;
} Camera;

typedef struct
{
    cl_uint width;         // window width
    cl_uint height;        // window height
    cl_uint n_objects;     // number of objects in scene
    Camera camera;         // camera struct
} RenderParams;