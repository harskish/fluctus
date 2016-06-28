#pragma once

#ifdef GPU
typedef float4 cl_float4;
typedef float cl_float;
typedef unsigned int cl_uint;
#endif

typedef struct
{
    cl_float4 orig;
    cl_float4 dir;
} Ray;

typedef struct
{
    cl_float R;     // 4B (padded to 16B?)
    cl_float4 pos;  // 16B
    cl_float4 Kd;   // 16B
} Sphere;           // 48B

typedef struct
{
    cl_float4 pos;
    cl_float4 dir;
    cl_float fov;
} Camera;


typedef struct
{
    cl_float4 a;
} TestStruct;

typedef struct
{
    cl_uint width;         // window width
    cl_uint height;        // window height
    cl_uint n_objects;     // number of objects in scene
    Camera camera;              // camera struct
    cl_float sin2;                 // sinewave for movement etc.
} RenderParams;