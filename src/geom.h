#pragma once

#ifdef GPU
typedef float cl_float;
typedef int cl_int;
typedef unsigned int cl_uint;
#else
#include "math/float3.hpp"
using FireRays::float4;
#endif

#define PI 3.14159265358979323846
#define toRad(deg) (deg * PI / 180)

typedef struct
{
    float4 orig;
    float4 dir;
} Ray;

typedef struct
{
    cl_float R;  // 4B (padded to 16B?)
    float4 P;    // 16B
    float4 Kd;   // 16B
} Sphere;        // 48B

enum lightType
{
    POINT,
    AREA,
    DIRECTIONAL
};

typedef struct
{
    enum lightType type;
    float4 color;
    cl_float intensity;
    union
    {
        float4 pos; // P/A
        float4 dir; // D
    };
    // more params needed for area lights...
} Light;

typedef struct
{
    float4 P;
    float4 N;
    cl_float t;
    cl_int i; // index of hit primitive, -1 by default
} Hit;

typedef struct
{
    float4 pos;
    float4 dir;
    float4 up;
    float4 right;
    cl_float fov;
} Camera;

typedef struct
{
    cl_uint width;         // window width
    cl_uint height;        // window height
    cl_uint n_objects;     // number of objects in scene
    cl_uint n_lights;      // number of lights in scene
    Camera camera;         // camera struct
} RenderParams;
