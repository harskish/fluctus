#pragma once

#ifdef GPU
typedef float cl_float;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef char cl_uchar;
typedef bool cl_bool;
#else
#include "math/float2.hpp"
#include "math/float3.hpp"
using FireRays::float2;
using FireRays::float3;
#endif

#define PI 3.14159265358979323846f
#define toRad(deg) (deg * PI / 180)

typedef struct
{
    float3 orig;
    float3 dir;
} Ray;

typedef struct
{
    cl_float R;  // 4B (padded to 16B?)
    float3 P;    // 16B
    float3 Kd;   // 16B
} Sphere;        // 48B

typedef struct
{
    float3 min;
    float3 max;
} AABB;

typedef struct
{
    AABB box;
    cl_int parent;
    union {
        cl_uint iStart;     // leaf node, index into index list
        cl_uint rightChild; // internal node, index into node vector (left child always current + 1)
    };
    cl_uchar nPrims;        // 0 for interior nodes
} GPUNode;

// Node for a simulated stack on the GPU
typedef struct
{
    cl_uint i; // index of node
    float mint;
} SimStackNode;

typedef struct
{
    float3 p;
    float3 n;
    float3 t;
} Vertex;

typedef struct
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
} Triangle;

typedef struct
{
    float3 E;   // Diffuse emission (W/m^2), ~'color * intensity'?
    float3 pos;
} PointLight;

typedef struct
{
    float3 right;
    float3 up;
    float3 N;
    float3 pos;
    float3 E;        // Diffuse emission (W/m^2)
    float2 size;     // Half of the total width/height, measured from center
} AreaLight;

typedef struct
{
    float3 P;
    float3 N;
    cl_float t;
    cl_int i; // index of hit primitive, -1 by default
} Hit;

typedef struct
{
    float3 pos;     // 16B
    float3 dir;     // 16B
    float3 up;      // 16B
    float3 right;   // 16B
    cl_float fov;   // 4B
} Camera;

typedef struct
{
    AreaLight areaLight;
    Camera camera;         // camera struct
    cl_uint width;         // window width
    cl_uint height;        // window height
    cl_uint n_objects;     // number of objects in scene
    cl_uint n_tris;
    cl_uint n_lights;      // number of lights in scene
    cl_uint useEnvMap;
    cl_uint flashlight;
} RenderParams; //
