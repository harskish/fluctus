#pragma once

#ifdef GPU
typedef float cl_float;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef char cl_uchar;
typedef bool cl_bool;
typedef float2 cl_float2;
#else
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"
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
    float3 p; // 16B
    float3 n; // 16B
    float3 t; // 16B
} Vertex; // >= 48B

typedef struct
{
    Vertex v0;
    Vertex v1;
    Vertex v2;
    cl_int matId;
} Triangle; // this struct is used interchangeably with RTTriangle...sizes must match!

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
    float3 Kd;     // diffuse reflectivity
    float3 Ks;     // specular reflectivity 
    float3 Ke;     // emission
    cl_float Ns;   // specular exponent (shininess), normally in [0, 1000]
    cl_float Ni;   // index of refraction
    cl_int map_Kd; // diffuse texture descriptor idx
    cl_int map_Ks; // specular texture descriptor idx
} Material;

typedef struct
{
    cl_uint offset; // start of texture data in global array
    cl_uint width;
    cl_uint height;
} TexDescriptor;

typedef struct
{
    float3 P;
    float3 N;
    cl_float t;
    cl_float2 uvTex;
    cl_int i; // index of hit triangle, -1 by default
    cl_int matId; // index of hit material
} Hit;

#define EMPTY_HIT(tmax) { (float3)(0.0f), (float3)(0.0f), tmax, (float2)(0.0f), -1, -1 }

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
} RenderParams;


// Microkernel state structs
typedef enum
{
    MK_RT_NEXT_VERTEX = 0,
    MK_HIT_NOTHING = 1,
    MK_HIT_OBJECT = 2,
    MK_DL_ILLUMINATE = 3,
    MK_DL_SAMPLE_BSDF = 4,
    MK_RT_DL = 5,
    MK_GENERATE_NEXT_VERTEX_RAY = 6,
    MK_SPLAT_SAMPLE = 7,
    MK_NEXT_SAMPLE = 8,
    MK_GENERATE_CAMERA_RAY = 9,
    MK_DONE = 10
} PathPhase;

typedef struct
{
    float3 T; // throughput
    PathPhase phase;
    float pdf;
    cl_uint seed;
} GPUTaskState;