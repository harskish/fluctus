#pragma once

#ifdef GPU
typedef float cl_float;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef char cl_uchar;
#else
#include "math/float3.hpp"
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
    union {
        cl_uint iStart;		// leaf node, index into index list
        cl_uint rightChild; // internal node, index into node vector (left child always current + 1)
    };
    cl_uchar nPrims;	    // 0 for interior nodes
} GPUNode;

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

enum lightType
{
    POINT,
    AREA,
    DIRECTIONAL
};

typedef struct
{
    enum lightType type;
    float3 color;
    cl_float intensity;
    union
    {
        float3 pos; // P/A
        float3 dir; // D
    };
    // more params needed for area lights...
} Light;

typedef struct
{
    float3 P;
    float3 N;
    cl_float t;
    cl_int i; // index of hit primitive, -1 by default
} Hit;

typedef struct
{
    float3 pos;
    float3 dir;
    float3 up;
    float3 right;
    cl_float fov;
} Camera;

typedef struct
{
    cl_uint width;         // window width
    cl_uint height;        // window height
    cl_uint n_objects;     // number of objects in scene
    cl_uint n_tris;
    cl_uint n_lights;      // number of lights in scene
    Camera camera;         // camera struct
} RenderParams;
