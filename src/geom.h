#pragma once

#ifdef GPU
typedef float cl_float;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef char cl_uchar;
typedef bool cl_bool;
#else
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include "cl2.hpp"
#include "math/float2.hpp"
#include "math/float3.hpp"
using FireRays::float2;
using FireRays::float3;
#endif

#define PI (3.14159265358979323846f)
#define M_INV_PI (0.3183098861837907f)
#define M_2PI_F (6.2831853071795864f)
#define toRad(deg) (deg * PI / 180)

// For handling SoA data, only used on GPU
// Variable names gid, numTasks are assumed for brevity
#ifndef USE_SOA
#define ReadF32(member, ptr) ptr[gid].member
#define WriteF32(member, ptr, value) ptr[gid].member = value
#define ReadI32(member, ptr) ptr[gid].member
#define WriteI32(member, ptr, value) ptr[gid].member = value
#define ReadU32(member, ptr) ptr[gid].member
#define WriteU32(member, ptr, value) ptr[gid].member = value
#define ReadFloat2(member, ptr) ptr[gid].member
#define ReadFloat3(member, ptr) ptr[gid].member
#define WriteFloat2(member, ptr, value) ptr[gid].member = (float2)(value.x, value.y)
#define WriteFloat3(member, ptr, value) ptr[gid].member = (float3)(value.x, value.y, value.z)
#else
#define OffsetOf(member) (uint)(&((GPUTaskState*)0)->member)
#define ReadF32(member, ptr) ((global float*)ptr)[OffsetOf(member) / (uint)sizeof(float) * numTasks + gid]
#define WriteF32(member, ptr, value) ReadF32(member, ptr) = value
#define ReadF32Vec(member, cmp, ptr) ((global float*)ptr)[(OffsetOf(member) + cmp * (uint)sizeof(float)) / (uint)sizeof(float) * numTasks + gid]
#define ReadI32(member, ptr) ((global int*)ptr)[OffsetOf(member) / (uint)sizeof(int) * numTasks + gid]
#define WriteI32(member, ptr, value) ReadI32(member, ptr) = value
#define ReadU32(member, ptr) ((global uint*)ptr)[OffsetOf(member) / (uint)sizeof(uint) * numTasks + gid]
#define WriteU32(member, ptr, value) ReadU32(member, ptr) = value
#define ReadFloat2(member, ptr) (float2)(ReadF32Vec(member, 0, ptr), ReadF32Vec(member, 1, ptr))
#define ReadFloat3(member, ptr) (float3)(ReadF32Vec(member, 0, ptr), ReadF32Vec(member, 1, ptr), ReadF32Vec(member, 2, ptr))
#define WriteFloat2(member, ptr, value) ReadF32Vec(member, 0, ptr) = value.x; ReadF32Vec(member, 1, ptr) = value.y;
#define WriteFloat3(member, ptr, value) ReadF32Vec(member, 0, ptr) = value.x; ReadF32Vec(member, 1, ptr) = value.y; ReadF32Vec(member, 2, ptr) = value.z;
#endif

typedef struct
{
    float3 orig;
    float3 dir;
} Ray;

typedef struct
{
    float3 P;    // 16B
    float3 Kd;   // 16B
    cl_float R;  // 4B (padded to 16B?)
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
    cl_int type;   // BXDF type, defined in bxdf.cl
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
    float2 uvTex;
    cl_float t;
    cl_int i; // index of hit triangle, -1 by default
    cl_int areaLightHit;
    cl_int matId; // index of hit material
} Hit;

#define EMPTY_HIT(tmax) { (float3)(0.0f), (float3)(0.0f), (float2)(0.0f), tmax, -1, 0, -1 }

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
    cl_uint useAreaLight;
    cl_float envMapStrength;
    cl_uint flashlight;
    cl_uint maxBounces;
    cl_uint sampleImpl;    // use implicit light source sampling
    cl_uint sampleExpl;    // use next event estimation
    cl_float worldRadius;
} RenderParams;


// Microkernel state structs
typedef enum
{
    MK_RT_NEXT_VERTEX = 0,
    MK_SAMPLE_BSDF = 1,
    MK_SAMPLE_LIGHT_IMPL = 2,
    MK_HIT_NOTHING = 3,
    MK_SPLAT_SAMPLE = 4,
    MK_GENERATE_CAMERA_RAY = 5,
    MK_DONE = 6
} PathPhase;


// State for a single path in the microkernel paradigm.
// Stored in SoA format, hence no structs (Laine 2013: 'Megakernels Considered Harmful')
// Laine: 212 bytes per path
typedef struct
{
    // Path state:
    float3 orig;     // path segment origin
    float3 dir;      // path segment direction
    float3 T;        // throughput
    float3 Ei;       // irradiance
    // Last hit:
    float3 P;
    float3 N;
    float2 uvTex;
    // Path state:
    PathPhase phase;
    cl_float pdf;
	cl_float lastPdfW; // prev. brdf pdf, for MIS
    cl_uint pathLen; // number of segments in path
    cl_uint seed;
	cl_uint lastSpecular; // prevents NEE
	cl_int samples;  // counted per pixel
    // Last hit:
    cl_float t;
    cl_int i;        // index of hit triangle, -1 by default
    cl_int areaLightHit;
    cl_int matId;    // index of hit material
} GPUTaskState;

typedef struct
{
    cl_uint primaryRays;
    cl_uint extensionRays;
    cl_uint shadowRays;
    cl_uint samples;
} RenderStats;