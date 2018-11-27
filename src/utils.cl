#ifndef CL_UTILS
#define CL_UTILS

#include "geom.h"
#include "random.cl"
#include "ptx_asm.cl"

#define printVec3(title, v) printf("%s: { %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z)
#define printVec4(title, v) printf("%s: { %.4f, %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z, (v).w)

#define swap_m(a, b, t) { t tmp = a; a = b; b = tmp; }

inline void swap(float *a, float *b)
{
  float tmp = *b;
  *b = *a;
  *a = tmp;
}

inline bool isZero(float3 v)
{
	return (v.x == 0.0f && v.y == 0 && v.z == 0);
}

inline float3 lerp(float u, float v, float3 v1, float3 v2, float3 v3)
{
    return (1.0f - u - v) * v1 + u * v2 + v * v3;
}

inline float3 reflect(float3 dir, float3 n) // dir normalized?
{
    return dir - 2.0f * dot(dir, n) * n;
}

// Wi points inwards
inline float3 refract(float3 wi, float3 n, float eta)
{
	float iDotN = dot(-wi, n);
	float sin2ThetaI = max(0.0f, 1.0f - iDotN * iDotN);
	float sin2ThetaT = eta * eta * sin2ThetaI;
	float cosThetaT = sqrt(max(0.0f, 1.0f - sin2ThetaT));
	return wi * eta + n * (eta * iDotN - cosThetaT);
}

inline void calcNormalSphere(global Sphere *scene, Hit *hit)
{
    hit->N = normalize(hit->P - (scene +hit->i)->P);
}

inline void makeOrthoBasis(const float3 N, float3 *a, float3 *b)
{
	if(N.x != N.y || N.x != N.z)
		*a = (float3)(N.z-N.y, N.x-N.z, N.y-N.x);
	else
		*a = (float3)(N.z-N.y, N.x+N.z, -N.y-N.x);

	*a = normalize(*a);
	*b = cross(N, *a);
}

// Return a sample through the center of the Nth subpixel
inline float2 sampleRegular(int n, int dim)
{
    float d = 1.0f / dim;
	int x = n % dim;
	int y = n / dim;

    float xm1 = d * (x + 0.5f);
    float ym1 = d * (y + 0.5f);

    return (float2)(xm1, ym1);
}

// http://mathworld.wolfram.com/DiskPointPicking.html
inline float2 uniformSampleDisk(uint *seed)
{
    float sqrt_r = sqrt(rand(seed));
    float th = M_2PI_F * rand(seed);
    return (float2)(sqrt_r * cos(th), sqrt_r * sin(th));
}


inline float3 cosSampleHemisphere(float3 n, uint *seed, float *p)
{
    float r1 = 2.0f * M_PI_F * rand(seed);
    float r2 = rand(seed);
    float r2s = sqrt(r2);

    float3 w = n;

    float3 u;
    if (fabs(w.x) > 0.1f) {
        float3 a = (float3)(0.0f, 1.0f, 0.0f);
        u = cross(a, w);
    }
    else {
        float3 a = (float3)(1.0f, 0.0f, 0.0f);
        u = cross(a, w);
    }
    u = normalize(u);

    float3 v = cross(w, u);
    
    u *= (cos(r1) * r2s);
    v *= (sin(r1) * r2s);
    w *= (sqrt(1 - r2));

    float3 dir = u + v + w;
    float costh = dot(n, dir);
    *p = costh / M_PI_F; //pdf
	return dir;
}

inline int2 getTexelCoords(float2 uv, uint width, uint height)
{
    uv.x *= width;
    uv.y *= height;
    int tx = ((int)(floor(uv.x)) % width + width) % width;
    int ty = ((int)(floor(uv.y)) % height + height) % height;

    return clamp((int2)(tx + uv.x - floor(uv.x), ty + uv.y - floor(uv.y)), (int2)(0), (int2)(width - 1, height - 1));
    //return Vec2f(tx + uv.x - floor(uv.x), ty + uv.y - floor(uv.y)).clamp(Vec2f(0), Vec2f(size)-Vec2f(1));
}

inline float3 readTexture(float2 uvTex, TexDescriptor tex, global uchar *data)
{
    int2 coords = getTexelCoords(uvTex, tex.width, tex.height);
    global uchar *pix = data + tex.offset + coords.x * 4 + coords.y * tex.width * 4;
    float3 c = (float3)(*(pix + 0), *(pix + 1), *(pix + 2));
	c /= 255.0f;
	
    return c;
}

// Performs gamma correction
inline float3 matGetAlbedo(float3 fallback, float2 uv, int idx, global TexDescriptor *textures, global uchar *texData)
{
	float3 val = (idx != -1) ? readTexture(uv, textures[idx], texData) : fallback;
	val.xyz = pow(val.xyz, 2.2f);
    return val;
}

inline float3 matGetFloat3(float3 fallback, float2 uv, int idx, global TexDescriptor *textures, global uchar *texData)
{
	return (idx != -1) ? readTexture(uv, textures[idx], texData) : fallback;
}

// Construct tangent space, convert normal into world space
inline float3 tangentSpaceNormal(Hit hit, global Triangle *tris, const Material mat, global TexDescriptor *textures, global uchar *texData)
{
    if (mat.map_N == -1)
        return hit.N;
    
    const float3 defaultVal = (float3)(0.5f, 0.5f, 1.0f); // flat surface
    float3 texNormal = matGetFloat3(defaultVal, hit.uvTex, mat.map_N, textures, texData);
    texNormal = 2.0f * texNormal - (float3)(1.0f, 1.0f, 1.0f);
    
    Triangle t = tris[hit.i];
    
    float3 e1 = t.v1.p - t.v0.p;
    float3 e2 = t.v2.p - t.v0.p;
    float3 t1 = t.v1.t - t.v0.t;
    float3 t2 = t.v2.t - t.v0.t;

    // Detect invalid normal map
    float det = (t1.x * t2.y - t1.y * t2.x);
    if (det == 0.0)
        return hit.N;

    // Compute T, B using inverse of [t1.x t1.y; t2.x t2.y]
    float invDet = 1.0f / det;
    float3 T = normalize(invDet * (e1 * t2.y - e2 * t1.y));
    float3 B = normalize(invDet * (e2 * t1.x - e1 * t2.x));

    // Expanded matrix multiply M * hit.N
    float3 N;
    N.x = T.x*texNormal.x + B.x*texNormal.y + hit.N.x*texNormal.z;
    N.y = T.y*texNormal.x + B.y*texNormal.y + hit.N.y*texNormal.z;
    N.z = T.z*texNormal.x + B.z*texNormal.y + hit.N.z*texNormal.z;

    return normalize(N);
}

// Read all material parameters at once
// Can alternatlvely be read separately in bsdf sampling/eval code
inline void getMaterialParameters(Hit hit, global Triangle *tris, global Material *materials, global uchar *texData, global TexDescriptor *textures, float3 *Kd, float3 *N, float3 *Ks, float *refr)
{
    const Material mat = materials[hit.matId];

	*Kd = matGetAlbedo(mat.Kd, hit.uvTex, mat.map_Kd, textures, texData);
	*Ks = matGetFloat3(mat.Ks, hit.uvTex, mat.map_Ks, textures, texData);
    *N = tangentSpaceNormal(hit, tris, mat, textures, texData);
    *refr = mat.Ni;
}

// Product area measure => solid angle measure
inline float pdfAtoW(const float pdf, const float dist, const float cosine)
{
    return pdf * (dist * dist) / fabs(cosine);
}

inline void writeHitSoA(Hit hit, global GPUTaskState *tasks, const size_t gid, const uint numTasks)
{
	WriteFloat3(P, tasks, hit.P);
	WriteFloat3(N, tasks, hit.N);
	WriteFloat2(uvTex, tasks, hit.uvTex);
	WriteF32(t, tasks, hit.t);
	WriteI32(i, tasks, hit.i);
	WriteI32(areaLightHit, tasks, hit.areaLightHit);
	WriteI32(matId, tasks, hit.matId);
}

inline Hit readHitSoA(global GPUTaskState *tasks, const size_t gid, const uint numTasks)
{
	Hit hit;
	hit.P = ReadFloat3(P, tasks);
	hit.N = ReadFloat3(N, tasks);
	hit.uvTex = ReadFloat2(uvTex, tasks);
	hit.t = ReadF32(t, tasks);
	hit.i = ReadI32(i, tasks);
	hit.areaLightHit = ReadI32(areaLightHit, tasks);
	hit.matId = ReadI32(matId, tasks);
	return hit;
}

inline void sampleAreaLight(AreaLight light, float *pdf, float3 *p, uint *seed)
{
	*pdf = 1.0f / (4.0f * light.size.x * light.size.y);
	*p = light.pos;
	float r1 = 2.0f * rand(seed) - 1.0f;
	float r2 = 2.0f * rand(seed) - 1.0f;
	*p += r1 * light.size.x * light.right;
	*p += r2 * light.size.y * light.up;
}

// sRGB luminance
inline float luminance(float3 v)
{
	return 0.212671f * v.x + 0.715160f * v.y + 0.072169f * v.z;
}

// OpenCL has no native atomic floats
// https://devtalk.nvidia.com/default/topic/458062/atomicadd-float-float-atomicmul-float-float-/
inline void atomic_add_float(volatile __global float* addr, float value)
{
    float old = value;
    while ((old = atomic_xchg(addr, atomic_xchg(addr, 0.0f) + old)) != 0.0f);
}

inline void atomic_add_float_v2(volatile __global float* addr, float value)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + value;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
            expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

inline void atomic_add_float3(volatile __global float3* ptr, float3 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomic_add_float(p + 0, value.x);
    atomic_add_float(p + 1, value.y);
    atomic_add_float(p + 2, value.z);
}

inline void atomic_add_float4(volatile __global float4* ptr, float4 value)
{
    volatile __global float* p = (volatile __global float*)ptr;
    atomic_add_float(p + 0, value.x);
    atomic_add_float(p + 1, value.y);
    atomic_add_float(p + 2, value.z);
    atomic_add_float(p + 3, value.w);
}

inline void normal_add_float3(__global float* ptr, float3 value)
{
    float4 prev = vload4(0, ptr);
    float4 sum = prev + (float4)(value, 0.0f);
    vstore4(sum, 0, ptr);
}

inline void normal_add_float4(__global float* ptr, float4 value)
{
    float4 prev = vload4(0, ptr);
    float4 sum = prev + value;
    vstore4(sum, 0, ptr);
}

inline void add_float3(__global float* ptr, float3 value)
{
#ifdef FLT_FLOAT_ATOMICS
    return atomic_add_float3(ptr, value);
#else
    return normal_add_float3(ptr, value);
#endif
}

inline void add_float4(__global float* ptr, float4 value)
{
#ifdef FLT_FLOAT_ATOMICS
    return atomic_add_float4(ptr, value);
#else
    return normal_add_float4(ptr, value);
#endif
}

inline float3 mulMat3x3(const float3 r1, const float3 r2, const float3 r3, const float3 v)
{
    return (float3)(dot(r1, v), dot(r2, v), dot(r3, v));
}

inline float4 mulMat4x4(const float4 r1, const float4 r2, const float4 r3, const float4 r4, const float4 v)
{
    return (float4)(dot(r1, v), dot(r2, v), dot(r3, v), dot(r4, v));
}


// Don't use within an if clause - activemask catches threads which have
// exited (returned), not those that aren't participating in the current branch.
// If within a branch, evaluate the mask with ballot and use atomicAggInc directly.
inline uint atomicIncAll(global uint* ctr)
{
#ifdef NVIDIA
    return atomicAggInc(ctr, activemask());
#else
    return atomic_inc(ctr);
#endif
}

// Place inside an if clause, and also provide a matching mask
inline uint atomicIncMasked(global uint* ctr, const uint mask)
{
#ifdef NVIDIA
    return atomicAggInc(ctr, mask);
#else
    (void)mask;
    return atomic_inc(ctr);
#endif
}

// More efficient implementation when no return value is needed
inline void atomicIncCounter(global uint* ctr)
{
#ifdef NVIDIA
    const uint increment = popcnt(activemask());
    if (laneid() == 0)
        atomic_add(ctr, increment);
#else
    atomic_inc(ctr);
#endif
}

#endif