#include "geom.h"

#define printVec3(title, v) printf("%s: { %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z)
#define printVec4(title, v) printf("%s: { %.4f, %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z, (v).w)
#define dbg(expr) if(get_global_id(0) == 0 && get_global_id(1) == 0) { expr; }
//#define dbg(expr) if(false) { expr; }

// For reading the environment map texture
constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_TRUE | // use UVs directly
    CLK_ADDRESS_CLAMP_TO_EDGE |
    CLK_FILTER_LINEAR;

#define swap_m(a, b, t) { t tmp = a; a = b; b = tmp; }
inline void swap(float *a, float *b)
{
  float tmp = *b;
  *b = *a;
  *a = tmp;
}

inline float3 lerp(float u, float v, float3 v1, float3 v2, float3 v3)
{
    return (1.0f - u - v) * v1 + u * v2 + v * v3;
}

inline float3 reflect(float3 dir, float3 n) // dir normalized?
{
    return dir - 2.0f * dot(dir, n) * n;
}

inline float3 refract(float3 dir, float3 n, float n1, float n2) // n1 = current, n2 = new
{
    float cosI = dot(-normalize(dir), n);
    float cosT = 1.0f - pow(n1 / n2, 2.0f) * (1.0f - pow(cosI, 2.0f));
	float raylen = length(dir);

	if (cosT < 0.0f)
		return raylen * reflect(normalize(dir), n); // Total internal reflection
	else
        return raylen * (normalize(dir) * (n1 / n2) + n * ((n1 / n2) * cosI - sqrt(cosT)));
}

inline bool sphereIntersect(Ray *r, global Sphere *s, float *t)
{
    float t0, t1;
    float radius2 = s->R * s->R;

    // Geometric solution
    float3 L = s->P - r->orig;
    float tca = dot(L, r->dir);
    float d2 = dot(L, L) - tca * tca;
    if (d2 > radius2) return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    if (t0 > t1)
    {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }

    if (t0 < 0)
    {
        t0 = t1;
        if (t0 < 0) return false;
    }

    *t = t0;

    return true;
}

#define NORMAL_X ((float3)(-1, 0, 0))
#define NORMAL_Y ((float3)(0, -1, 0))
#define NORMAL_Z ((float3)(0, 0, -1))

// Assign normal according to face hit
inline bool intersectSlab(Ray *r, global AABB *box, float *tminRet, float *tMaxRet, float3 *N)
{
    float3 n;
    float3 dinv = 1.0f / r->dir;

    // X-axis
    n = NORMAL_X;
    float dinvx = dinv.x;
    float tmin = (box->min.x - r->orig.x) * dinvx;
    float tmax = (box->max.x - r->orig.x) * dinvx;

    if (dinvx < 0)
    {
        swap(&tmin, &tmax);
        n *= -1.0f;
    }

    if (tmax < 0)
    {
        return false;
    }

    *N = n;

    // Y-axis
    n = NORMAL_Y;
    float dinvy = dinv.y;
    float tminy = (box->min.y - r->orig.y) * dinvy;
    float tmaxy = (box->max.y - r->orig.y) * dinvy;

    if (dinvy < 0)
    {
        swap(&tminy, &tmaxy);
        n *= -1.0f;
    }

    if (tmin > tmaxy || tmax < tminy)
    {
        return false;
    }

    if (tminy > tmin)
    {
        tmin = tminy;
        *N = n;
    }

    if (tmaxy < tmax)
    {
        tmax = tmaxy;
    }

    if (tmax < 0)
    {
        return false;
    }

    // Z-axis
    n = NORMAL_Z;
    float dinvz = dinv.z;
    float tminz = (box->min.z - r->orig.z) * dinvz;
    float tmaxz = (box->max.z - r->orig.z) * dinvz;

    if (dinvz < 0)
    {
        swap(&tminz, &tmaxz);
        n *= -1.0f;
    }

    if (tmin > tmaxz || tmax < tminz)
    {
        return false;
    }

    if (tminz > tmin)
    {
        tmin = tminz;
        *N = n;
    }

    if (tmaxz < tmax)
    {
        tmax = tmaxz;
    }

    if (tmax < 0)
    {
        return false;
    }

    // Assign output variables
    *tminRet = tmin;
    *tMaxRet = tmax;
    
    return true;
}

inline bool box_intersect(Ray *r, AABB *box, float *tcurr, float *tminRet)
{
    float3 tmin = (box->min - r->orig) / r->dir;
    float3 tmax = (box->max - r->orig) / r->dir;

    float3 t1 = min(tmin, tmax);
    float3 t2 = max(tmin, tmax);

    float ts = max(t1.x, max(t1.y, t1.z));
    float te = min(t2.x, min(t2.y, t2.z));

    if (te < 0.0f || ts > *tcurr || ts > te) return false;
    *tminRet = max(0.0f, ts);

    return true;
}

// Möller-Trumbore
#define EPSILON 1e-12f
inline bool intersectTriangle(Ray *r, global Triangle *tri, float *tret, float *uret, float *vret)
{
    float3 s1 = tri->v1.p - tri->v0.p;
    float3 s2 = tri->v2.p - tri->v0.p;
    float3 pvec = cross(r->dir, s2); // order matters!
    float det = dot(s1, pvec);

    // miss if det close to 0
    if (fabs(det) < EPSILON) return false;
    float iDet = 1.0f / det;

    float3 tvec = r->orig - tri->v0.p;
    float u = dot(tvec, pvec) * iDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, s1); // order matters!
    float v = dot(r->dir, qvec) * iDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    //float t = s2.dot(qvec) * iDet;
    float t = dot(s2, qvec) * iDet;

    if(t < 0.0f) return false;

    *tret = t;
    *uret = u;
    *vret = v;

    return true;
}

// For drawing the test area light
inline bool intersectTriangleLocal(Ray *r, Triangle *tri)
{
    float3 s1 = tri->v1.p - tri->v0.p;
    float3 s2 = tri->v2.p - tri->v0.p;
    float3 pvec = cross(r->dir, s2);
    float det = dot(s1, pvec);

    // miss if det close to 0
    if (fabs(det) < EPSILON) return false;
    float iDet = 1.0f / det;

    float3 tvec = r->orig - tri->v0.p;
    float u = dot(tvec, pvec) * iDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = cross(tvec, s1);
    float v = dot(r->dir, qvec) * iDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(s2, qvec) * iDet;

    if (t < 0.0f) return false;

    return true;
}

inline float3 evalEnvMap(read_only image2d_t envMap, float3 dir)
{
    int2 envMapDim = get_image_dim(envMap);
    float u = 1.0f + atan2(dir.x, -dir.z) / M_PI_F;
    float v = acos(dir.y) / M_PI_F;

    return read_imagef(envMap, sampler, (float2)(u / 2.0f, v)).xyz;
}

// BVH traversal using simulated stack
inline bool bvh_intersect_stack(Ray *r, Hit *hit, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    float lnear, lfar, rnear, rfar; //AABB limits
    uint closer, farther;

    bool found = false;

    // Stack state
    SimStackNode stack[64];
    int stackptr = 0;

    // Root node
    stack[stackptr].i = 0;
    stack[stackptr].mint = -FLT_MAX;

    while (stackptr >= 0)
    {
        // Next node
        int ni = stack[stackptr].i;
        float tnear = stack[stackptr].mint;
        stackptr--;
        const GPUNode n = nodes[ni];

        // Closer intersection found already
        if (tnear > hit->t)
            continue;

        if (n.nPrims != 0) // Leaf node
        {
            float tmin = FLT_MAX, umin = 0.0f, vmin = 0.0f;
            int imin = -1;
            for (uint i = n.iStart; i < n.iStart + n.nPrims; i++)
            {
                float t, u, v;
                if (intersectTriangle(r, &(tris[indices[i]]), &t, &u, &v))
                {
                    if (t > 0.0f && t < tmin)
                    {
                        imin = i;
                        tmin = t;
                        umin = u;
                        vmin = v;
                    }
                }
            }
            if (imin != -1 && tmin < hit->t)
            {
                found = true;
                hit->i = 0; //indices[imin];
                hit->t = tmin;
                hit->P = r->orig + tmin * r->dir;
                hit->N = lerp(umin, vmin, tris[indices[imin]].v0.n, tris[indices[imin]].v1.n, tris[indices[imin]].v2.n);
            }
        }
        else // Internal node
        {
            float3 N_tmp;
            bool leftWasHit = intersectSlab(r, &(nodes[ni + 1].box), &lnear, &lfar, &N_tmp);
            bool rightWasHit = intersectSlab(r, &(nodes[n.rightChild].box), &rnear, &rfar, &N_tmp);

            if (leftWasHit && rightWasHit)
            {
                closer = ni + 1;
                farther = n.rightChild;

                // Right child was closer -> swap
                if (rnear < lnear)
                {
                    swap_m(lnear, rnear, float);
                    swap_m(lfar, rfar, float);
                    swap_m(closer, farther, uint);
                }

                // Farther node pushed first
                stack[++stackptr] = (SimStackNode){farther, rnear};
                stack[++stackptr] = (SimStackNode){closer, lnear};
            }

            else if (leftWasHit)
            {
                stack[++stackptr] = (SimStackNode){ni + 1, lnear};
            }

            else if (rightWasHit)
            {
                stack[++stackptr] = (SimStackNode){n.rightChild, rnear};
            }
        }
    }

    return found;
}

// x and y include offsets when supersampling
inline Ray getCameraRay(const float2 pos, global RenderParams *params)
{
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction

    // NDC-space, [0,1]x[0,1]
    float NDCx = pos.x / params->width;
    float NDCy = pos.y / params->height;

    // Screen space, [-1,1]x[-1,1]
    float SCRx = 2.0f * NDCx - 1.0f;
    float SCRy = 2.0f * NDCy - 1.0f;

    // Aspect ratio fix applied horizontally
    SCRx *= (float)params->width / params->height;

    // Screen space coordinates scaled based on fov
    float scale = tan(toRad(0.5f * params->camera.fov)); // half of width
    SCRx *= scale;
    SCRy *= scale;

    // World space coorinates of pixel
    float3 rayTarget = params->camera.pos + params->camera.right * SCRx + params->camera.up * SCRy + params->camera.dir;
    float3 rayDirection = normalize(rayTarget - params->camera.pos);

    // Construct camera ray
    Ray r = { params->camera.pos, rayDirection };
    return r;
}

inline void calcNormalSphere(global Sphere *scene, Hit *hit)
{
    hit->N = normalize(hit->P - (scene +hit->i)->P);
}

// Will be replaced with a BVH in the future...
// The ray length encodes the maximum intersection distance!
inline Hit raycast(Ray *r, float tMax, global Sphere *scene, global Triangle *tris, global GPUNode *nodes, global uint *indices, global RenderParams *params)
{
    Hit hit = { (float3)(0.0f), (float3)(0.0f), tMax, -1 };

    // Spheres
    
    for(uint i = 0; i < params->n_objects; i++)
    {
        float t;
        bool found = sphereIntersect(r, &(scene[i]), &t);
        if(found && t < hit.t)
        {
            hit.t = t;
            hit.i = i;
            hit.P = r->orig + hit.t * r->dir;
            calcNormalSphere(scene, &hit);
        }
    }

    // Triangles
    bvh_intersect_stack(r, &hit, tris, nodes, indices);

    return hit;
}

// Blinn-Phong
inline float3 calcLighting(PointLight light, float3 V, Hit *hit, global Sphere *scene, global Triangle *tris, global GPUNode *nodes, global uint *indices, global RenderParams *params)
{
    float3 L = light.pos - hit->P;
    float dist = length(L);
    L = normalize(L);

    Ray shadowRay = { (hit->P + 1e-3f * hit->N), L };
    Hit shdw = raycast(&shadowRay, dist, scene, tris, nodes, indices, params);
    float visibility = (shdw.i == -1) ? 1.0f : 0.0f; // early exits useless on GPU

    // Testing material:
    float3 Ks = (float3)(1.0f); // fraction of light reflected, per color channel
    float glossiness = 0.025f; // probably not the right name...

    float3 H = normalize(L + V);
    float3 diffuse = scene[hit->i].Kd * max(0.0f, dot(L, hit->N));
    float3 specular = Ks * pow(max(0.0f, dot(hit->N, H)), 1.0f / glossiness);

    if(dot(hit->N, L) < 0) specular = (float3)(0.0f);

    float falloff = 1.0f / (dist * dist + 1e-5f);
    return visibility * light.E * falloff * (diffuse + specular);
}

inline float3 whittedShading(Hit *hit, global Sphere *scene, global Triangle *tris, global GPUNode *nodes, global uint *indices, global PointLight *lights, global RenderParams *params)
{
    float3 V = normalize(params->camera.pos - hit->P); // P to eye
    float vDotN = dot(V, hit->N);
    if(vDotN < 0)
    {
        hit->N *= -1.0f;
    }

    float3 res = (float3)(0.0f);

    // Point light assumed for now
    for(uint i = 0; i < params->n_lights; i++)
    {
        PointLight light = lights[i];
        res += calcLighting(light, V, hit, scene, tris, nodes, indices, params);
    }

    // Optional light at camera pos
    if(params->flashlight)
    {
        float3 lPos = params->camera.pos + 0.1f * params->camera.dir;
        PointLight flashlight = { (float3)(10.0f), lPos };
        res += calcLighting(flashlight, V, hit, scene, tris, nodes, indices, params);
    }

    return res;
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

// http://www.burtleburtle.net/bob/hash/integer.html
uint hash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

// Seed is modified (can be used in next iteration)
inline float rand(uint *seed)
{
    *seed = hash(*seed);
    return (float)(*seed) * (1.0f / 4294967296.0f); // 1.0f / 2^32
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

// For debugging: check if ray hits area light (for drawing a white square)
inline bool rayHitsLight(Ray *r, global RenderParams *params)
{
    float3 tl = (float3)(params->areaLight.pos + params->areaLight.size.x * params->areaLight.right + params->areaLight.size.y * params->areaLight.up);
    float3 tr = (float3)(params->areaLight.pos - params->areaLight.size.x * params->areaLight.right + params->areaLight.size.y * params->areaLight.up);
    float3 bl = (float3)(params->areaLight.pos + params->areaLight.size.x * params->areaLight.right - params->areaLight.size.y * params->areaLight.up);
    float3 br = (float3)(params->areaLight.pos - params->areaLight.size.x * params->areaLight.right - params->areaLight.size.y * params->areaLight.up);

    Triangle T1;
    T1.v0.p = tl;
    T1.v1.p = bl;
    T1.v2.p = br;
    if (intersectTriangleLocal(r, &T1)) return true;

    Triangle T2;
    T2.v0.p = tl;
    T2.v1.p = br;
    T2.v2.p = tr;
    if (intersectTriangleLocal(r, &T2)) return true;

    return false;
}


inline void sampleHemisphere(float3 *pos, float3 *n, float *costh, uint *seed, float *p, float3 *dir)
{
    float r1 = 2.0f * M_PI_F * rand(seed);
    float r2 = rand(seed);
    float r2s = sqrt(r2);

    float3 w = *n;

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

    *dir = u + v + w;
    *costh = dot(*n, *dir);
    *p = *costh / M_PI_F; //pdf
}

inline void getTextureParameters(Hit hit, global Sphere *scene, float3 *Kd, float3 *N, float3 *Ks)
{
    // Dummy method for now, should read from textures (if available)
    *Kd = scene[hit.i].Kd;
    *N = hit.N;
    *Ks = (float3)(1.0f);
}

inline float3 reflectionShading(Hit *hit, read_only image2d_t envMap, global RenderParams *params)
{
    float3 V = normalize(params->camera.pos - hit->P); // P to eye
    float vDotN = dot(V, hit->N);
    if(vDotN < 0)
    {
        hit->N *= -1.0f;
    }

    return evalEnvMap(envMap, reflect(-V, hit->N)) - (float3)(0.1f);
}

// Ray tracing!
inline float3 traceRay(float2 pos, global Sphere *scene, global PointLight *lights, global Triangle *tris, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params)
{
    float3 pixelColor = (float3)(0.0f);

    // Supersampling
    const int SAMPLES = 4;
    const int dim = (int)sqrt((float)SAMPLES);
    for (int n = 0; n < SAMPLES; n++)
    {
        pos += sampleRegular(n, dim);
        Ray r = getCameraRay(pos, params);
        Hit hit = raycast(&r, FLT_MAX, scene, tris, nodes, indices, params);

        if (hit.i > -1)
        {
            // Whitted shading
            //pixelColor = whittedShading(&hit, scene, tris, nodes, indices, lights, params);

            // Reflections + environment map
            pixelColor += reflectionShading(&hit, envMap, params);

            // Depth shading
            //pixelColor = (float3)(hit.t / 8.0f);

            // Intersection shading
            // pixelColor = scene[hit.i].Kd;
        }
        else if (params->useEnvMap)
        {
            // Ambient lighting
            pixelColor += evalEnvMap(envMap, r.dir);
        }
    }

    pixelColor /= (float)SAMPLES;
    return pixelColor;
}

// Path tracing!
inline float3 tracePath(float2 pos, uint iter, global Sphere *scene, global PointLight *lights, global Triangle *tris, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params)
{
    float3 Ei = (float3)(0.0f);         // Irradiance
    float3 throughput = (float3)(1.0f); // BRDF
    float prob = 1.0f;                  // PDF

    Ray r = getCameraRay(pos, params);
    Hit hit = raycast(&r, FLT_MAX, scene, tris, nodes, indices, params);
    if (hit.i < 0) return Ei;

    AreaLight areaLight = params->areaLight;
    float3 emission = areaLight.E;
    float3 nLight = areaLight.N;

    // State
    /* LordCRC: move seed to local store?? */
    uint seed = get_global_id(1) * params->width + get_global_id(0) + iter * params->width * params->height; // unique for each pixel
    const int MAX_BOUNCES = 4;
    float3 dir = r.dir; // updated at each bounce
    int i = 0;

    // TEST: show white area light on screen
    if (rayHitsLight(&r, params))
    {
        return (float3)(1.0f);
    }
    
    while(i < MAX_BOUNCES && prob > 0.0f)
    {
        float3 Kd, Ks, n; // fetched from texture/material
        getTextureParameters(hit, scene, &Kd, &n, &Ks);

        // Backside of triangle hit
        bool backside = (dot(n, dir) > 0.0f);
        if (backside) {
            n *= -1;
        }

        /* REFLECTION / REFRACTION */
        {
            // ...
        }


        /* SHADING */

        // If triangle is emissive and this is the first ray (very direct light) -> add emission to radiance
        // ...no emissive triangles in the current code...

        // Sample light source
        float pdf1;
        float3 posL;
        sampleAreaLight(areaLight, &pdf1, &posL, &seed);

        // Geometry term
        float3 orig = hit.P - 1e-3f * dir;  // avoid self-shadowing
        float3 L = posL - orig;
        float lenL = length(L);
        Ray rLight = { orig, L };
        Hit hitLight = raycast(&rLight, lenL, scene, tris, nodes, indices, params);
        
        if(hitLight.i == -1) // light not obstructed
        {
            float3 brdf = Kd / M_PI_F; // Kd = reflectivity/albedo
            float costh = max(dot(n, normalize(L)), 0.0f);
            Ei += brdf * throughput * emission * max(dot(normalize(-L), nLight), 0.0f) * costh / (lenL * lenL) / pdf1 / prob;
        }

        // Indirect
        float pdf, costh;
        sampleHemisphere(&(hit.P), &n, &costh, &seed, &pdf, &dir); // direction updated

        float3 brdf = Kd / M_PI_F;
        throughput *= brdf * costh;
        prob *= pdf;

        Ray rNew = { orig, dir };
        hit = raycast(&rNew, FLT_MAX, scene, tris, nodes, indices, params);

        if (hit.i < 0)
            break;
        
        i++;
    }

    return Ei;
}

/* 
OPENCL MEMORY SPACES:
| OpenCL   | OpenCL keyword | Scope           | CUDA     | CUDA keyword |
|----------+----------------+-----------------+----------+--------------+
| Global   | __global       | Kernel-wide     | Global   |              |
| Constant | __constant     | Kernel-wide     | Constant | __constant__ |
| Local    | __local        | Work-group-wide | Shared   | __shared__   |
| Private  | __private      | Work-item-wide  | Local    |              |
*/
kernel void trace(global float *out, global Sphere *scene, global PointLight *lights, global Triangle *tris, global GPUNode *nodes, global uint *indices, read_only image2d_t envMap, global RenderParams *params, uint iteration)
{
    const uint x = get_global_id(0); // left to right
    const uint y = get_global_id(1); // bottom to top

    if(x >= params->width || y >= params->height) return;

    // Trace
    //float3 pixelColor = traceRay((float2)(x, y), scene, lights, tris, nodes, indices, envMap, params);
    float3 pixelColor = tracePath((float2)(x, y), iteration, scene, lights, tris, nodes, indices, envMap, params);
    
    // Accumulate
    float4 prev = vload4((y * params->width + x), out);
    pixelColor = (pixelColor + iteration * prev.xyz) / (iteration + 1);

    vstore4((float4)(pixelColor, 0.0f), (y * params->width + x), out); // (value, offset, ptr)
}
