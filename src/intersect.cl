#ifndef CL_INTERSECT
#define CL_INTERSECT

#include "geom.h"
#include "utils.cl"

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
        // TODO: Use swap!
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
inline bool intersectTriangleLocal(Ray *r, Triangle *tri, float *tres)
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

    *tres = t;
    return true;
}

// For debugging: check if ray hits area light (for drawing a white square)
inline bool rayHitsLight(Ray *r, global RenderParams *params, float *tres)
{
    float3 tl = (float3)(params->areaLight.pos + params->areaLight.size.x * params->areaLight.right + params->areaLight.size.y * params->areaLight.up);
    float3 tr = (float3)(params->areaLight.pos - params->areaLight.size.x * params->areaLight.right + params->areaLight.size.y * params->areaLight.up);
    float3 bl = (float3)(params->areaLight.pos + params->areaLight.size.x * params->areaLight.right - params->areaLight.size.y * params->areaLight.up);
    float3 br = (float3)(params->areaLight.pos - params->areaLight.size.x * params->areaLight.right - params->areaLight.size.y * params->areaLight.up);

    Triangle T1;
    T1.v0.p = tl;
    T1.v1.p = bl;
    T1.v2.p = br;
    if (intersectTriangleLocal(r, &T1, tres)) return true;

    Triangle T2;
    T2.v0.p = tl;
    T2.v1.p = br;
    T2.v2.p = tr;
    if (intersectTriangleLocal(r, &T2, tres)) return true;

    return false;
}

#endif