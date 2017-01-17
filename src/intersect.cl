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

// Assign normal according to face hit
inline bool intersectAABB(Ray *r, global AABB *box, float *tminRet, float *tMaxRet)
{
    const float3 dinv = native_recip(r->dir);
    const float3 tmp = (box->min - r->orig) * dinv;
    float3 tmaxv = (box->max - r->orig) * dinv;
    const float3 tminv = fmin(tmp, tmaxv);
    tmaxv = fmax(tmp, tmaxv);

    float tmin = fmax( fmax( tminv.x, tminv.y ), tminv.z );
    float tmax = fmin( fmin( tmaxv.x, tmaxv.y ), tmaxv.z );

    if (tmax < 0) return false;
    if (tmin > tmax) return false;

    // Assign output variables
    *tminRet = tmin;
    *tMaxRet = tmax;

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