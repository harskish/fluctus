#include "geom.h"

#define PI 3.14159265358979323846
#define toRad(deg) (deg * PI / 180)
#define dbg(expr) if(get_global_id(0) == 0 && get_global_id(1) == 0) { expr; }

inline bool sphereIntersect(Ray *r, global Sphere *s, float *t)
{
    float t0, t1;
    float radius2 = s->R * s->R;

    // Geometric solution
    float4 L = s->pos - r->orig;
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

inline Ray getCameraRay(const uint x, const uint y, global RenderParams *params)
{
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction
    float4 camRight = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
    float4 camUp =    (float4)(0.0f, 1.0f, 0.0f, 0.0f);

    // NDC-space, [0,1]x[0,1]
    float NDCx = (x + 0.5f) / params->width;
    float NDCy = (y + 0.5f) / params->height;

    // Screen space, [-1,1]x[-1,1]
    float SCRx = 2.0f * NDCx - 1.0f;
    float SCRy = 2.0f * NDCy - 1.0f;

    // Aspect ratio fix applied horizontally
    SCRx *= (float)params->width / params->height;

    // Screen space coordinates scaled based on fov
    float scale = tan(toRad(params->camera.fov / 2)); // half of width
    SCRx *= scale;
    SCRy *= scale;

    // World space coorinates of pixel
    float4 rayTarget = params->camera.pos + camRight * SCRx + camUp * SCRy + params->camera.dir;
    float4 rayDirection = normalize(rayTarget - params->camera.pos);

    // Construct camera ray
    Ray r = { params->camera.pos, rayDirection };
    return r;
}

kernel void trace(global float *out, global Sphere *scene, global RenderParams *params)
{
    const uint x = get_global_id(0); // left to right
    const uint y = get_global_id(1); // bottom to top

    if(x >= params->width || y >= params->height) return;

    /*
    float4 p = params->camera.pos;
    float4 d = params->camera.dir;
    dbg(printf("Camera pos: { %.2f, %.2f, %.2f, %.2f }\n", p.x, p.y, p.z, p.w));
    dbg(printf("Camera dir: { %.2f, %.2f, %.2f, %.2f }\n", d.x, d.y, d.z, d.w));
    dbg(printf("Width: %d\n", params->width));
    */

    Ray r = getCameraRay(x, y, params);

    // Check intersections
    float tmin = FLT_MAX;
    int imin;
    for(uint i = 0; i < params->n_objects; i++)
    {
        float t;
        bool hit = sphereIntersect(&r, &(scene[i]), &t);
        if(hit && t < tmin)
        {
            tmin = t;
            imin = i;
        }
    }

    float4 pixelColor = (tmin != FLT_MAX) ? scene[imin].Kd : float4(0.0f);

    //float4 prev = vload4((y * width + x), out);
    //float4 newCol = 0.005f * pixelColor + prev;

    vstore4(pixelColor, (y * params->width + x), out); // (value, offset, ptr)
}