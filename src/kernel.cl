#include "geom.h"

#define printVec(title, v) printf("%s: { %.4f, %.4f, %.4f, %.4f }\n", title, (v).x, (v).y, (v).z, (v).w)
#define dbg(expr) if(get_global_id(0) == 0 && get_global_id(1) == 0) { expr; }
//#define dbg(expr) if(false) { expr; }

inline bool sphereIntersect(Ray *r, global Sphere *s, float *t)
{
    float t0, t1;
    float radius2 = s->R * s->R;

    // Geometric solution
    float4 L = s->P - r->orig;
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

    // NDC-space, [0,1]x[0,1]
    float NDCx = (x + 0.5f) / params->width;
    float NDCy = (y + 0.5f) / params->height;

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
    float4 rayTarget = params->camera.pos + params->camera.right * SCRx + params->camera.up * SCRy + params->camera.dir;
    float4 rayDirection = normalize(rayTarget - params->camera.pos);

    // Construct camera ray
    Ray r = { params->camera.pos, rayDirection };
    return r;
}

inline void calcNormalSphere(global Sphere *scene, Hit *hit)
{
    hit->N = normalize(hit->P - (scene + hit->i)->P);
}

// Will be replaced with a BVH in the future...
// The ray length encodes the maximum intersection distance!
inline Hit raycast(Ray *r, float tMax, global Sphere *scene, global RenderParams *params)
{
    Hit hit = { (float4)(0.0f), (float4)(0.0f), tMax, -1 };

    for(uint i = 0; i < params->n_objects; i++)
    {
        float t;
        bool found = sphereIntersect(r, &(scene[i]), &t);
        if(found && t < hit.t)
        {
            hit.t = t;
            hit.i = i;
        }
    }

    // Done once
    calcNormalSphere(scene, &hit);
    hit.P = r->orig + hit.t * r->dir;

    return hit;
}

inline float4 whittedShading(Hit *hit, global Sphere *scene, global Light *lights, global RenderParams *params)
{
    dbg(printVec("Hit->P", hit->P));
    dbg(printVec("Hit->N", hit->N));

    float4 res = (float4)(0.0f);
    float4 lifted = hit->P + 1e-3f * hit->N;
    dbg(printVec("Lifted", lifted));
    float4 V = params->camera.pos - hit->P;

    // Point light assumed for now
    for(uint i = 0; i < params->n_lights; i++)
    {
        float4 L = lights[i].pos - hit->P;
        dbg(printVec("Light pos", lights[i].pos));

        Ray shadowRay = { lifted, normalize(L) };
        Hit shdw = raycast(&shadowRay, length(L), scene, params);
        float visibility = (shdw.i == -1) ? 0.0f : 1.0f; // early exits useless on GPU

        // Blinn-Phong

        // Testing material:
        float4 Ks = (float4)(1.0f, 1.0f, 1.0f, 0.0f);
        float glossiness = 0.4f; // probably not the right name...

        float4 H = normalize(L + V);

        dbg(printVec("Kd", scene[hit->i].Kd));

        float4 diffuse = scene[hit->i].Kd * max(0.0f, dot(L, hit->N));
        float4 specular = Ks * pow(max(0.0f, dot(hit->N, H)), 1.0f / glossiness);

        if(dot(hit->N, L) < 0) specular = (float4)(0.0f);

        float dist = fast_length(L);
        float falloff = 1.0f / (dist * dist + 1e-5f);

        float4 color = lights[i].intensity * falloff * (diffuse + specular);
        color.w = 0.0f;
        res += color;
    }

    dbg(printVec("Final color", res));

    return res;
}

kernel void trace(global float *out, global Sphere *scene, global Light *lights, global RenderParams *params)
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
    Hit hit = raycast(&r, FLT_MAX, scene, params);

    float4 pixelColor = (hit.i == -1) ? (float4)(0.0f) : whittedShading(&hit, scene, lights, params);
    //float4 pixelColor = (hit.i != -1) ? scene[hit.i].Kd : (float4)(0.0f);

    //float4 prev = vload4((y * width + x), out);
    //float4 newCol = 0.005f * pixelColor + prev;

    dbg(printf("\n"));

    vstore4(pixelColor, (y * params->width + x), out); // (value, offset, ptr)
}
