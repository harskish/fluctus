#include "geom.h"
#define toRad(deg) (deg * M_PI / 180)

inline bool shereIntersect(Ray *r, global Sphere *s, float *t)
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

kernel void trace(global float *out, global Sphere *scene, const uint width, const uint height, const float sin2)
{        
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if(x >= width || y >= height) return;

    
    // Camera plane is 1 unit away, by convention
    // Camera points in the negative z-direction
    float4 camOrig = (float4)(0.0f, 0.0f, 1.5f + sin2, 0.0f);
    float4 camDir = (float4)(0.0f, 0.0f, -1.0f, 0.0f);
    float4 camRight = (float4)(1.0f, 0.0f, 0.0f, 0.0f);
    float4 camUp = (float4)(0.0f, 1.0f, 0.0f, 0.0f);

    // NDC-space, [0,1]x[0,1]
    float NDCx = (x + 0.5f) / width;
    float NDCy = (y + 0.5f) / height;

    // Screen space, [-1,1]x[-1,1]
    float SCRx = 2.0f * NDCx - 1.0f;
    float SCRy = -2.0f * NDCy + 1.0f;

    // Aspect ratio fix applied horizontally
    SCRx *= (float)width / height;

    // Screen space coordinates scaled based on fov
    const int fov = 90;
    float scale = tan(toRad(fov / 2)); // half of width
    SCRx *= scale;
    SCRy *= scale;

    // World space coorinates of pixel
    float4 rayTarget = camOrig + camRight * SCRx + camUp * SCRy + camDir;
    float4 rayDirection = normalize(rayTarget - camOrig);

    // Construct camera ray
    Ray r = { camOrig, rayDirection };

    // Check intersections
    float tmin = FLT_MAX;
    for(int i = 0; i < 2; i++)
    {
        float t;
        bool hit = shereIntersect(&r, &(scene[i]), &t);
        if(hit && t < tmin)
        {
            tmin = t;
        }
    }
    
    float4 pixelColor = (tmin != FLT_MAX) ? (float4)(1.0f) : float4(0.0f);
    vstore4(pixelColor, (y * width + x), out); // (value, offset, ptr)
}