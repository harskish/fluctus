#include "geom.h"

kernel void trace(global float *out, global Sphere *scene, const uint width, const uint height, const float sin2) {
        
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if(x >= width || y >= height) return;

    float px = (float)x / width;
    float py = (float)y / height;

    /*
    Sphere s; // = { 1.0f, (float4)(0.0f, 0.0f, 0.0f, 0.0f), (float4)(1.0f, 0.0f, 0.0f, 0.0f) };
    if(x == 0)
    {
        s = scene[0];
        printf("Sphere 0: { %.2f, { %.2f, %.2f, %.2f, %.2f }, { %.2f, %.2f, %.2f, %.2f } }\n", s.R, s.pos.x, s.pos.y, s.pos.z, s.pos.w, s.Kd.x, s.Kd.y, s.Kd.z, s.Kd.w);
        s = scene[1];
        printf("Sphere 1: { %.2f, { %.2f, %.2f, %.2f, %.2f }, { %.2f, %.2f, %.2f, %.2f } }\n", s.R, s.pos.x, s.pos.y, s.pos.z, s.pos.w, s.Kd.x, s.Kd.y, s.Kd.z, s.Kd.w);
        float4 c = vload4(0, out);
        printf("Pixel 0 color: { %.2f, %.2f, %.2f, %.2f })\n", c.x, c.y, c.z, c.w);
    }
    */

    float intensity = pow(pow(px - 0.5f, 10) + pow(py - 0.5f, 10), 1.0f / 10.0f);
    float4 pixelColor = sin2 * intensity * (float4)(1.0f, 1.0f - sin2, sin2, 0.0f);

    vstore4(pixelColor, (y * width + x), out); // (value, offset, ptr)
}