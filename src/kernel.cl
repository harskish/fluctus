#include "geom.h"

kernel void trace(__global float *out, const uint width, const uint height, const float sin2) {
        
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if(x >= width || y >= height) return;

    float px = (float)x / width;
    float py = (float)y / height;

    //Sphere s = { 1.0f, (float4)(0.0f, 0.0f, 0.0f, 0.0f), (float4)(1.0f, 0.0f, 0.0f, 0.0f) };

    float intensity = pow(pow(px - 0.5f, 10) + pow(py - 0.5f, 10), 1.0f / 10.0f);
    float4 pixelColor = sin2 * intensity * (float4)(1.0f, 1.0f - sin2, sin2, 0.0f);

    vstore4(pixelColor, (y * width + x), out);
}