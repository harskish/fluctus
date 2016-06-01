kernel void trace(__global float *out, const uint width, const uint height) {
        
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    if(x >= width || y >= height) return;

    float intensity = (float)(x + y) / (width + height);

    float4 pixelColor = intensity * (float4)(1.0f, 0.0f, 0.0f, 0.0f);
    
    if(x == 799 && y == 599)
    {
        printf("[%d,%d]: Setting color [%.2f,%.2f,%.2f]\n", x, y, pixelColor.x, pixelColor.y, pixelColor.z);
    }

    vstore4(pixelColor, (y * width + x), out);
}