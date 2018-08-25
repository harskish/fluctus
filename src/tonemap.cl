#pragma once

float3 uc2TonemapFunc(float3 x)
{
    float A = 0.22; // shoulder strength
    float B = 0.30; // linear strength
    float C = 0.10; // linear angle
    float D = 0.20; // toe strength
    float E = 0.01; // toe numerator
    float F = 0.30; // tone denominator

    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

float3 uncharted2Tonemap(float3 x)
{
    float W = 11.2; // linear white point

    float exposureBias = 2.0;
	float3 color = uc2TonemapFunc(exposureBias * x) / uc2TonemapFunc((float3)(W, W, W));
    return color;
}

float3 reinhardTonemap(float3 color)
{
    return color / (1.0f + color);
}