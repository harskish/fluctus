#include "utils.cl"

/* Utilities for evaluating environment map Li's and pdf's */

// integer UVs
constant sampler_t samplerInt = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// floating point UVs in range [0,1]
constant sampler_t samplerFloat = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// Mapping from http://gl.ict.usc.edu/Data/HighResProbes/
// U mapped from [0,2] to [0,1]
inline float2 directionToUV(float3 dir)
{
    if (dir.x == 0.0f && dir.y == 0.0f && dir.z == 0.0f)
        return (float2)(0.0f, 0.0f);

    float u = 1.0f + atan2(dir.x, -dir.z) / M_PI_F;
    float v = acos(dir.y / length(dir)) / M_PI_F;

    return (float2)(u * 0.5f, v);
}

// Mapping from http://gl.ict.usc.edu/Data/HighResProbes/
// U mapped from [0,1] to [0,2]
inline float3 UVToDirection(float u, float v)
{
    float phi = v * M_PI_F;
    float theta = (u * 2.0f - 1.0f) * M_PI_F;
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);
    float sinTh = sin(theta);
    float cosTh = cos(theta);
    return (float3)(sinPhi*sinTh, cosPhi, -sinPhi*cosTh);
}

inline float3 evalEnvMapDir(read_only image2d_t envMap, float3 dir)
{
    float2 uv = directionToUV(dir);
    return read_imagef(envMap, samplerFloat, uv).xyz;
}

inline float3 evalEnvMapUVfloat(read_only image2d_t envMap, float u, float v)
{
    return read_imagef(envMap, samplerFloat, (float2)(u, v)).xyz;
}

inline float3 evalEnvMapUVint(read_only image2d_t envMap, int u, int v)
{
    return read_imagef(envMap, samplerInt, (int2)(u, v)).xyz;
}

typedef struct
{
    const int width;
    const int height;
    global const float *cdfTable;
    global const float *pdfTable;
	global const float *pdfTable1D;
    global const float *probTable;
    global const int *aliasTable;
} EnvMapContext;

// Uses the Alias Method
inline void sampleEnvMapAlias(float rnd, float3 *L, float *pdfW, EnvMapContext ctx)
{
    const int width = ctx.width;
    const int height = ctx.height;

	// Sample 1D distribution over whole image
	float rand = rnd * width * height;
	int i = min((int)floor(rand), width * height - 1);
    float mProb = ctx.probTable[i];
    int uvInd = (rand - i < mProb) ? i : ctx.aliasTable[i];
    float pdf_uv = ctx.pdfTable1D[uvInd];

    // Compute outgoing dir
	int uInd = uvInd % width;
	int vInd = uvInd / width;
    float u = (float)(uInd + 0.5f) / width;
    float v = (float)(vInd + 0.5f) / height;
    *L = UVToDirection(u, v);

    // Compute pdf
    const float lightPickProb = 1.0f;
    float sinTh = sin(M_PI_F * v);
    float directPdfUV = pdf_uv * lightPickProb;
    if (sinTh != 0.0f)
        *pdfW = directPdfUV / (2.0f * M_PI_F * M_PI_F * sinTh);
    else
        *pdfW = 0.0f;
}

// Uses PBRT's binary search method. Adapted from kernel_light.h in Blender's Cycles renderer.
inline void sampleEnvMapBinarySearch(float2 rnd, float3 *L, float *pdfW, EnvMapContext ctx)
{
    const int width = ctx.width;
    const int height = ctx.height;

    // this is basically std::lower_bound as used by pbrt
    int first = 0;
    int count = height;

    while (count > 0)
    {
        int step = count / 2;
        int middle = first + step;

        float cdf = ctx.cdfTable[middle * (width + 2) + width + 1];
        if (cdf < rnd.y)
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
            count = step;
    }

    int index_v = max(0, first - 1);

    float2 cdf_v = ctx.cdfTable[index_v * (width + 2) + width + 1];
    float pdf_v = ctx.pdfTable[index_v * (width + 2) + width + 1];
    float2 cdf_next_v = ctx.cdfTable[(index_v + 1) * (width + 2) + width + 1];
    float2 cdf_last_v = ctx.cdfTable[height * (width + 2) + width + 1];

    // importance-sampled V direction
    float dv = (rnd.y - cdf_v.y) / (cdf_next_v.y - cdf_v.y);
    float v = (index_v + dv) / height;

    // this is basically std::lower_bound as used by pbrt
    first = 0;
    count = width;
    while (count > 0)
    {
        int step = count / 2;
        int middle = first + step;

        float cdf = ctx.cdfTable[index_v * (width + 2) + middle];
        if (cdf < rnd.x)
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
            count = step;
    }

    int index_u = max(0, first - 1);

    float2 cdf_u = ctx.cdfTable[index_v * (width + 2) + index_u];
    float pdf_u = ctx.pdfTable[index_v * (width + 2) + index_u];
    float2 cdf_next_u = ctx.cdfTable[index_v * (width + 2) + index_u + 1];
    float2 cdf_last_u = ctx.cdfTable[index_v * (width + 2) + width];

    // importance-sampled U direction
    float du = (rnd.x - cdf_u.y) / (cdf_next_u.y - cdf_u.y);
    float u = (index_u + du) / width;

    // compute pdf
    float sin_theta = sin(M_PI_F * v);
    if (sin_theta == 0.0f)
        *pdfW = 0.0f;
    else
        *pdfW = (pdf_u * pdf_v) / (M_2PI_F * M_PI_F * sin_theta);

    // compute direction
    *L = UVToDirection(u, v);
}

// Get pdf of sampling 'direction', used in MIS
float envMapPdf(int width, int height, global float *pdfTable, float3 direction)
{
    float2 uv = directionToUV(direction);
    float sinTh = sin(uv.y * M_PI_F);
    
    if (sinTh == 0.0f)
        return 0.0f;

    int index_u = min((int)floor(uv.x * width), width - 1);
    int index_v = min((int)floor(uv.y * height), height - 1);
    
    float pdfU = pdfTable[index_v * (width + 2) + index_u];
    float pdfV = pdfTable[index_v * (width + 2) + width + 1];

    return (pdfU * pdfV) / (M_2PI_F * M_PI_F * sinTh);
}