#include "utils.cl"

/* Utilities for evaluating environment map Li's and pdf's */
/* HDRIs are typically stored in linear space, no gamma correction needed! */

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
    float r = clamp(dir.y / length(dir), -1.0f, 1.0f);
    float v = acos(r) / M_PI_F;

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
	global const float *pdfTable;
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
    float pdf_uv = ctx.pdfTable[uvInd];

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

// Get pdf of sampling 'direction', used in MIS
float envMapPdf(int width, int height, global float *pdfTable, float3 direction)
{
    float2 uv = directionToUV(direction);
    float sinTh = sin(uv.y * M_PI_F);
    
    if (sinTh == 0.0f)
        return 0.0f;

    int iu = min((int)floor(uv.x * width), width - 1);
    int iv = min((int)floor(uv.y * height), height - 1);

    return pdfTable[iv * width + iu] / (M_2PI_F * M_PI_F * sinTh);
}