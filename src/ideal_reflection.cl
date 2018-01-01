#ifndef CL_BXDF_IDEAL_CONDUCTOR
#define CL_BXDF_IDEAL_CONDUCTOR

#include "utils.cl"

// Ideal conductive specular reflection (mirror)
// Check PBRT 8.2 (p.516)

float3 sampleIdealReflection(Hit *hit, Material *material, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 *dirOut, float *pdfW, uint *randSeed)
{
	float len = length(dirIn);
	*dirOut = len * reflect(normalize(dirIn), hit->N);
	*pdfW = 1.0f;

	// Fresnel ignored in ideal spacular case

	// PBRT eq. 8.8
	// cosTh of geometry term needs to be cancelled out
	float3 ks = matGetFloat3(material->Ks, hit->uvTex, material->map_Ks, textures, texData);
	float cosO = dot(normalize(*dirOut), hit->N);
	return (cosO != 0.0f) ? ks / cosO : 0.0f;
}

// BSDF (dirac delta) is non-zero with zero probability for two given directions
float3 evalIdealReflection()
{
	return (float3)(0.0f, 0.0f, 0.0f);
}

// Probability of supplying a correct refl/refr direction pair is zero
float pdfIdealReflection()
{
	return 0.0f;
}

#endif