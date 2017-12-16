#ifndef CL_BXDF_DIFFUSE
#define CL_BXDF_DIFFUSE

#include "utils.cl"

// Ideal lambertian reflectance
//	 brdf = Kd / PI
//	 pdf = costh / PI
float3 sampleDiffuse(Hit *hit, Material *mat, global TexDescriptor *textures, global uchar *texData, float3 *dirOut, float *pdfW, uint *randSeed)
{
	*dirOut = cosSampleHemisphere(hit->N, randSeed, pdfW);
	float3 Kd = matGetFloat3(mat->Kd, hit->uvTex, mat->map_Kd, textures, texData);
	return Kd * M_INV_PI;
}

float3 evalDiffuse(Hit *hit, Material *mat, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 dirOut)
{
	float3 Kd = matGetFloat3(mat->Kd, hit->uvTex, mat->map_Kd, textures, texData);
	return Kd * M_INV_PI;
}

float pdfDiffuse(Hit *hit, float3 dirOut)
{
	// TODO: shading normal?
	return dot(hit->N, dirOut) * M_INV_PI;
}


#endif