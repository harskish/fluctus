#ifndef CL_BXDF_IDEAL_DIELECTRIC
#define CL_BXDF_IDEAL_DIELECTRIC

#include "utils.cl"
#include "fresnel.cl"

// Ideal dielectric
// Check PBRT 8.2 (p.516)

float3 sampleIdealDielectric(Hit *hit, Material *material, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 *dirOut, float *pdfW, uint *randSeed)
{
	float raylen = length(dirIn);
	float3 bsdf = (float3)(1.0f, 1.0f, 1.0f);

	float cosI = dot(normalize(-dirIn), hit->N);
	float n1 = 1.0f, n2 = material->Ni;
	if (backface) swap_m(n1, n2, float); // inside of material
	float eta = n1 / n2;
	float cosT2 = 1.0f - eta * eta * (1.0f - cosI * cosI);

	float fr = fresnelDielectric(cosI, n1, n2);
	if (rand(randSeed) < fr)
	{
		// Reflection
		*dirOut = raylen * reflect(normalize(dirIn), hit->N);
	}
	else
	{
		// Refraction
		*dirOut = raylen * refract(normalize(dirIn), hit->N, eta);
		bsdf *= eta * eta; // eta^2 applied in case of radiance transport (16.1.3)
		
		// Simulate absorption
		float3 Ks = matGetFloat3(material->Ks, hit->uvTex, material->map_Ks, textures, texData);
		bsdf *= Ks;
	}

	// (1-fr) or (fr) in pdf and BSDF cancel out
	*pdfW = 1.0f;

	// PBRT eq. 8.8
	// cosTh of geometry term needs to be cancelled out
	float cosO = dot(normalize(*dirOut), hit->N);
	return bsdf / cosO;
}

// BSDF (dirac delta) is non-zero with zero probability for two given directions
float3 evalIdealDielectric()
{
	return (float3)(0.0f, 0.0f, 0.0f);
}

// Probability of supplying a correct refl/refr direction pair is zero
float pdfIdealDielectric()
{
	return 0.0f;
}

#endif