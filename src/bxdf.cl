#ifndef CL_BXDF
#define CL_BXDF

#include "bxdf_types.h"
#include "diffuse.cl"
#include "ideal_dielectric.cl"
#include "ideal_conductor.cl"

// Placeholder
typedef float3 Spectrum;

// Note: In these formulas, dirIn points towards the surface, not away from it!

// Generate outgoing direction and pdf given invoming direction
Spectrum bxdfSample(
	Hit *hit,
	Material *material,
	bool backface, // indicates that normal has been flipped (affects refraction)
	global TexDescriptor *textures,
	global uchar *texData,
	float3 dirIn,
	float3 *dirOut,
	float *pdfW,
	uint *randSeed)
{
	switch(material->type)
	{
		case BXDF_DIFFUSE:
			return sampleDiffuse(hit, material, textures, texData, dirOut, pdfW, randSeed);
		case BXDF_CONDUCTOR:
		case BXDF_IDEAL_CONDUCTOR:
			return sampleIdealConductor(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
		case BXDF_DIELECTRIC:
		case BXDF_IDEAL_DIELECTRIC:
			return sampleIdealDielectric(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
		case BXDF_EMISSIVE:
			return (float3)(1.0f, 1.0f, 1.0f);
	}

	return (float3)(0.0f, 0.0f, 0.0f);
}

// Evaluate bxdf value given invoming and outgoing directions.
Spectrum bxdfEval(
	Hit *hit,
	Material *material,
	global TexDescriptor *textures,
	global uchar *texData,
	float3 dirIn,
	float3 dirOut)
{
	switch(material->type)
	{
		case BXDF_DIFFUSE:
			return evalDiffuse(hit, material, textures, texData, dirIn, dirOut);
		case BXDF_CONDUCTOR:
		case BXDF_IDEAL_CONDUCTOR:
			return evalIdealConductor();
		case BXDF_DIELECTRIC:
		case BXDF_IDEAL_DIELECTRIC:
			return evalIdealDielectric();
		case BXDF_EMISSIVE:
			return (float3)(1.0f, 1.0f, 1.0f);
	}
	
	return (float3)(0.0f, 0.0f, 0.0f);
}

// Get pdf given incoming, outgoing directions (mostly for MIS)
float bxdfPdf(
	Hit *hit,
	Material *material,
	global TexDescriptor *textures,
	global uchar *texData,
	float3 dirIn,
	float3 dirOut)
{
	switch(material->type)
	{
		case BXDF_DIFFUSE:
			return pdfDiffuse(hit, dirOut);
		case BXDF_CONDUCTOR:
		case BXDF_IDEAL_CONDUCTOR:
			return pdfIdealConductor();
		case BXDF_DIELECTRIC:
		case BXDF_IDEAL_DIELECTRIC:
			return pdfIdealDielectric();
		case BXDF_EMISSIVE:
			return 0.0f;
	}

	return 0.0f;
}

#endif