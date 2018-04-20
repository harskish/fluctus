#ifndef CL_BXDF_PARTIAL
#define CL_BXDF_PARTIAL

#include "bxdf_types.h"
#include "diffuse.cl"
#include "ideal_dielectric.cl"
#include "ideal_reflection.cl"
#include "ggx.cl"
#include "glossy.cl"

/*
    Specialized bxdf sampling code
    Used in wavefront material evaluation kernels to turn off certain code paths
*/

typedef float3 Spectrum;

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
#ifdef BXDF_USE_DIFFUSE
		case BXDF_DIFFUSE:
			return sampleDiffuse(hit, material, textures, texData, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_GLOSSY
		case BXDF_GLOSSY:
			return sampleGlossy(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_GGX_ROUGH_REFLECTION
		case BXDF_GGX_ROUGH_REFLECTION:
			return sampleGGXReflect(hit, material, textures, texData, dirIn, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_IDEAL_REFLECTION
		case BXDF_IDEAL_REFLECTION:
			return sampleIdealReflection(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_GGX_ROUGH_DIELECTRIC
		case BXDF_GGX_ROUGH_DIELECTRIC:
			return sampleGGXRefract(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_IDEAL_DIELECTRIC
		case BXDF_IDEAL_DIELECTRIC:
			return sampleIdealDielectric(hit, material, backface, textures, texData, dirIn, dirOut, pdfW, randSeed);
#endif
#ifdef BXDF_USE_EMISSIVE
		case BXDF_EMISSIVE:
			return (float3)(1.0f, 1.0f, 1.0f);
#endif
	}

	return (float3)(0.0f, 0.0f, 0.0f);
}

// Evaluate bxdf value given invoming and outgoing directions.
Spectrum bxdfEval(
	Hit *hit,
	Material *material,
	bool backface,
	global TexDescriptor *textures,
	global uchar *texData,
	float3 dirIn,
	float3 dirOut)
{
	switch(material->type)
	{
#ifdef BXDF_USE_DIFFUSE
		case BXDF_DIFFUSE:
			return evalDiffuse(hit, material, textures, texData, dirIn, dirOut);
#endif
#ifdef BXDF_USE_GLOSSY
		case BXDF_GLOSSY:
			return evalGlossy(hit, material, backface, textures, texData, dirIn, dirOut);
#endif
#ifdef BXDF_USE_GGX_ROUGH_REFLECTION
		case BXDF_GGX_ROUGH_REFLECTION:
			return evalGGXReflect(hit, material, textures, texData, dirIn, dirOut);
#endif
#ifdef BXDF_USE_IDEAL_REFLECTION
		case BXDF_IDEAL_REFLECTION:
			return evalIdealReflection();
#endif
#ifdef BXDF_USE_GGX_ROUGH_DIELECTRIC
		case BXDF_GGX_ROUGH_DIELECTRIC:
			return evalGGXRefract(hit, material, backface, textures, texData, dirIn, dirOut);
#endif
#ifdef BXDF_USE_IDEAL_DIELECTRIC
		case BXDF_IDEAL_DIELECTRIC:
			return evalIdealDielectric();
#endif
#ifdef BXDF_USE_EMISSIVE
		case BXDF_EMISSIVE:
			return (float3)(1.0f, 1.0f, 1.0f);
#endif
	}
	
	return (float3)(0.0f, 0.0f, 0.0f);
}

// Get pdf given incoming, outgoing directions (mainly for MIS)
float bxdfPdf(
	Hit *hit,
	Material *material,
	bool backface,
	global TexDescriptor *textures,
	global uchar *texData,
	float3 dirIn,
	float3 dirOut)
{
	switch(material->type)
	{
#ifdef BXDF_USE_DIFFUSE
		case BXDF_DIFFUSE:
			return pdfDiffuse(hit, dirOut);
#endif
#ifdef BXDF_USE_GLOSSY
		case BXDF_GLOSSY:
			return pdfGlossy(hit, material, textures, texData, backface, dirIn, dirOut);
#endif
#ifdef BXDF_USE_GGX_ROUGH_REFLECTION
		case BXDF_GGX_ROUGH_REFLECTION:
			return pdfGGXReflect(hit, material, dirIn, dirOut);
#endif
#ifdef BXDF_USE_IDEAL_REFLECTION
		case BXDF_IDEAL_REFLECTION:
			return pdfIdealReflection();
#endif
#ifdef BXDF_USE_GGX_ROUGH_DIELECTRIC
		case BXDF_GGX_ROUGH_DIELECTRIC:
			return pdfGGXRefract(hit, material, backface, dirIn, dirOut);
#endif
#ifdef BXDF_USE_IDEAL_DIELECTRIC
		case BXDF_IDEAL_DIELECTRIC:
			return pdfIdealDielectric();
#endif
#ifdef BXDF_USE_EMISSIVE
		case BXDF_EMISSIVE:
			return 0.0f;
#endif
	}

	return 0.0f;
}

#endif