#ifndef CL_BXDF_GLOSSY
#define CL_BXDF_GLOSSY

#include "utils.cl"

// Lambertian reflector with varnish coat (microfacet)
// Parameters modeled after LuxRender's glossy material (http://www.luxrender.net/wiki/LuxRender_Materials_Glossy):
//	* Ni used for fresnel
//  * If no Ks is specified, it is calculated based on Ni
//  * If no Ni is specified, it is calculated based on Ks

inline float3 etaToKs(float eta)
{
	float r = (eta > 0.0f) ? ((eta - 1) / (eta + 1)) : 0.0f;
	return (float3)(r * r);
}

inline float ksToEta(float3 Ks)
{
	float k = clamp((Ks.x + Ks.y + Ks.z) / 3.0f, 0.0f, 0.99f);
	return (sqrt(k) + 1) / (1 - sqrt(k));
}

float3 sampleGlossy(Hit *hit, Material *mat, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 *dirOut, float *pdfW, uint *seed)
{
	// Backside => only diffuse
	//if (backface)
	//	return sampleDiffuse(hit, mat, textures, texData, dirOut, pdfW, seed);

	// Check Ks and Ni
	// TODO: ok to just modify? (yes, not a global variable...)
	Material m = *mat;
	m.Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	m.Ni = (mat->Ni > 0.0f) ? mat->Ni : ksToEta(m.Ks);
	if (isZero(m.Ks)) m.Ks = etaToKs(m.Ni);

	// Choose between materials based on fresnel
	float cosTh = dot(normalize(-dirIn), hit->N);
	float F = fresnelDielectric(cosTh, 1.0f, m.Ni);
	
	float basePdf, coatingPdf;
	float3 baseBrdf, coatingBrdf;
	if (rand(seed) < F)
	{
		// Importance sample specular
		coatingBrdf = sampleGGXReflect(hit, &m, textures, texData, dirIn, dirOut, &coatingPdf, seed);
		baseBrdf = evalDiffuse(hit, &m, textures, texData, dirIn, *dirOut);
		basePdf = pdfDiffuse(hit, *dirOut);
	}
	else
	{
		// Importance sample diffuse
		baseBrdf = sampleDiffuse(hit, &m, textures, texData, dirOut, &basePdf, seed);
		coatingBrdf = evalGGXReflect(hit, &m, textures, texData, dirIn, *dirOut);
		coatingPdf = pdfGGXReflect(hit, &m, dirIn, *dirOut);
	}

	if (dot(hit->N, *dirOut) < 1e-5f)
		return (float3)(0.0f);

	*pdfW = (1 - F) * basePdf + F * coatingPdf;
	return (baseBrdf * (1 - F) + coatingBrdf); // coatingBrdf contains F
}

float3 evalGlossy(Hit *hit, Material *mat, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 dirOut)
{
	// Backside => only diffuse
	//if (backface)
	//	return evalDiffuse(hit, mat, textures, texData, dirIn, dirOut);

	// Check Ks and Ni
	// TODO: ok to just modify? (yes, not a global variable...)
	Material m = *mat;
	m.Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	m.Ni = (mat->Ni > 0.0f) ? mat->Ni : ksToEta(m.Ks);
	if (length(m.Ks) == 0.0f) m.Ks = etaToKs(m.Ni);

	float3 baseBrdf = evalDiffuse(hit, &m, textures, texData, dirIn, dirOut);
	float3 coatingBrdf = evalGGXReflect(hit, &m, textures, texData, dirIn, dirOut);

	float cosTh = dot(normalize(-dirIn), hit->N);
	float F = fresnelDielectric(cosTh, 1.0f, m.Ni);
	return (baseBrdf * (1 - F) + coatingBrdf); // coatingBrdf contains F
}

float pdfGlossy(Hit *hit, Material *mat, global TexDescriptor *textures, global uchar *texData, bool backface, float3 dirIn, float3 dirOut)
{
	// Backside => only diffuse
	//if (backface)
	//	return pdfDiffuse(hit, dirOut);

	float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	float Ni = (mat->Ni > 0.0f) ? mat->Ni : ksToEta(Ks);

	float basePdf = pdfDiffuse(hit, dirOut);
	float coatingPdf = pdfGGXReflect(hit, mat, dirIn, dirOut); // Ni and Ks don't affect

	float cosTh = dot(normalize(-dirIn), hit->N);
	float F = fresnelDielectric(cosTh, 1.0f, Ni);
	return (1 - F) * basePdf + F * coatingPdf;
}


#endif