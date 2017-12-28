#ifndef CL_BXDF_GGX
#define CL_BXDF_GGX

#include "utils.cl"

/* 
 * GGX microfacet with Smith shadowing-masking function
 * Dielectric fresnel used for now (until spectral rendering is supported)
 */

// Use proposed mapping from phong exponent (from mtl) to Beckmann alpha
inline float toRoughness(float shininess)
{
	return sqrt(2.0f / (2.0f + shininess));
}

// Importence sample lobe, eq. 35, 36
float3 ggxSampleLobe(float alpha, float3 dirIn, float3 N, uint *seed)
{
	// Create orthonormal basis
	float3 X, Y, Z = N;
	makeOrthoBasis(Z, &X, &Y);

	// Sample lobe in local coordinates
	float2 rnd = (float2)(rand(seed), rand(seed));
	float theta = atan2(alpha*sqrt(rnd.x), sqrt(1 - rnd.x));
	float phi = M_2PI_F * rnd.y;

	// Return halfway vector in global coordinate frame
	float sinTheta = native_sin(theta);
	float cosTheta = native_cos(theta);
	float sinPhi = native_sin(phi);
	float cosPhi = native_cos(phi);
    return normalize(X * sinTheta * cosPhi + Y * sinTheta * sinPhi + Z * cosTheta);
}

// Unidirectional shadowing-masking function
// G1(v, m) = 2 / 1 + sqrt( 1 + a^2 * tan^2v )  (eg. 34)
float ggxG1(float alpha, float3 v, float3 n, float3 m)
{
	float mDotV = dot(m,v);
	float nDotV = dot(n,v);
	
	// Check sidedness agreement (eq. 7)
	if (nDotV * mDotV <= 0.0f)
		return 0.0f;

	// tan^2 = sin^2 / cos^2
	float cosThSq = nDotV * nDotV;
	float tanSq = (cosThSq > 0.0f) ? ((1.0f - cosThSq) / cosThSq) : 0.0f;
    return 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tanSq));
}

// Smith approximation for G
float ggxG(float alpha, float3 dirIn, float3 dirOut, float3 n, float3 m)
{
	// Return product of the unidirectional masking functions
	return ggxG1(alpha, dirIn, n, m) * ggxG1(alpha, dirOut, n, m);
}

// Microfacet distribution function (GTR2)
// D(m) = a^2 / PI * cosT^4 * (a^2 + tanT^2)^2  (eq. 33)
float ggxD(float alpha, float3 n, float3 m)
{
	float nDotM = dot(n, m);

	if (nDotM <= 0.0f)
		return 0.0f;

	// tan^2 = sin^2 / cos^2
	float nDotMSq = nDotM * nDotM;
	float tanSq = nDotM != 0.0f ? ((1.0f - nDotMSq) / nDotMSq) : 0.0f;

	float aSq = alpha * alpha;
	float denom = M_PI * nDotMSq * nDotMSq * (aSq + tanSq) * (aSq + tanSq);
	return denom > 0.0f ? (aSq / denom) : 0.0f;
}

// See eq. 24, 14
float ggxPdfReflect(float alpha, float3 dirOut, float3 N, float3 H)
{
	float nDotH = fabs(dot(N, H));
	float oDotH = fabs(dot(dirOut, H));
	float jInv = 4.0f * oDotH; // inverse of H to dirOut Jacobian (eq. 14)
	return jInv == 0.0f ? 0.0f : ggxD(alpha, N, H) * nDotH / jInv;
}

float3 sampleGGXReflect(Hit *hit, Material *mat, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 *dirOut, float *pdfW, uint *seed)
{
	// Setup parameters
	dirIn *= -1; // points outwards
	float alpha = toRoughness(mat->Ns);
	
	// Importance sample GGX lobe
	float3 H = ggxSampleLobe(alpha, dirIn, hit->N, seed);
	*dirOut = reflect(-dirIn, H);

	// Output pdf
	*pdfW = ggxPdfReflect(alpha, *dirOut, hit->N, H);

	// TODO: Fresnel should be applied implicitly in case of layered material
	float iDotN = dot(dirIn, hit->N);
	float oDotN = dot(*dirOut, hit->N);
	float F = 1.0f; //(mat->Ni > 0.0f) ? fresnelDielectric(iDotN, 1.0f, mat->Ni) : 1.0f;

	// Evaluate BSDF (eq. 20)
	float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	float D = ggxD(alpha, hit->N, H);
	float G = ggxG(alpha, dirIn, *dirOut, hit->N, H);
	float den = (4.0f * iDotN * oDotN);
	return (den != 0.0f) ? (Ks * F * G * D / den) : 0.0f;
}

float3 evalGGXReflect(Hit *hit, Material *mat, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 dirOut)
{
	// Setup parameters
	dirIn *= -1; // points outwards
	float alpha = toRoughness(mat->Ns);
	
	// Calculate halfway vector
	float3 H = normalize(dirIn+ dirOut);

	// TODO: Fresnel should be applied implicitly in case of layered material
	float iDotN = dot(dirIn, hit->N);
	float oDotN = dot(dirOut, hit->N);
	float F = 1.0f; // (mat->Ni > 0.0f) ? fresnelDielectric(iDotN, 1.0f, mat->Ni) : 1.0f;

	// Evaluate BSDF (eq. 20)
	float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	float D = ggxD(alpha, hit->N, H);
	float G = ggxG(alpha, dirIn, dirOut, hit->N, H);
	float den = (4.0f * iDotN * oDotN);
	return (den != 0.0f) ? (Ks * F * G * D / den) : 0.0f;
}

float pdfGGXReflect(Hit *hit, Material *mat, float3 dirIn, float3 dirOut)
{
	dirIn *= -1;
	float alpha = toRoughness(mat->Ns);
	float3 H = normalize(dirIn+ dirOut);
	return ggxPdfReflect(alpha, dirOut, hit->N, H);
}

#endif