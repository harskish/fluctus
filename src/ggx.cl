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
	float denom = M_PI_F * nDotMSq * nDotMSq * (aSq + tanSq) * (aSq + tanSq);
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
	float F = (mat->Ni > 1.0f) ? fresnelDielectric(iDotN, 1.0f, mat->Ni) : 1.0f;

	// Evaluate BSDF (eq. 20)
	float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	float D = ggxD(alpha, hit->N, H);
	float G = ggxG(alpha, dirIn, *dirOut, hit->N, H);
	float den = (4.0f * iDotN * oDotN);
	return (den != 0.0f) ? (Ks * F * G * D / den) : (float3)(0.0f, 0.0f, 0.0f);
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
	float F = (mat->Ni > 1.0f) ? fresnelDielectric(iDotN, 1.0f, mat->Ni) : 1.0f;

	// Evaluate BSDF (eq. 20)
	float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
	float D = ggxD(alpha, hit->N, H);
	float G = ggxG(alpha, dirIn, dirOut, hit->N, H);
	float den = (4.0f * iDotN * oDotN);
	return (den != 0.0f) ? (Ks * F * G * D / den) : (float3)(0.0f, 0.0f, 0.0f);
}

float pdfGGXReflect(Hit *hit, Material *mat, float3 dirIn, float3 dirOut)
{
	dirIn *= -1;
	float alpha = toRoughness(mat->Ns);
	float3 H = normalize(dirIn + dirOut);
	return ggxPdfReflect(alpha, dirOut, hit->N, H);
}

// See eq. 24, 17
float ggxPdfRefract(float alpha, float etaI, float etaO, float3 dirIn, float3 dirOut, float3 N, float3 H)
{
	float nDotH = fabs(dot(N, H));
	float iDotH = fabs(dot(dirIn, H));
	float oDotH = fabs(dot(dirOut, H));
	float sqrtJInv = etaI * iDotH + etaO * oDotH; // squre root of inverse of H to dirOut Jacobian (eq. 17)
	return sqrtJInv == 0.0f ? 0.0f : ggxD(alpha, N, H) * nDotH * oDotH * etaO * etaO / (sqrtJInv * sqrtJInv);
}

float3 sampleGGXRefract(Hit *hit, Material *mat, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 *dirOut, float *pdfW, uint *seed)
{
	// Setup parameters
	dirIn *= -1; // points outwards
	float raylen = length(dirIn);
	float alpha = toRoughness(mat->Ns);
	float etaI = 1.0f, etaO = mat->Ni; // assume air-dielectric interface
	if (backface) swap_m(etaI, etaO, float);
	float iDotN = dot(normalize(dirIn), hit->N);
	
	// Importance sample GGX lobe
	float3 H = ggxSampleLobe(alpha, dirIn, hit->N, seed);

	// Importance sample refrection and refraction based on Fresnel term
	float F = fresnelDielectric(iDotN, etaI, etaO);
	if (rand(seed) < F)
	{
		// Reflection
		*dirOut = raylen * reflect(normalize(-dirIn), H);
		*pdfW = ggxPdfReflect(alpha, *dirOut, hit->N, H); // TODO: pdf for total internal reflection?

		// Evaluate BSDF
		// White reflections (Ks only for sim. absorption)
		float oDotN = dot(*dirOut, hit->N);
		float D = ggxD(alpha, hit->N, H);
		float G = ggxG(alpha, dirIn, *dirOut, hit->N, H);
		float den = (4.0f * iDotN * oDotN);
		return (den != 0.0f) ? (F * G * D / den) : 0.0f;
	}
	else
	{
		// Refraction
		float eta = etaI / etaO;
		*dirOut = raylen * refract(normalize(-dirIn), hit->N, eta);

		// Recalculate H (eq. 16)
		H = normalize(-(dirIn * etaI + *dirOut * etaO));
		*pdfW = ggxPdfRefract(alpha, etaI, etaO, dirIn, *dirOut, ((backface) ? -hit->N : hit->N), H);

		// eta^2 applied in case of radiance transport (16.1.3)
		const bool lightTracing = false;
		float3 bsdf = (lightTracing) ? (float3)(1.0f) : (float3)(eta * eta);
		
		// Simulate absorption
		float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
		bsdf *= Ks;

		float iDotH = fabs(dot(normalize(dirIn), H));
		float oDotH = fabs(dot(*dirOut, H));
		float oDotN = dot(*dirOut, hit->N);

		// Focus term (eq. 21)
		float focusTermDenom = iDotN * oDotN * (etaI * iDotH + etaO * oDotH) * (etaI * iDotH + etaO * oDotH);
		if (focusTermDenom == 0.0f)
			return (float3)(0.0f);

		float focusTerm = etaO * etaO * iDotH * oDotH / focusTermDenom;
	
		// Evaluate BSDF
		float D = ggxD(alpha, ((backface) ? -hit->N : hit->N), H);
		float G = ggxG(alpha, dirIn, *dirOut, ((backface) ? -hit->N : hit->N), H);
		return (1.0f - F) * bsdf * D * G * focusTerm;
	}
}

float3 evalGGXRefract(Hit *hit, Material *mat, bool backface, global TexDescriptor *textures, global uchar *texData, float3 dirIn, float3 dirOut)
{
	// Setup parameters
	dirIn *= -1; // points outwards
	float raylen = length(dirIn);
	float alpha = toRoughness(mat->Ns);
	float etaI = 1.0f, etaO = mat->Ni; // assume air-dielectric interface
	if (backface) swap_m(etaI, etaO, float);
	float iDotN = dot(normalize(dirIn), hit->N);
	float oDotN = dot(normalize(dirOut), hit->N);
	float F = fresnelDielectric(iDotN, etaI, etaO);
	
	if (!backface)
	{
		// Reflected ray
		// Evaluate BSDF
		// White reflections (Ks only for sim. absorption)
		float3 H = normalize(dirIn + dirOut);
		float D = ggxD(alpha, hit->N, H);
		float G = ggxG(alpha, dirIn, dirOut, hit->N, H);
		float den = (4.0f * iDotN * oDotN);
		return (den != 0.0f) ? (F * G * D / den) : (float3)(0.0f, 0.0f, 0.0f);
	}
	else
	{
		// Refracted ray
		float3 H = normalize(-(dirIn * etaI + dirOut * etaO));
		float eta = etaI / etaO;

		// eta^2 applied in case of radiance transport (16.1.3)
		const bool lightTracing = false;
		float3 bsdf = (lightTracing) ? (float3)(1.0f) : (float3)(eta * eta);
		
		// Simulate absorption
		float3 Ks = matGetFloat3(mat->Ks, hit->uvTex, mat->map_Ks, textures, texData);
		bsdf *= Ks;

		float iDotH = fabs(dot(normalize(dirIn), H));
		float oDotH = fabs(dot(normalize(dirOut), H));

		// Focus term (eq. 21)
		float focusTermDenom = iDotN * oDotN * (etaI * iDotH + etaO * oDotH) * (etaI * iDotH + etaO * oDotH);
		if (focusTermDenom == 0.0f)
			return (float3)(0.0f, 0.0f, 0.0f);

		float focusTerm = etaO * etaO * iDotH * oDotH / focusTermDenom;
	
		// Evaluate BSDF
		float D = ggxD(alpha, -hit->N, H);
		float G = ggxG(alpha, dirIn, dirOut, -hit->N, H);
		return (1.0f - F) * bsdf * D * G * focusTerm;
	}
}

float pdfGGXRefract(Hit *hit, Material *mat, bool backface, float3 dirIn, float3 dirOut)
{
	dirIn *= -1;
	float alpha = toRoughness(mat->Ns);
	float etaI = 1.0f, etaO = mat->Ni;
	
	if (!backface)
	{
		// Reflected
		float3 H = normalize(dirIn + dirOut);
		return ggxPdfReflect(alpha, dirOut, hit->N, H);
	}
	else
	{
		// Refracted ray
		swap_m(etaI, etaO, float);
		float3 H = normalize(-(dirIn * etaI + dirOut * etaO));
		return ggxPdfRefract(alpha, etaI, etaO, dirIn, dirOut, -hit->N, H);
	}
}




#endif