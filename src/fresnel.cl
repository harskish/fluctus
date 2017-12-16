#ifndef CL_FRESNEL
#define CL_FRESNEL

// Fresnel for dielectrics, unpolarized light (PBRT p.519)
inline float fresnelDielectric(float cosThI, float etaI, float etaT)
{
	float sinThetaI = sqrt(max(0.0f, 1.0f - cosThI * cosThI));
	float sinThetaT = etaI / etaT * sinThetaI;
	float cosThetaT = sqrt(max(0.0f, 1.0f - sinThetaT * sinThetaT));

	if (sinThetaT >= 1.0f)
		return 1.0f;

	float parl = ((etaT * cosThI) - (etaI * cosThetaT)) /
				 ((etaT * cosThI) + (etaI * cosThetaT));
	float perp = ((etaI * cosThI) - (etaT * cosThetaT)) /
				 ((etaI * cosThI) + (etaT * cosThetaT));
	
	return 0.5f * (parl * parl + perp * perp);
}

// Schlick's approximation for dielectrics
inline float schlickDielectric(float cosThI, float etaI, float etaT)
{
	float eta = etaI / etaT;
	float sinThetaI = sqrt(max(0.0f, 1.0f - cosThI * cosThI));

	if (eta * sinThetaI >= 1.0f)
		return 1.0f;
	
	float r0 = ((1.0f - eta) * (1.0f - eta)) / ((1.0f + eta) * (1.0f + eta));
	float c = 1.0f - fabs(cosThI);
	return r0 + (1.0f - r0) * native_powr(c, 5.0f);
}

// Fresnel for conductor-dielectric interface
// PBRT equations [8.3, 8.4] (src/core/reflection.cpp)
// Imaginary part = absorption coefficient
// The parameters are wavelength-dependent, here given for R/G/B
inline float3 fresnelConductor(float cosThetaI, float3 etai, float3 etat, float3 k)
{
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    float3 eta = etat / etai;
    float3 etak = k / etai;

    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1.0f - cosThetaI2;
    float3 eta2 = eta * eta;
    float3 etak2 = etak * etak;

    float3 t0 = eta2 - etak2 - sinThetaI2;
    float3 a2plusb2 = sqrt(t0 * t0 + 4.0f * eta2 * etak2);
    float3 t1 = a2plusb2 + cosThetaI2;
    float3 a = sqrt(0.5f * (a2plusb2 + t0));
    float3 t2 = 2.0f * cosThetaI * a;
    float3 Rs = (t1 - t2) / (t1 + t2);

    float3 t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    float3 t4 = t2 * sinThetaI2;
    float3 Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (Rp + Rs);
}

#endif