#include "geom.h"
#include "bvh.cl"
#include "utils.cl"
#include "intersect.cl"

// Microkernel for direct (explicit) light sampling
// State changes:
//   MK_SAMPLE_LIGHT_EXPL => MK_SPLAT_SAMPLE || MK_SAMPLE_BSDF
kernel void sampleLightExplicit(
    global GPUTaskState *tasks,
    global Material *materials,
    global uchar *texData,
    global TexDescriptor *textures,
    global Triangle *tris,
    global GPUNode *nodes,
    global uint *indices,
    global RenderParams *params,
    global RenderStats *stats,
    uint numTasks)
{
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks);
    uint seed = ReadU32(seed, tasks);

    if (gid >= limit)
        return;

    // Read the path state
    global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
    if (*phase != MK_SAMPLE_LIGHT_EXPL)
        return;

    const float3 rayOrig = ReadFloat3(orig, tasks);
    const float3 rayDir = ReadFloat3(dir, tasks);
    Ray r = { rayOrig, rayDir };

    // Read hit from path state
    Hit hit = readHitSoA(tasks, gid, numTasks);

    // Read BSDF
    float3 N, Kd, Ks;
    float Ni;
    getMaterialParameters(hit, tris, materials, texData, textures, &Kd, &N, &Ks, &Ni);

    // Fix backside hits
    if (dot(N, r.dir) > 0.0f) N *= -1;
	float3 orig = hit.P - 1e-3f * r.dir;  // avoid self-shadowing

	bool lastSpecular = ReadU32(lastSpecular, tasks);
	if (params->sampleExpl && !lastSpecular)
	{
		// Sample light source
		float directPdfA;
		float3 posL;
		AreaLight areaLight = params->areaLight;
		sampleAreaLight(areaLight, &directPdfA, &posL, &seed);

		// Shadow ray
		float3 L = posL - orig;
		float lenL = length(L);
		L = normalize(L);
		Ray rLight = { orig, L };

		// TODO: BAD! Collect all shadow ray casts together (in queue, i.e. buffer of gids + atomic counter)!
		bool occluded = bvh_occluded(&rLight, &lenL, tris, nodes, indices);
		atomic_inc(&stats->shadowRays);

		// Calculate direct lighting
		float cosLight = max(dot(params->areaLight.N, -L), 0.0f);
		if (!occluded && cosLight > 1e-6f)
		{
			const float lightPickProb = 1.0f;
			const float3 brdf = Kd / M_PI_F; // Kd = reflectivity/albedo
			float cosTh = max(0.0f, dot(normalize(-r.dir), N)); // cos at surface
			float directPdfW = pdfAtoW(directPdfA, lenL, cosLight); // 'how small area light looks'
			float bsdfPdfW = max(0.0f, cosTh / M_PI_F);

			float weight = 1.0f;
			if (params->sampleImpl)
			{
				weight = (directPdfW * lightPickProb) / (directPdfW * lightPickProb + bsdfPdfW);
			}

			const float3 T = ReadFloat3(T, tasks);
			const float prob = ReadF32(pdf, tasks);
			const float3 contrib = brdf * T * params->areaLight.E * weight * cosTh / (lightPickProb * directPdfW * prob);
			const float3 newEi = ReadFloat3(Ei, tasks) + contrib;
			WriteFloat3(Ei, tasks, newEi);
		}
	}

    // Check path termination
    uint len = ReadU32(pathLen, tasks);
    if (len + 1 >= params->maxBounces)
    {
        *phase = MK_SPLAT_SAMPLE;
    }
    else
    {
        // Generate continuation ray
        float pdf, costh;
        sampleHemisphere(&(hit.P), &N, &costh, &seed, &pdf, &r.dir); // r.dir updated

        float3 brdf = Kd / M_PI_F;
        float3 newT = ReadFloat3(T, tasks) * brdf * costh;
        float newPdf = ReadF32(pdf, tasks) * pdf;
        
        // Update path state
        WriteFloat3(T, tasks, newT);
        WriteFloat3(orig, tasks, orig);
        WriteFloat3(dir, tasks, r.dir);
        WriteF32(pdf, tasks, newPdf);
		WriteF32(lastPdfW, tasks, pdf);

        *phase = MK_RT_NEXT_VERTEX; //MK_SAMPLE_BSDF?
    }

    // Update RNG seed
    WriteU32(seed, tasks, seed);
}
