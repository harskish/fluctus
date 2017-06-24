#include "geom.h"
#include "bvh.cl"
#include "utils.cl"
#include "intersect.cl"

kernel void nextVertex(
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
    const uint limit = min(params->width * params->height, numTasks); // TODO: remove need for params, use only numTasks!

    if (gid >= limit)
        return;

    // Read the path state
    global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
    if (*phase != MK_RT_NEXT_VERTEX)
        return;

	// Don't continue paths with near zero pdf => no NaNs (div by zero)
	float pdf = ReadF32(pdf, tasks);
	if (pdf < 1e-6f)
	{
		*phase = MK_SPLAT_SAMPLE;
		return;
	}

	const float3 rayOrig = ReadFloat3(orig, tasks);
    const float3 rayDir = ReadFloat3(dir, tasks);
    Ray r = { rayOrig, rayDir };

    // Trace ray
    Hit hit = EMPTY_HIT(FLT_MAX); // TODO: Max distance?
    bvh_intersect(&r, &hit, tris, nodes, indices);
    if (params->sampleImpl) intersectLight(&hit, &r, params);

    // Write hit to path state
    writeHitSoA(hit, tasks, gid, numTasks);

    // Update render statistics
    global uint *len = &ReadU32(pathLen, tasks);
    atomic_inc((*len == 0) ? &stats->primaryRays : &stats->extensionRays);
    *len += 1;

    // Environment map
    if (hit.i < 0)
    {
        *phase = MK_SPLAT_SAMPLE;
    }
    // Implicit light sample
    else if (hit.areaLightHit)
    {
		float misWeight = 1.0f;
		bool lastSpecular = ReadU32(lastSpecular, tasks);
		if (params->sampleExpl && *len > 1 && !lastSpecular) // not very direct + MIS needed
		{
			const float directPdfA = 1.0f / (4.0f * params->areaLight.size.x * params->areaLight.size.y);
			const float directPdfW = pdfAtoW(directPdfA, length(hit.P - r.orig), dot(normalize(-r.dir), hit.N));
			const float lightPickProb = 1.0f;
			const float lastPdfW = ReadF32(lastPdfW, tasks);
			misWeight = lastPdfW / (lastPdfW + directPdfW * lightPickProb);
		}

		// Pdf (i.e. extension ray pdf = lastPdfW) included in prob
		float3 T = ReadFloat3(T, tasks);
		float3 newEi = ReadFloat3(Ei, tasks) + T * misWeight * params->areaLight.E / pdf;
		WriteFloat3(Ei, tasks, newEi);
        
		// No reflective lights
        *phase = MK_SPLAT_SAMPLE;
    }
    // Scene hit
	else
	{
		// Read BSDF
		float3 N, Kd, Ks;
		float Ni;
		getMaterialParameters(hit, tris, materials, texData, textures, &Kd, &N, &Ks, &Ni);

		bool backside = dot(hit.N, r.dir) > 0.0f;
		if (backside)
		{
			hit.N *= -1.0f;
		}

		// Refract
		if (Ni > 1.0f)
		{
			float3 orig;
			const float EPS_REFR = 1e-5f;
			float cosI = dot(-normalize(r.dir), hit.N);
			uint seed = ReadU32(seed, tasks);

			float n1 = 1.0f, n2 = Ni;
			if (backside) swap_m(n1, n2, float); // inside of material

			float cosT = 1.0f - pow(n1 / n2, 2.0f) * (1.0f - pow(cosI, 2.0f));
			float raylen = length(r.dir);

            global float *pdf = &ReadF32(pdf, tasks);

			// Total internal reflection
			if (cosT < 0.0f)
			{
				orig = hit.P + EPS_REFR * hit.N;
				r.dir = raylen * reflect(normalize(r.dir), hit.N);
			}
			else
			{
				// Fresnel: reflectance for unpolarized light
				cosT = sqrt(cosT);
				float fr = 0.5f * (pow((n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT), 2.0f) + pow((n2 * cosI - n1 * cosT) / (n1 * cosT + n2 * cosI), 2.0f));

				if (rand(&seed) < fr)
				{
					// Reflection
					orig = hit.P + EPS_REFR * hit.N;
					r.dir = raylen * reflect(normalize(r.dir), hit.N);
                    //*pdf *= fr; // TODO: why does this produce weird results?
				}
				else
				{
					// Refraction
					orig = hit.P - EPS_REFR * hit.N;
					r.dir = raylen * (normalize(r.dir) * (n1 / n2) + hit.N * ((n1 / n2) * cosI - cosT));
                    //*pdf *= (1 - fr); // TODO: why does this produce weird results?
				}
			}

            // Simulate absorption by decreasing throughput
            float3 newT = ReadFloat3(T, tasks) * Ks;
            WriteFloat3(T, tasks, newT);

			// Update state
			WriteFloat3(orig, tasks, orig);
			WriteFloat3(dir, tasks, r.dir);
			WriteU32(seed, tasks, seed);
			WriteU32(lastSpecular, tasks, 1);
			*phase = MK_RT_NEXT_VERTEX;
		}
		else
		{
			// Explicit light sample (direct lighting)
			WriteU32(lastSpecular, tasks, 0);
			*phase = MK_SAMPLE_LIGHT_EXPL;
		}
        
    }
}
