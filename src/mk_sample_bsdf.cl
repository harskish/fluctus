#include "geom.h"
#include "bvh.cl"
#include "utils.cl"
#include "intersect.cl"
#include "env_map.cl"
#include "bxdf.cl"

// Microkernel for BSDF sampling and NEE
// State changes:
//   MK_SAMPLE_BSDF => MK_SPLAT_SAMPLE
kernel void sampleBsdf(
    global GPUTaskState *tasks,
    global Material *materials,
    global uchar *texData,
    global TexDescriptor *textures,
    read_only image2d_t envMap,
    global float *probTable,
    global int *aliasTable,
    global float *pdfTable,
    global Triangle *tris,
    global GPUNode *nodes,
    global uint *indices,
    global RenderParams *params,
    global RenderStats *stats,
    uint numTasks,
    uint iteration)
{
    const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
    const uint limit = min(params->width * params->height, numTasks);
    uint seed = ReadU32(seed, tasks);

    if (gid >= limit)
        return;

    // Read the path state
    global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
    if (*phase != MK_SAMPLE_BSDF)
        return;

    const float3 rayOrig = ReadFloat3(orig, tasks);
    const float3 rayDir = ReadFloat3(dir, tasks);
    Ray r = { rayOrig, rayDir };

    // Read hit from path state
    Hit hit = readHitSoA(tasks, gid, numTasks);
    Material mat = materials[hit.matId];

    // Fix backside hits
    bool backface = dot(hit.N, r.dir) > 0.0f;
    if (backface) hit.N *= -1.0f;
    float3 orig = hit.P - 1e-3f * r.dir;  // avoid self-shadowing

    // Perform next event estimation
    bool lastSpecular = ReadU32(lastSpecular, tasks);
    if (params->sampleExpl && !lastSpecular)
    {
        const float lightPickProb = 1.0f;

        // Importance sample env map (using alias method)
        if (params->useEnvMap)
        {
            int2 envMapDims = get_image_dim(envMap);
            const int width = envMapDims.x, height = envMapDims.y;

            float3 L;
            float directPdfW = 0.0f;
            EnvMapContext ctx = { width, height, pdfTable, probTable, aliasTable };
            sampleEnvMapAlias(rand(&seed), &L, &directPdfW, ctx);

            // Shadow ray
            float lenL = 2.0f * params->worldRadius;
            L = normalize(L);
            Ray rLight = { orig, L };

            // TODO: BAD! Collect all shadow ray casts together (in queue, i.e. buffer of gids + atomic counter)!
            Hit hit = EMPTY_HIT(lenL);
            if (params->useAreaLight) intersectLight(&hit, &rLight, params);
            bool occluded = (hit.i > -1) || bvh_occluded(&rLight, &lenL, tris, nodes, indices);
            atomic_inc(&stats->shadowRays);

            // Compute contribution
            if (!occluded && directPdfW != 0.0f)
            {
                const float3 brdf = bxdfEval(&hit, &mat, textures, texData, r.dir, L);
                float cosTh = max(0.0f, dot(L, hit.N)); // cos at surface
                float bsdfPdfW = max(0.0f, bxdfPdf(&hit, &mat, textures, texData, r.dir, L));

                float weight = 1.0f;
                if (params->sampleImpl)
                {
                    weight = (directPdfW * lightPickProb) / (directPdfW * lightPickProb + bsdfPdfW);
                }

                const float3 T = ReadFloat3(T, tasks);
                const float prob = ReadF32(pdf, tasks);
                const float3 envMapLi = evalEnvMapDir(envMap, L) * params->envMapStrength;
                const float3 contrib = brdf * T * envMapLi * weight * cosTh / (lightPickProb * directPdfW * prob);
                const float3 newEi = ReadFloat3(Ei, tasks) + contrib;
                WriteFloat3(Ei, tasks, newEi);
            }
        }
        
        // Sample area light source
        if (params->useAreaLight)
        {
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
                const float3 brdf = bxdfEval(&hit, &mat, textures, texData, r.dir, L);
                float cosTh = max(0.0f, dot(L, hit.N)); // cos at surface
                float directPdfW = pdfAtoW(directPdfA, lenL, cosLight); // 'how small area light looks'
                float bsdfPdfW = max(0.0f, bxdfPdf(&hit, &mat, textures, texData, r.dir, L));

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
        float pdfW;
        float3 newDir;
        float3 bsdf = bxdfSample(&hit, &mat, backface, textures, texData, r.dir, &newDir, &pdfW, &seed);
        lastSpecular = BXDF_IS_SINGULAR(mat.type);
        float costh = dot(hit.N, normalize(newDir));

        float3 newT = ReadFloat3(T, tasks) * bsdf * costh;
        float newPdf = ReadF32(pdf, tasks) * pdfW;
        
        // Avoid self-shadowing
        orig = hit.P + 1e-4f * newDir;
        r.dir = newDir;

        // Update path state
        WriteFloat3(T, tasks, newT);
        WriteFloat3(orig, tasks, orig);
        WriteFloat3(dir, tasks, r.dir);
        WriteF32(pdf, tasks, newPdf);
        WriteF32(lastPdfW, tasks, pdfW);

        // Perform raycast
        *phase = MK_RT_NEXT_VERTEX;
    }

    // Update RNG seed
    WriteU32(seed, tasks, seed);
}
