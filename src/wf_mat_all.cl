#include "utils.cl"
#include "bxdf_partial.cl"
#include "ptx_asm.cl"

kernel void wavefrontAllMaterials(
    global GPUTaskState *tasks,
    global QueueCounters *queueLens,
    global uint *materialQueue,
    global uint *extensionQueue,
    global Material *materials,
    global uchar *texData,
    global TexDescriptor *textures,
    global RenderParams *params,
    uint numTasks
)
{
    const uint gid_direct = get_global_id(0);
    if (gid_direct >= queueLens->diffuseQueue) // technically stored in diffuse queue
        return;

    uint gid = materialQueue[gid_direct];
    uint seed = ReadU32(seed, tasks);

    Hit hit = readHitSoA(tasks, gid, numTasks);
    Material mat = materials[hit.matId];
    bool backface = (bool)ReadU32(backfaceHit, tasks);

    float3 dirIn = ReadFloat3(dir, tasks); // points toward surface!
    float3 L = ReadFloat3(shadowDir, tasks);

    const float3 bsdfNEE = bxdfEval(&hit, &mat, backface, textures, texData, dirIn, L);
    const float bsdfPdfW = max(0.0f, bxdfPdf(&hit, &mat, backface, textures, texData, dirIn, L));
    WriteFloat3(lastBsdf, tasks, bsdfNEE);
    WriteF32(lastPdfImplicit, tasks, bsdfPdfW);
    
    // Generate continuation ray by sampling BSDF
    float pdfW;
    float3 newDir;
    float3 bsdf = bxdfSample(&hit, &mat, backface, textures, texData, dirIn, &newDir, &pdfW, &seed);
    float costh = dot(hit.N, normalize(newDir));
	
    // Update throughput * pdf
	const float3 oldT = ReadFloat3(T, tasks);
    float3 newT;
    if (pdfW == 0.0f || isZero(bsdf))
		newT = (float3)(0.0f, 0.0f, 0.0f);
    else
        newT = oldT * bsdf * costh / pdfW;
        
    // Avoid self-shadowing
    float3 orig = hit.P + 1e-4f * newDir;

	// Update path state
	WriteFloat3(lastT, tasks, oldT);
    WriteFloat3(T, tasks, newT);
	WriteFloat3(orig, tasks, orig);
	WriteFloat3(dir, tasks, newDir);
	WriteF32(lastPdfW, tasks, pdfW);
	WriteU32(seed, tasks, seed);
	WriteU32(lastSpecular, tasks, BXDF_IS_SINGULAR(mat.type));

    // Add to extension queue
    uint idx = atomicIncAll(&queueLens->extensionQueue);
    extensionQueue[idx] = gid;
}