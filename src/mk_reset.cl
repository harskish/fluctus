#include "geom.h"

// Reset state of all paths. Done after camera/renderparam changes.
kernel void reset(
    global GPUTaskState *tasks,
    global float *pixels,
    global float* denoiserAlbedo,
    global float* denoiserNormal,
    global RenderParams *params,
    uint numTasks
)
{
	const size_t gid = get_global_id(0) + get_global_id(1) * params->width;
	const uint limit = min(params->width * params->height, numTasks);

	if (gid >= limit)
		return;

	const uint x = get_global_id(0);
	const uint y = get_global_id(1);

	// Reset color for each pixel
	vstore4((float4)(0.0f), gid, pixels);
    vstore4((float4)(0.0f), gid, denoiserNormal);
    vstore4((float4)(0.1f, 0.1f, 0.1f, 0.0f), gid, denoiserAlbedo);

	// Reset path phase
	global PathPhase *phase = (global PathPhase*)&ReadI32(phase, tasks);
	*phase = MK_GENERATE_CAMERA_RAY;

	// Reset path state
	const float4 zero = (float4)(0.0f);
	const float4 one = (float4)(1.0f);
	WriteFloat3(Ei, tasks, zero);
	WriteFloat3(T, tasks, one);
	WriteU32(pathLen, tasks, 0);
	WriteU32(lastSpecular, tasks, 1);
	WriteF32(lastPdfW, tasks, 1.0f);
    WriteU32(firstDiffuseHit, tasks, 0);

	// Reset RNG seed
	WriteU32(seed, tasks, gid);
}
