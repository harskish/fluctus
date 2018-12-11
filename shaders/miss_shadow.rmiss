#version 460
#extension GL_NV_ray_tracing : require

layout(location = 1) rayPayloadInNV ShadowPayload {
	uint blocked;
} shadowPayload;

/* Shadow ray that misses all geometry */

void main() {
    shadowPayload.blocked = 0;
}
