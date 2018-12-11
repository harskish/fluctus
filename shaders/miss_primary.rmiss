#version 460
#extension GL_NV_ray_tracing : require

layout(location = 0) rayPayloadInNV Payload {
	vec3 color;
} payload;

/* Primary ray miss shader */

void main() {
    payload.color = vec3(0.2, 0.2, 0.2);
}
