#version 460
#extension GL_NV_ray_tracing : require

#include "utils.glsl"

layout(location = 0) rayPayloadNV PrimaryPayload payload;

/* Primary ray miss shader */


void main() {
    payload.color = vec3(0.2, 0.2, 0.2);
}
