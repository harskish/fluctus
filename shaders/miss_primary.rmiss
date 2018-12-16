#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "utils.glh"

layout(location = 0) rayPayloadNV PrimaryPayload payload;

/* Primary ray miss shader */


void main() {
    payload.color = vec3(0.2, 0.2, 0.2);
}
