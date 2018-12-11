#version 460
#extension GL_NV_ray_tracing : require

// Aligned as vec4
struct Vertex {
	vec3 pos;
	vec3 normal;
	vec3 color;
};

layout(location = 0) rayPayloadInNV Payload {
	vec3 color;
} payload;

layout(location = 1) rayPayloadNV ShadowPayload {
	uint blocked;
} shadowPayload;

hitAttributeNV vec3 attribs;

layout(std430, binding = 3) readonly buffer Indices {
    uint indices[];
};

layout(std430, binding = 4) readonly buffer Vertices {
    Vertex vertices[];
};

layout(binding = 5) uniform accelerationStructureNV bvh;

layout (std140, binding = 6) readonly uniform UBO
{
	mat4 invR;
	vec4 camPos;
	vec4 lightPos;
	float aspectRatio;
	float fov;
} ubo;

const vec3 lightColor = vec3(1.0, 1.0, 1.0);
const float lightIntensity = 1.0;

void main() {
	vec3 bar = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

	uint i0 = uint(indices[gl_PrimitiveID * 3 + 0]);
	uint i1 = uint(indices[gl_PrimitiveID * 3 + 1]);
	uint i2 = uint(indices[gl_PrimitiveID * 3 + 2]);

	vec3 C0 = vertices[i0].color;
	vec3 C1 = vertices[i1].color;
	vec3 C2 = vertices[i2].color;
	vec3 C = bar.x * C0 + bar.y * C1 + bar.z * C2;

	vec3 N0 = vertices[i0].normal;
	vec3 N1 = vertices[i1].normal;
	vec3 N2 = vertices[i2].normal;
	vec3 N = bar.x * N0 + bar.y * N1 + bar.z * N2;

	vec3 dirIn = gl_WorldRayDirectionNV;
	dirIn.y *= -1.0;

	vec3 posWorld = gl_WorldRayOriginNV + gl_WorldRayDirectionNV * gl_HitTNV;
	vec3 L = ubo.lightPos.xyz - posWorld;
	float lightDist = length(L);

	shadowPayload.blocked = 0;
	const uint rayFlags = gl_RayFlagsOpaqueNV | gl_RayFlagsTerminateOnFirstHitNV;
	traceNV(bvh, rayFlags, 0xff, 1, 0, 1, posWorld, 1e-3f, normalize(L), lightDist, 1);

	if (shadowPayload.blocked == 0)
		payload.color = abs(dot(L, N)) * C * lightColor * lightIntensity / (lightDist * lightDist);
	else
		payload.color = C * 0.05;
}
