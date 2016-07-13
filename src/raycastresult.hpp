#pragma once

#include "rttriangle.hpp"
#include "math/float3.hpp"
#include "math/float2.hpp"
//#include "base/Math.hpp"

#include <limits>

using FireRays::float3;
using FireRays::float2;

// Result information of a raycast.
struct RaycastResult {
	const RTTriangle* tri; // The triangle that was hit.
	float t;               // Hit position is orig + t * dir.
	float u, v;            // Barycentric coordinates at the hit triangle.
	float3 point;           // Hit position.
	float3 orig, dir;       // The traced ray. Convenience for tracing and visualization. This is not strictly needed.

	float3 dPdx, dPdy, dDdx, dDdy; // EXTRA: Partial derivatives for texture filtering.
	float2 delta_size;

	RaycastResult(const RTTriangle* tri,
                  float t, float u, float v,
				  float3 point, const float3& orig, const float3& dir)
		: tri(tri),
          t(t),
          u(u), v(v),
          point(point),
          orig(orig), dir(dir)
    {}

	RaycastResult()
        : tri(nullptr),
          t(std::numeric_limits<float>::max()),
	      u(), v(),
          point(),
          orig(), dir()
    {}

	inline operator bool() { return tri != nullptr; }
};