#pragma once

#include "cl2.hpp"
#include "triangle.hpp"
#include "math/float3.hpp"
#include <iostream>
#include <cfloat>

namespace fr = FireRays;

// TODO: remove these!
typedef cl_uint U32;
typedef cl_int S32;
typedef cl_float F32;
typedef cl_uchar U8;

enum class SplitMode {
	SpatialMedian,
	ObjectMedian,
	SAH
};

struct AABB_t {
    fr::float3 min, max;
    inline AABB_t() : min(FLT_MAX), max(-FLT_MAX) {}
    inline AABB_t(const fr::float3& min, const fr::float3& max) : min(min), max(max) {}
	inline F32 area() const {
        fr::float3 d(max - min);
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
	inline U32 maxDim() const {
		U32 axis = 0; // index of longest axis, assume x
        fr::float3 d = max - min;
		if (d.y > d[axis]) axis = 1;
		if (d.z > d[axis]) axis = 2;
		return axis;
	}
	inline fr::float3 centroid() {
		return 0.5f * (min + max);
	}
	inline void expand(const RTTriangle &t) {
		min = vmin(min, t.min());
		max = vmax(max, t.max());
	}
	inline void expand(const AABB_t &box) {
		min = vmin(min, box.min);
		max = vmax(max, box.max);
	}
	inline void expand(const fr::float3 &p) {
		min = vmin(min, p);
		max = vmax(max, p);
	}
	inline void intersect(const AABB_t &box) {
		min = vmax(min, box.min);
		max = vmin(max, box.max);
	}
};

inline std::ostream& operator<<(std::ostream& os, const fr::float4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& os, const AABB_t& bb) {
    return os << "BB(" << bb.min << ", " << bb.max << ")";
}
