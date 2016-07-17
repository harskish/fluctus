#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "cl2.hpp"
#include "triangle.hpp"
#include "math/float3.hpp"
#include <iostream>

using FireRays::float3;
using FireRays::float4;

// TODO: remove these!
typedef cl_uint U32;
typedef cl_int S32;
typedef cl_float F32;
typedef cl_uchar U8;

enum SplitMode {
	SplitMode_SpatialMedian,
	SplitMode_ObjectMedian,
	SplitMode_Sah
};

struct AABB_t {
    float3 min, max;
    inline AABB_t() : min(), max() {}
    inline AABB_t(const float3& min, const float3& max) : min(min), max(max) {}
    inline F32 area() const {
        float3 d(max - min);
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
	inline U32 maxDim() const {
		U32 axis = 0; // index of longest axis, assume x
		float3 d = max - min;
		if (d.y > d[axis]) axis = 1;
		if (d.z > d[axis]) axis = 2;
		return axis;
	}
	inline void expand(const RTTriangle &t) {
		if (length(max - min) == 0) { // First triangle
			min = t.min();
			max = t.max();
		}
		else {
			min = vmin(min, t.min());
			max = vmax(max, t.max());
		}
	}
};


inline std::ostream& operator<<(std::ostream& os, const float4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& os, const AABB_t& bb) {
    return os << "BB(" << bb.min << ", " << bb.max << ")";
}