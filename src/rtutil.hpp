#pragma once

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#include "cl2.hpp"
#include "triangle.hpp"
#include "math/float3.hpp"
#include <iostream>
#include <cfloat>

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
    inline AABB_t() : min(FLT_MAX), max(-FLT_MAX) {}
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
	inline float3 centroid() {
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
	inline void expand(const float3 &p) {
		min = vmin(min, p);
		max = vmax(max, p);
	}
	inline void intersect(const AABB_t &box) {
		min = vmax(min, box.min);
		max = vmin(max, box.max);
	}
};

inline std::string unixifyPath(std::string path) {
	size_t index = 0;
	while (true) {
		index = path.find("\\", index);
		if (index == std::string::npos) break;

		path.replace(index, 1, "/");
		index += 1;
	}

	return path;
}


inline std::ostream& operator<<(std::ostream& os, const float4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& os, const AABB_t& bb) {
    return os << "BB(" << bb.min << ", " << bb.max << ")";
}