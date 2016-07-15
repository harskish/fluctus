#pragma once

#include "triangle.hpp"
#include "math/float3.hpp"
#include <iostream>

using FireRays::float3;
using FireRays::float4;

// TODO: remove these!
typedef unsigned int U32;
typedef int S32;
typedef float F32;
typedef char U8;

enum SplitMode {
	SplitMode_SpatialMedian,
	SplitMode_ObjectMedian,
	SplitMode_Sah
};

struct AABB {
    float3 min, max;
    inline AABB() : min(), max() {}
    inline AABB(const float3& min, const float3& max) : min(min), max(max) {}
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

    /*
    // SLAB AABB intersection algorithm
	inline bool intersectSlab(const float3& orig, const float3& dinv, float *tminRet, float *tMaxRet) const {
		float dinvx = dinv.x;

		float tmin = (min.x - orig.x) * dinvx;
		float tmax = (max.x - orig.x) * dinvx;

		if (dinvx < 0) {
			std::swap(tmin, tmax);
		}

		if (tmax < 0) {
			return false;
		}

		float dinvy = dinv.y;
		float tminy = (min.y - orig.y) * dinvy;
		float tmaxy = (max.y - orig.y) * dinvy;

		if (dinvy  < 0) {
			std::swap(tminy, tmaxy);
		}

		if (tmin > tmaxy || tmax < tminy) {
			return false;
		}

		if (tminy > tmin) {
			tmin = tminy;
		}

		if (tmaxy < tmax) {
			tmax = tmaxy;
		}

		if (tmax < 0) {
			return false;
		}

		float dinvz = dinv.z;
		float tminz = (min.z - orig.z) * dinvz;
		float tmaxz = (max.z - orig.z) * dinvz;

		if (dinvz <  0) {
			std::swap(tminz, tmaxz);
		}

		if (tmin > tmaxz || tmax < tminz) {
			return false;
		}

		if (tminz > tmin) {
			tmin = tminz;
		}

		if (tmaxz < tmax) {
			tmax = tmaxz;
		}

		// Assign output variables
		*tminRet = tmin;
		*tMaxRet = tmax;
		return true;
	}*/
};


inline std::ostream& operator<<(std::ostream& os, const float4& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
}

inline std::ostream& operator<<(std::ostream& os, const AABB& bb) {
    return os << "BB(" << bb.min << ", " << bb.max << ")";
}