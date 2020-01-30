#pragma once

#include "math/float3.hpp"

namespace fr = FireRays;

struct VertexPNT
{
    fr::float3 p; // position
    fr::float3 n; // normal
    fr::float3 t; // texture coordinates
    // float3 c; // color

    VertexPNT(void) {}
    VertexPNT(const fr::float3& pp, const fr::float3& nn, const fr::float3& tt) : p(pp), n(nn), t(tt) {}
};

struct RTTriangle {

    VertexPNT v0, v1, v2;
    int matId = 0; // default material, defined in scene constructor

	// TODO: Fix alignment issues!
    RTTriangle(const VertexPNT &v0i, const VertexPNT &v1i, const VertexPNT &v2i) {
        v0 = v0i;
        v1 = v1i;
        v2 = v2i;
    }

    inline fr::float3 min() const {
        return vmin(v0.p, vmin(v1.p, v2.p));
    }

    inline fr::float3 max() const {
        return vmax(v0.p, vmax(v1.p, v2.p));
    }

    inline fr::float3 centroid() const {
        return (v0.p + v1.p + v2.p) * (1.0f / 3.0f);
    }

    inline float area() const {
        return length(cross(v1.p - v0.p, v2.p - v0.p)) * .5f;
    }

    fr::float3 normal() const {
        return normalize(cross(v1.p - v0.p, v2.p - v0.p));
    }
};
