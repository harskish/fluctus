#pragma once

#include "math/float3.hpp"

using FireRays::float3;

struct VertexPNT
{
    float3 p; // position
    float3 n; // normal
    float3 t; // texture coordinates
    // float3 c; // color

    VertexPNT(void) {}
    VertexPNT(const float3& pp, const float3& nn, const float3& tt) : p(pp), n(nn), t(tt) {}
};

struct RTTriangle {

    VertexPNT v0, v1, v2;
    int matId;

	// TODO: Fix alignment issues!
    RTTriangle(const VertexPNT &v0i, const VertexPNT &v1i, const VertexPNT &v2i) {
        v0 = v0i;
        v1 = v1i;
        v2 = v2i;
    }

    inline float3 min() const {
        return vmin(v0.p, vmin(v1.p, v2.p));
    }

    inline float3 max() const {
        return vmax(v0.p, vmax(v1.p, v2.p));
    }

    inline float3 centroid() const {
        return (v0.p + v1.p + v2.p) * (1.0f / 3.0f);
    }

    inline float area() const {
        return length(cross(v1.p - v0.p, v2.p - v0.p)) * .5f;
    }

    float3 normal() const {
        return normalize(cross(v1.p - v0.p, v2.p - v0.p));
    }
};
