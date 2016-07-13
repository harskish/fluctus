#pragma once

#include "math/float3.hpp"
#include "math/float2.hpp"
#include "math/int3.hpp"
#include "math/matrix.hpp"
//#include "3d/Mesh.hpp"
//#include "base/math.hpp"

using FireRays::float3;
using FireRays::float2;
using FireRays::int3;
using FireRays::matrix;

/*inline float2 getTexelCoords(float2 uv, const Vec2i size) {
    uv -= float2(floor(uv.x), floor(uv.y));
    return float2(uv.x * (size.x - 1), uv.y * (size.y - 1));
}*/

struct VertexPNTC
{
    float3 p; // position
    float3 n; // normal
    float3 t; // texture coordinates
    float3 c; // color

    VertexPNTC(void) {}
    VertexPNTC(const float3& pp, const float3& nn, const float3& tt, const float3& cc) : p(pp), n(nn), t(tt), c(cc) {}
};

struct tri_data {
    int3 vertex_indices; // indices to the vertex array of the mesh
    matrix M; // Mat3f (for Woop)
    float3 N; // for Woop intersection

    tri_data() : M(), N() { }
    //tri_data() : N() { }

    tri_data(const tri_data &other) : M(other.M), N(other.N), vertex_indices(other.vertex_indices) { }
    //tri_data(const tri_data &other) : N(other.N), vertex_indices(other.vertex_indices) { }

    /*tri_data(const float3 &v0, const float3 &v1, const float3 &v2, const float3 normal) {
        M.setCol(0, v1 - v0);
        M.setCol(1, v2 - v0);
        M.setCol(2, normal);

        M.invert();
        N = -M * v0;
    }*/

    tri_data(const float3 &v0, const float3 &v1, const float3 &v2, const float3 normal) {
        float3 col0 = v1 - v0;
        float3 col1 = v2 - v0;
        float3 col2 = normal;

        M.m00 = col0.x; M.m01 = col1.x; M.m02 = col2.x;
        M.m10 = col0.y; M.m11 = col1.y; M.m12 = col2.y;
        M.m20 = col0.z; M.m21 = col1.z; M.m22 = col2.z;

        M = inverse(M);
        N = -M * v0;
    }
};

// The user pointer member can be used for identifying the triangle in the "parent" mesh representation.
struct RTTriangle {

    VertexPNTC m_vertices[3];            // The vertices of the triangle.

    //MeshBase::Material *m_material;    // Material of the triangle
    tri_data m_data;                     // Holds the matrix and vector necessary for Woop intersection and vertex index in the mesh

    RTTriangle(const VertexPNTC v0, const VertexPNTC v1, const VertexPNTC v2) {
        m_vertices[0] = v0;
        m_vertices[1] = v1;
        m_vertices[2] = v2;
        m_data = tri_data(v0.p, v1.p, v2.p, normal());
    }

    inline float3 min() const {
        return std::min(m_vertices[0].p, std::min(m_vertices[1].p, m_vertices[2].p));
    }

    inline float3 max() const {
        return std::max(m_vertices[0].p, std::max(m_vertices[1].p, m_vertices[2].p));
    }

    inline float3 centroid() const {
        return (m_vertices[0].p + m_vertices[1].p + m_vertices[2].p) * (1.0f / 3.0f);
    }

    inline float area() const {
        return length(cross(m_vertices[1].p - m_vertices[0].p, m_vertices[2].p - m_vertices[0].p)) * .5f;
    }

    float3 normal() const {
        return normalize(cross(m_vertices[1].p - m_vertices[0].p, m_vertices[2].p - m_vertices[0].p));
    }

    /*
    bool isOpaque(F32 u, F32 v) const {
        MeshBase::Material *mat = m_material;
        Vec2f uv = (1 - u - v) * m_vertices[0].t + u * m_vertices[1].t + v * m_vertices[2].t;
        Texture &tex = mat->textures[MeshBase::TextureType_Alpha];
        F32 alpha = 1.0f; // opaque by default
        if (tex.exists()) {
            const Image &img = *tex.getImage();
            Vec2i texelCoords = getTexelCoords(uv, img.getSize());
            alpha = img.getVec4f(texelCoords).getXYZ().y; // alpha in green channel
        }
        return alpha > 0.5f;
    }
     */

    //Triangle intersection as suggested in [Woop04]
    /*
    bool RTTriangle::intersect_woop(const float3 &orig, const float3 &dir, float &t, float &u, float &v) const {

        float3 transformed_orig = m_data.M * orig + m_data.N,
                transformed_dir = m_data.M * dir;

        t = -transformed_orig.z / transformed_dir.z;
        u = transformed_orig.x + transformed_dir.x * t;
        v = transformed_orig.y + transformed_dir.y * t;

        return u > .0f && v > .0f && u + v < 1.0f;
    }*/
};
