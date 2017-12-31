#ifndef CL_BVH
#define CL_BVH

#include "geom.h"
#include "utils.cl"
#include "intersect.cl"

//#define USE_BITSTACK

#ifdef USE_BITSTACK
// Traversal with bitstacks - https://github.com/martinradev/BVH-algo-lib/blob/master/shaders/trace.glsl
inline void bvh_intersect(Ray *r, Hit *hit, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    int top = 0;
    int lstack = 0;
    int rstack = 0;

    while (top != -1) // not in root node
    {
        bool trackback = false;
        GPUNode n = nodes[top]; // updated in backtracing stage => not const

        if (n.nPrims != 0) // Leaf node
        {
            float tmin = FLT_MAX, umin = 0.0f, vmin = 0.0f;
            int imin = -1;
            for (uint i = n.iStart; i < n.iStart + n.nPrims; i++)
            {
                float t, u, v;
                if (intersectTriangle(r, &(tris[indices[i]]), &t, &u, &v))
                {
                    if (t > 0.0f && t < tmin)
                    {
                        imin = i;
                        tmin = t;
                        umin = u;
                        vmin = v;
                    }
                }
            }
            if (imin != -1 && tmin < hit->t)
            {
                hit->i = indices[imin];
                hit->matId = tris[indices[imin]].matId;
                hit->t = tmin;
                hit->P = r->orig + tmin * r->dir;
                hit->N = normalize(lerp(umin, vmin, tris[indices[imin]].v0.n, tris[indices[imin]].v1.n, tris[indices[imin]].v2.n));
                hit->uvTex = lerp(umin, vmin, tris[indices[imin]].v0.t, tris[indices[imin]].v1.t, tris[indices[imin]].v2.t).xy;
            }

            trackback = true;
        }
        else {
            float dummy, t1, t2;

            bool r1 = intersectAABB(r, &(nodes[top + 1].box), &t1, &dummy, hit->t);
            bool r2 = intersectAABB(r, &(nodes[n.rightChild].box), &t2, &dummy, hit->t);

            if (r1 && r2)
            {
                if (t1 <= t2)
                {
                    // first left
                    top = top + 1; // left child
                    lstack = (lstack | 1) << 1;
                    rstack <<= 1;
                }
                else
                {
                    // first right
                    top = n.rightChild;
                    rstack = (rstack | 1) << 1;
                    lstack <<= 1;
                }
            }
            else if (r1)
            {
                top = top + 1;
                lstack <<= 1;
                rstack <<= 1;
            }
            else if (r2)
            {
                top = n.rightChild;
                lstack <<= 1;
                rstack <<= 1;
            }
            else
            {
                trackback = true;
            }
        }

        if (trackback) {
            bool f = false;

            while (lstack != 0 || rstack != 0)
            {
                n = nodes[top];
                if ((lstack & 1) != 0) {
                    // visit right node
                    top = n.rightChild;
                    lstack &= ~1;
                    lstack <<= 1;
                    rstack <<= 1;
                    f = true;
                    break;
                }
                else if ((rstack & 1) != 0) {
                    // visit left node
                    top = top + 1;
                    rstack &= ~1;
                    lstack <<= 1;
                    rstack <<= 1;
                    f = true;
                    break;
                }
                top = n.parent;
                lstack >>= 1;
                rstack >>= 1;
            }

            if (!f)
                break;
        }
    }
}

// Traversal with bitstacks - https://github.com/martinradev/BVH-algo-lib/blob/master/shaders/trace.glsl
inline bool bvh_occluded(Ray *r, float *maxDist, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    int top = 0;
    int lstack = 0;
    int rstack = 0;

    while (top != -1) // not in root node
    {
        bool trackback = false;
        GPUNode n = nodes[top]; // updated in backtracing stage => not const

        if (n.nPrims != 0) // Leaf node
        {
            for (uint i = n.iStart; i < n.iStart + n.nPrims; i++)
            {
                float t, u, v;
                if (intersectTriangle(r, &(tris[indices[i]]), &t, &u, &v) && t > 0.0f && t < *maxDist)
                {
                    return true;
                }
            }

            trackback = true;
        }
        else {
            float dummy, t1, t2;

            bool r1 = intersectAABB(r, &(nodes[top + 1].box), &t1, &dummy, *maxDist);
            bool r2 = intersectAABB(r, &(nodes[n.rightChild].box), &t2, &dummy, *maxDist);

            if (r1 && r2)
            {
                if (t1 <= t2)
                {
                    // first left
                    top = top + 1; // left child
                    lstack = (lstack | 1) << 1;
                    rstack <<= 1;
                }
                else
                {
                    // first right
                    top = n.rightChild;
                    rstack = (rstack | 1) << 1;
                    lstack <<= 1;
                }
            }
            else if (r1)
            {
                top = top + 1;
                lstack <<= 1;
                rstack <<= 1;
            }
            else if (r2)
            {
                top = n.rightChild;
                lstack <<= 1;
                rstack <<= 1;
            }
            else
            {
                trackback = true;
            }
        }

        if (trackback) {
            bool f = false;

            while (lstack != 0 || rstack != 0)
            {
                n = nodes[top];
                if ((lstack & 1) != 0) {
                    // visit right node
                    top = n.rightChild;
                    lstack &= ~1;
                    lstack <<= 1;
                    rstack <<= 1;
                    f = true;
                    break;
                }
                else if ((rstack & 1) != 0) {
                    // visit left node
                    top = top + 1;
                    rstack &= ~1;
                    lstack <<= 1;
                    rstack <<= 1;
                    f = true;
                    break;
                }
                top = n.parent;
                lstack >>= 1;
                rstack >>= 1;
            }

            if (!f)
                break;
        }
    }

    return false;
}

#else
// BVH traversal using simulated stack
inline void bvh_intersect(Ray *r, Hit *hit, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    float lnear, lfar, rnear, rfar; // AABB limits
    uint closer, farther;

    // Stack state
    uint stack[64]; // causes large stack frames (NVIDIA build log)
    int stackptr = 0;

    // Root node
    stack[stackptr] = 0;

    while (stackptr >= 0)
    {
        // Next node
        int ni = stack[stackptr];
        stackptr--;
        const GPUNode n = nodes[ni];

        if (n.nPrims != 0) // Leaf node
        {
            float tmin = FLT_MAX, umin = 0.0f, vmin = 0.0f;
            int imin = -1;
            for (uint i = n.iStart; i < n.iStart + n.nPrims; i++)
            {
                float t, u, v;
                if (intersectTriangle(r, &(tris[indices[i]]), &t, &u, &v))
                {
                    if (t > 0.0f && t < tmin)
                    {
                        imin = i;
                        tmin = t;
                        umin = u;
                        vmin = v;
                    }
                }
            }
            if (imin != -1 && tmin < hit->t)
            {
                hit->i = indices[imin];
                hit->matId = tris[indices[imin]].matId;
                hit->t = tmin;
                hit->P = r->orig + tmin * r->dir;
                hit->N = normalize(lerp(umin, vmin, tris[indices[imin]].v0.n, tris[indices[imin]].v1.n, tris[indices[imin]].v2.n));
                hit->uvTex = lerp(umin, vmin, tris[indices[imin]].v0.t, tris[indices[imin]].v1.t, tris[indices[imin]].v2.t).xy;
            }
        }
        else // Internal node
        {
            bool leftWasHit = intersectAABB(r, &(nodes[ni + 1].box), &lnear, &lfar, hit->t);
            bool rightWasHit = intersectAABB(r, &(nodes[n.rightChild].box), &rnear, &rfar, hit->t);

            if (leftWasHit && rightWasHit)
            {
                closer = ni + 1;
                farther = n.rightChild;

                // Right child was closer -> swap
                if (rnear < lnear) swap_m(closer, farther, uint);

                // Farther node pushed first
                stack[++stackptr] = farther;
                stack[++stackptr] = closer;
            }

            else if (leftWasHit)
            {
                stack[++stackptr] = ni + 1;
            }

            else if (rightWasHit)
            {
                stack[++stackptr] = n.rightChild;
            }
        }
    }
}

inline bool bvh_occluded(Ray *r, float *maxDist, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    float lnear, lfar, rnear, rfar; // AABB limits
    uint closer, farther;

    // Stack state
    uint stack[64]; // causes large stack frames (NVIDIA build log)
    int stackptr = 0;

    // Root node
    stack[stackptr] = 0;

    while (stackptr >= 0)
    {
        // Next node
        int ni = stack[stackptr];
        stackptr--;
        const GPUNode n = nodes[ni];

        if (n.nPrims != 0) // Leaf node
        {
            for (uint i = n.iStart; i < n.iStart + n.nPrims; i++)
            {
                float t, u, v;
                if (intersectTriangle(r, &(tris[indices[i]]), &t, &u, &v) && t > 0.0f && t < *maxDist)
                {
                    return true;
                }
            }
        }
        else // Internal node
        {
            bool leftWasHit = intersectAABB(r, &(nodes[ni + 1].box), &lnear, &lfar, *maxDist);
            bool rightWasHit = intersectAABB(r, &(nodes[n.rightChild].box), &rnear, &rfar, *maxDist);

            if (leftWasHit && rightWasHit)
            {
                closer = ni + 1;
                farther = n.rightChild;

                // Right child was closer -> swap
                if (rnear < lnear) swap_m(closer, farther, uint);

                // Farther node pushed first
                stack[++stackptr] = farther;
                stack[++stackptr] = closer;
            }

            else if (leftWasHit)
            {
                stack[++stackptr] = ni + 1;
            }

            else if (rightWasHit)
            {
                stack[++stackptr] = n.rightChild;
            }
        }
    }

    return false;
}
#endif

#endif