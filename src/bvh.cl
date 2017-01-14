#ifndef CL_BVH
#define CL_BVH

#include "geom.h"
#include "utils.cl"
#include "intersect.cl"

// BVH traversal using simulated stack
inline bool bvh_intersect_stack(Ray *r, Hit *hit, global Triangle *tris, global GPUNode *nodes, global uint *indices)
{
    float lnear, lfar, rnear, rfar; //AABB limits
    uint closer, farther;

    bool found = false;

    // Stack state
    SimStackNode stack[32]; // this causes the large stack frames seen in the build log
    int stackptr = 0;

    // Root node
    stack[stackptr].i = 0;
    stack[stackptr].mint = -FLT_MAX;

    while (stackptr >= 0)
    {
        // Next node
        int ni = stack[stackptr].i;
        float tnear = stack[stackptr].mint;
        stackptr--;
        const GPUNode n = nodes[ni];

        // Closer intersection found already
        if (tnear > hit->t)
            continue;

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
                found = true;
                hit->i = indices[imin];
                hit->t = tmin;
                hit->P = r->orig + tmin * r->dir;
                hit->N = lerp(umin, vmin, tris[indices[imin]].v0.n, tris[indices[imin]].v1.n, tris[indices[imin]].v2.n);
            }
        }
        else // Internal node
        {
            float3 N_tmp;
            bool leftWasHit = intersectSlab(r, &(nodes[ni + 1].box), &lnear, &lfar, &N_tmp);
            bool rightWasHit = intersectSlab(r, &(nodes[n.rightChild].box), &rnear, &rfar, &N_tmp);

            if (leftWasHit && rightWasHit)
            {
                closer = ni + 1;
                farther = n.rightChild;

                // Right child was closer -> swap
                if (rnear < lnear)
                {
                    swap_m(lnear, rnear, float);
                    swap_m(lfar, rfar, float);
                    swap_m(closer, farther, uint);
                }

                // Farther node pushed first
                stack[++stackptr] = (SimStackNode){farther, rnear};
                stack[++stackptr] = (SimStackNode){closer, lnear};
            }

            else if (leftWasHit)
            {
                stack[++stackptr] = (SimStackNode){ni + 1, lnear};
            }

            else if (rightWasHit)
            {
                stack[++stackptr] = (SimStackNode){n.rightChild, rnear};
            }
        }
    }

    return found;
}

#endif