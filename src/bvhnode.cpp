#include "bvhnode.hpp"

void BuildNode::computeBB(std::vector<U32> &indices, std::vector<RTTriangle> *tris) {
    for (U32 i = iStart; i <= iEnd; i++) {
        RTTriangle tri = (*tris)[indices[i]];
        box.expand(tri);
    }
}