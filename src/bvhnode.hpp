#pragma once

#include <memory>
#include <vector>
#include "rtutil.hpp"
//#include "RaycastResult.hpp"

/* Fat node used in BVH construction */
struct BuildNode
{
	AABB_t box;								// Axis-aligned bounding box
	U32 iStart, iEnd;						// Indices in the index list
	S32 rightChild = -1;					// Index into node vector (left child always current + 1)

	void computeBB(std::vector<U32> &indices, std::vector<RTTriangle> *tris);
	inline U32 spannedTris() const { return iEnd - iStart + 1; }
};

/* Small node used in BVH traversal */
struct alignas(32) Node
{
	AABB_t box;
	union {
		U32 iStart;		// leaf node, indiex into index list
		U32 rightChild; // internal node, index into node vector (left child always current + 1)
	};
	U8 nPrims = 0;		// 0 for interior nodes
};