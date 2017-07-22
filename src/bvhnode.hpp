#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include "rtutil.hpp"
#include "math/float3.hpp"

struct TriRef
{
	U32 ind;	// index of tri
	AABB_t box;	// bounding box
	float3 pos;	// triangle centroid

	TriRef(void) {}
	TriRef(const TriRef &other) : ind(other.ind), box(other.box), pos(other.pos) {}
	TriRef(U32 i, RTTriangle &tri) : ind(i), box(tri.min(), tri.max()), pos(tri.centroid()) {}
};

/* Fat node used in BVH construction */
struct BuildNode
{
	AABB_t box;								// Axis-aligned bounding box
	U32 iStart, iEnd;						// Indices in the index list
	S32 rightChild = -1;					// Index into node vector (left child always current + 1)
    S32 parent;                             // -1 for root

	void computeBB(std::vector<TriRef> &refs);
	inline U32 spannedTris() const { return iEnd - iStart + 1; }

	BuildNode(void) : iStart(1), iEnd(0) {} // => 0 spanned tris
	BuildNode(U32 s, U32 e) : iStart(s), iEnd(e) {}
};

/* Pointer-based node used in SBVH construction */
struct SBVHNode
{
	AABB_t box;
	U32 lo, hi; // inclusive
	SBVHNode *leftChild = nullptr;
	SBVHNode *rightChild = nullptr;
	inline U32 spannedTris() const { return hi - lo; }
	inline bool isLeaf() const { return !leftChild && !rightChild; }
	void deleteTree();

	SBVHNode(const AABB_t &b, SBVHNode* l, SBVHNode* r) : box(b), leftChild(l), rightChild(r) {} // inner node
	SBVHNode(const AABB_t &b, int l, int h) : box(b), lo(l), hi(h) {} // leaf node
};

/* Small node used in BVH traversal */
struct Node
{
	AABB_t box;
	S32 parent;
	union {
		U32 iStart;		// leaf node, indiex into index list
		U32 rightChild; // internal node, index into node vector (left child always current + 1)
	};
	U8 nPrims = 0;		// 0 for interior nodes
};