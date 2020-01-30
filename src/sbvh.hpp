#pragma once

#include <vector>
#include <fstream>
#include "bvh.hpp"
#include "math/int3.hpp"

class ProgressView;
struct SBVHNode;

/*
	Split BVH (SBVH), based on "Spatial Splits in Bounding Volume Hierarchies" by Stich et al.
	Tree built from right to left, so that duplicated refs can be pushed to end of ref stack.
	Based on implementation by Aila & Laine 09.
*/
class SBVH : public BVH
{
public:
	SBVH(std::vector<RTTriangle>* tris, SplitMode mode, ProgressView *progress);
	SBVH(std::vector<RTTriangle>* tris, const std::string filename) : BVH(tris, filename) {}
	~SBVH() {}

private:
	struct NodeSpec;

	SBVHNode* build(NodeSpec &spec, int depth, F32 progressStart, F32 progressEnd);
	SBVHNode* createLeaf(const NodeSpec& spec);
	SplitInfo binSplit(const NodeSpec& spec, F32 nodeSAH);
	SplitInfo sahSplit(const NodeSpec& spec, F32 nodeSAH);
	void partitionObject(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SplitInfo& split);
	void partitionSpatial(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SplitInfo& split);
	void splitReference(TriRef& left, TriRef& right, const TriRef& ref, int dim, F32 coord);
	void lazyPrintBuildStatus(F32 progress);
	void convertTree(SBVHNode *node, S32 parentId);

	enum
	{
		MaxLeafElems = 8,
		MinLeafElems = 1,
		MaxDepth = 64,
		MaxSpatialDepth = 48,
		NumSpatialBins = 128
	};

	struct
	{
		U32 depth = 0;
		U32 bad_splits = 0;
		U32 splits = 0;
		U32 duplicates = 0;
	} metrics;

	struct NodeSpec
	{
		S32 refs;
		AABB_t box;
		NodeSpec(void) : refs(0) {}
	};

	struct Bin
	{
		AABB_t bounds;
		S32 entering;
		S32 exiting;
	};

	ProgressView *progress;
	
	Bin bins[3][NumSpatialBins];
	F32 splitAlpha = 1e-5f; // ~35% duplication rate
	F32 minOverlap;         // min area that triggers spatial split search
};
