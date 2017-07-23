#include "sbvh.hpp"

SBVH::SBVH(std::vector<RTTriangle>* tris, SplitMode mode)
{
	m_triangles = tris;

	NodeSpec rootSpec;
	rootSpec.refs = tris->size();

	// Setup references for building
	m_refs.resize(rootSpec.refs);
	for (int i = 0; i < m_triangles->size(); i++)
	{
		m_refs[i] = TriRef(i, (*m_triangles)[i]);
		rootSpec.box.expand(m_refs[i].box);
	}

	// Shared vector to avoid reallocations
	rightBoxes.resize(std::max(m_triangles->size(), (size_t)NumSpatialBins) - 1);
	minOverlap = rootSpec.box.area() * splitAlpha;

	// Perform building
	SBVHNode* root = build(rootSpec, 0, 0.0f, 1.0f);
	printf("\rSBVH builder: progress 100%% (%.2f%% duplicates)\n", metrics.duplicates * 100.0f / m_triangles->size());

	// Indices relative to LAST triangle => reverse
	std::reverse(m_indices.begin(), m_indices.end());

	// Convert tree structure to small node vector
	convertTree(root, -1);
	root->deleteTree();
	assert(metrics.depth <= MaxDepth);
	assert(m_indices.size() >= m_triangles->size());

	if (metrics.depth > MaxDepth)
		std::cout << "WARN: SBVH might not fit traversal stack! (" << metrics.depth << " > " << MaxDepth << ")" << std::endl;
	
	std::cout
		<< "======================" << std::endl
		<< "SBVH" << std::endl
		<< "Splits: " << metrics.splits << " (" << int(metrics.bad_splits / float(metrics.splits) * 100.0f) << "% bad)" << std::endl
		<< "Depth: " << metrics.depth << std::endl
		<< "Leaves: " << metrics.splits + 1 << std::endl
		<< "Duplicates: " << metrics.duplicates << " (" << int(metrics.duplicates * 100.0f / m_triangles->size()) << "%)" << std::endl
		<< "======================" << std::endl;
}

// Convert pointer tree to linear node vector
void SBVH::convertTree(SBVHNode *node, S32 parentId)
{
	U32 ind = m_nodes.size();
	m_nodes.push_back(Node());
	m_nodes[ind].box = node->box;
    m_nodes[ind].parent = parentId;

	if (node->isLeaf())
	{
		m_nodes[ind].iStart = m_indices.size() - node->hi;
		U32 sp = node->spannedTris();
		if (sp > std::numeric_limits<U8>::max())
			throw std::runtime_error("Too many prims to fit into U8!");
		m_nodes[ind].nPrims = (U8)(sp);
	}
	else
	{
		convertTree(node->leftChild, ind);
		m_nodes[ind].rightChild = m_nodes.size(); // save current vector size
		convertTree(node->rightChild, ind);
	}
}

// Too frequent printing is actually a bottleneck!
void SBVH::lazyPrintBuildStatus(F32 progress)
{
	S32 percentage = S32(ceil(progress * 100.0f));
	if (percentage > buildPercentage)
	{
		buildPercentage = percentage;
		F32 duplicates = metrics.duplicates * 100.0f / m_triangles->size();
		printf("\rSBVH builder: progress %d%% (%.2f%% duplicates)", percentage, duplicates);
	}
}

// Create leaf node. References are removed from stack.
// Index list will be reversed after building to fix indexing.
SBVHNode* SBVH::createLeaf(const NodeSpec& spec)
{
	for (int i = 0; i < spec.refs; i++)
	{
		TriRef last = m_refs.back();
		m_refs.pop_back();
		m_indices.push_back(last.ind);
	}

	int start = m_indices.size() - spec.refs;
	int end = m_indices.size();
	return new SBVHNode(spec.box, start, end);
}

// SBVH construction algorithm, in line with Stich et al. chapter 4.1
SBVHNode* SBVH::build(NodeSpec spec, int depth, F32 progressStart, F32 progressEnd)
{
	lazyPrintBuildStatus(progressStart);
	metrics.depth = std::max(metrics.depth, (U32)depth);

	if (spec.refs <= MinLeafElems || depth >= MaxDepth)
		return createLeaf(spec);

	// 1. Find object split candidate using full SAH search
	F32 parentArea = spec.box.area();
	F32 nodeSAH = parentArea * 2 * 1;
	SplitInfo objectSplit = sahSplit(spec, nodeSAH);

	// 2. Find spatial split candidate using chopped binning
	SplitInfo spatialSplit;
	if (depth < MaxSpatialDepth)
	{
		AABB_t overlap = objectSplit.leftBounds;
		overlap.intersect(objectSplit.rightBounds);
		if (overlap.area() >= minOverlap)
			spatialSplit = binSplit(spec, nodeSAH);
	}

	// 3. Select the winner candidate
	F32 parentCost = parentArea * (spec.refs) * sahParams.costTri;
	F32 minCost = std::min(objectSplit.cost, std::min(spatialSplit.cost, parentCost));

	// Check if parent is cheaper (SAH)
	if (minCost == parentCost && spec.refs <= MaxLeafElems)
	{
		assert(spec.refs <= std::numeric_limits<U8>::max());
		return createLeaf(spec);
	}

	// Perform partitioning
	NodeSpec left, right;
	if (minCost == spatialSplit.cost)
		partitionSpatial(left, right, spec, spatialSplit);
	if (!left.refs || !right.refs)
		partitionObject(left, right, spec, objectSplit);

	metrics.splits++;

	// Create inner node.
	metrics.duplicates += left.refs + right.refs - spec.refs;
	F32 progressMid = lerp(progressStart, progressEnd, (F32)right.refs / (F32)(left.refs + right.refs));

	// Built from right to left (so that duplicates can be added to end of ref list)
	SBVHNode* rightNode = build(right, depth + 1, progressStart, progressMid);
	SBVHNode* leftNode = build(left, depth + 1, progressMid, progressEnd);

	return new SBVHNode(spec.box, leftNode, rightNode);
}

BVH::SplitInfo SBVH::sahSplit(const NodeSpec& spec, F32 nodeSAH)
{
	F32 bestTieBreak = FLT_MAX;
	SplitInfo info;

	// Rightmost N references
	int start = m_refs.size() - spec.refs;
	int end = m_refs.size() - 1;

	// Loop over all three axes to find best split
	for (U32 dim = 0; dim < 3; dim++)
	{
		// Sort along axis
		sortReferences(start, end, dim);

		// Create AABB lookup
		//buildBoxLookup(n);

		AABB_t rightBounds;
		for (int i = spec.refs - 1; i > 0; i--)
		{
			rightBounds.expand(m_refs[start + i].box);
			rightBoxes[i - 1] = rightBounds; // use relative indexing (doesn't grow too large)
		}

		AABB_t leftBox;
		U32 leftCount = 0;

		// Try different split points along axis
		for (int i = 1; i < spec.refs; i++) // exclude first and last
		{
			leftBox.expand(m_refs[start + i - 1].box);
			leftCount++;

			AABB_t &rightBox = rightBoxes[i - 1];
			F32 areaRight = rightBox.area();
			F32 areaLeft = leftBox.area();

			// New best split?
			// F32 cost = sahCost(leftCount, areaLeft,
			//	spanSize - leftCount, areaRight, parentArea);

			// TODO: override only sahCost?
			F32 cleft = areaLeft * leftCount * sahParams.costTri;
			F32 cright = areaRight * (spec.refs - leftCount) * sahParams.costTri;
			F32 cost = nodeSAH + cleft + cright;
			F32 tieBreak = pow((F32)i, 2) + pow((F32)(spec.refs - i), 2);

			if (cost < info.cost || (cost == info.cost && tieBreak < bestTieBreak))
			{
				info.cost = cost;
				info.i = i;
				info.leftBounds = leftBox;
				info.rightBounds = rightBox;
				info.dim = dim;
				bestTieBreak = tieBreak;
			}
		} // END LOOP SPLIT POINTS
	} // END LOOP AXES

	assert(info.cost != FLT_MAX);
	assert(info.i > -1);

	return info;
}

// Object split cheapest => just sort triangles, update reference ranges
void SBVH::partitionObject(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SplitInfo& info)
{
	assert(info.dim > -1);
	
	int start = m_refs.size() - spec.refs;
	int end = m_refs.size() - 1;
	sortReferences(start, end, info.dim);

	left.refs = info.i;
	left.box = info.leftBounds;
	right.refs = spec.refs - info.i;
	right.box = info.rightBounds;
}

// Find cheapest spatial split using binned SAH
// 1. Chop triangles into bins, update bin bounds + triangle counts
// 2. Build area lookup (per bin boundary), calculate SAH, keep cheapest
SBVH::SplitInfo SBVH::binSplit(const NodeSpec& spec, F32 nodeSAH)
{
	float3 origin = spec.box.min;
	float3 binSize = (spec.box.max - origin) * (1.0f / (F32)NumSpatialBins);
	float3 invBinSize = 1.0f / binSize;

	// Init bins
	for (int dim = 0; dim < 3; dim++)
	{
		for (int i = 0; i < NumSpatialBins; i++) 
		{
			Bin& bin = bins[dim][i];
			bin.bounds = AABB_t();
			bin.entering = 0;
			bin.exiting = 0;
		}
	}

	// Perform chopped binning on spanned triangles
	for (int refIdx = m_refs.size() - spec.refs; refIdx < m_refs.size(); refIdx++)
	{
		const TriRef& ref = m_refs[refIdx];

		// Find bins that AABB overlaps
		int3 firstBin = vclamp(int3((ref.box.min - origin) * invBinSize), 0, NumSpatialBins - 1);
		int3 lastBin = vclamp(int3((ref.box.max - origin) * invBinSize), firstBin, NumSpatialBins - 1);

		// Clip AABB against bins, expand their boxes
		for (int dim = 0; dim < 3; dim++)
		{
			TriRef currRef = ref;
			for (int i = firstBin[dim]; i < lastBin[dim]; i++)
			{
				TriRef leftRef, rightRef;
				F32 splitCoord = origin[dim] + binSize[dim] * (F32)(i + 1);
				splitReference(leftRef, rightRef, currRef, dim, splitCoord);
				bins[dim][i].bounds.expand(leftRef.box);
				currRef = rightRef;
			}

			bins[dim][lastBin[dim]].bounds.expand(currRef.box);
			bins[dim][firstBin[dim]].entering++;
			bins[dim][lastBin[dim]].exiting++;
		}
	}

	// Select best split plane
	SplitInfo split;
	for (int dim = 0; dim < 3; dim++)
	{
		// Build AABB lookup, per bin boundary
		AABB_t rightBounds;
		for (int i = NumSpatialBins - 1; i > 0; i--)
		{
			rightBounds.expand(bins[dim][i].bounds);
			rightBoxes[i - 1] = rightBounds;
		}

		// Sweep left to right and select lowest SAH
		AABB_t leftBounds;
		int leftNum = 0;
		int rightNum = spec.refs;

		for (int i = 1; i < NumSpatialBins; i++)
		{
			leftBounds.expand(bins[dim][i - 1].bounds);
			leftNum += bins[dim][i - 1].entering;
			rightNum -= bins[dim][i - 1].exiting;

			F32 leftArea = leftBounds.area();
			F32 rightArea = rightBoxes[i - 1].area();
			F32 sah = nodeSAH + leftArea * (leftNum) * 1 + rightArea * (rightNum) * 1;
			if (sah < split.cost)
			{
				split.cost = sah;
				split.dim = dim;
				split.pos = origin[dim] + binSize[dim] * (F32)i;
			}
		}
	}
	return split;
}

// Spatial split was cheapest => distribute references (while potentially splitting)
// Only chosen cheapest split dimension is considered
void SBVH::partitionSpatial(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SplitInfo& split)
{
	// Left-hand side:      [leftStart, leftEnd[
	// Uncategorized/split: [leftEnd, rightStart[
	// Right-hand side:     [rightStart, m_refs.size()[

	int leftStart = m_refs.size() - spec.refs;
	int leftEnd = leftStart;
	int rightStart = m_refs.size();
	left.box = right.box = AABB_t();

	// Scan refs, swap non-intersecting tris to their corresponding sides
	// Results in a range of intersecting references in the middle
	for (int i = leftEnd; i < rightStart; i++)
	{
		// Entirely on the left-hand side?
		if (m_refs[i].box.max[split.dim] <= split.pos)
		{
			left.box.expand(m_refs[i].box);
			std::swap(m_refs[i], m_refs[leftEnd++]);
		}
		// Entirely on the right-hand side?
		else if (m_refs[i].box.min[split.dim] >= split.pos)
		{
			right.box.expand(m_refs[i].box);
			std::swap(m_refs[i--], m_refs[--rightStart]);
		}
	}

	// Process middle range, duplicating or unsplitting references
	while (leftEnd < rightStart)
	{
		TriRef lref, rref;
		splitReference(lref, rref, m_refs[leftEnd], split.dim, split.pos);

		// Check how unsplitting / duplicating affects existing AABBs
		AABB_t lub = left.box;  // left unsplit
		AABB_t rub = right.box; // right unsplit
		AABB_t ldb = left.box;  // left duplicate
		AABB_t rdb = right.box; // right duplicate
		lub.expand(m_refs[leftEnd].box);
		rub.expand(m_refs[leftEnd].box);
		ldb.expand(lref.box);
		rdb.expand(rref.box);

		F32 lac = sahParams.costTri * (leftEnd - leftStart);
		F32 rac = sahParams.costTri * (m_refs.size() - rightStart);
		F32 lbc = sahParams.costTri * (leftEnd - leftStart + 1);
		F32 rbc = sahParams.costTri * (m_refs.size() - rightStart + 1);

		F32 unsplitLeftSAH = lub.area() * lbc + right.box.area() * rac;
		F32 unsplitRightSAH = left.box.area() * lac + rub.area() * rbc;
		F32 duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
		F32 minSAH = std::min(unsplitLeftSAH, std::min(unsplitRightSAH, duplicateSAH));

		if (minSAH == unsplitLeftSAH)
		{
			left.box = lub;
			leftEnd++;
		}
		else if (minSAH == unsplitRightSAH)
		{
			right.box = rub;
			std::swap(m_refs[leftEnd], m_refs[--rightStart]);
		}
		else
		{
			left.box = ldb;
			right.box = rdb;
			m_refs[leftEnd++] = lref;
			m_refs.push_back(rref);
		}
	}

	left.refs = leftEnd - leftStart;
	right.refs = m_refs.size() - rightStart;
}

// Split triangle (reference) into two references based on bin boundary coord
void SBVH::splitReference(TriRef& left, TriRef& right, const TriRef& ref, int dim, F32 coord)
{
	left.ind = right.ind = ref.ind;
	left.box = right.box = AABB_t();

	const U32 offsets[] = { 2, 0, 1 };
	const RTTriangle &tri = (*m_triangles)[ref.ind];
	const VertexPNT verts[3] = { tri.v0, tri.v1, tri.v2 };

	// Compare each vertex against plane that splits left and right bins
	for (int i = 0; i < 3; i++)
	{
		const VertexPNT *v1v = verts + i;
		const VertexPNT *v2v = verts + offsets[i];
		float3 p1 = v2v->p;
		float3 p2 = v1v->p;
		F32 v0p = p1[dim];
		F32 v1p = p2[dim];

		// Left or right box?
		if (v0p <= coord)
			left.box.expand(p1);
		if (v0p >= coord)
			right.box.expand(p1);

		// Check if edge intersects plane (both box AABBs have to be expanded)
		if ((v0p < coord && v1p > coord) || (v0p > coord && v1p < coord))
		{
			float3 t = lerp(p1, p2, std::max(0.0f, std::min(1.0f, (coord - v0p) / (v1p - v0p))));
			left.box.expand(t);
			right.box.expand(t);
		}
	}

	// Intersect with original bounds
	left.box.max[dim] = coord;
	right.box.min[dim] = coord;
	left.box.intersect(ref.box);
	right.box.intersect(ref.box);
}
