#include <iostream>
#include <cfloat>
#include <cassert>
#include "bvh.hpp"

BVH::BVH(std::vector<RTTriangle>* tris, SplitMode mode)
{
    m_triangles = tris;
    m_mode = mode;

	// Setup references for building
	m_refs.resize(m_triangles->size());
	for (int i = 0; i < m_triangles->size(); i++)
	{
		m_refs[i] = TriRef(i, (*m_triangles)[i]);
	}

	// Shared vector to avoid reallocations
	rightBoxes.resize(m_triangles->size());

	BuildNode root(0, (U32)m_triangles->size() - 1, -1);
	m_build_nodes.push_back(root);
	nodes++;
    
	build(0, 0, 0.0f, 1.0f); // root, depth 0
	printf("\rBVH builder: progress 100%%\n");
	assert(m_build_nodes[0].rightChild != -1);
	assert(metrics.depth <= MaxDepth);
	assert(m_indices.size() == 0);

	if (metrics.depth > MaxDepth)
		std::cout << "WARN: BVH might not fit traversal stack! (" << metrics.depth << " > " << MaxDepth << ")" << std::endl;

	createIndexList();
	createSmallNodes();

	std::cout
		<< "======================" << std::endl
		<< ((m_mode == SplitMode::SAH) ? "SAH" : (m_mode == SplitMode::ObjectMedian)
            ? "Object Median" : (m_mode == SplitMode::SpatialMedian) ? "Spatial Median" : "Unknown") << std::endl
		<< "Splits: " << metrics.splits << " (" << int(metrics.bad_splits / float(metrics.splits) * 100.0f) << "% bad)" << std::endl
		<< "Depth: " << metrics.depth << std::endl
		<< "Leaves: " << metrics.splits + 1 << std::endl
		<< "======================" << std::endl;
}

BVH::BVH(std::vector<RTTriangle>* tris, const std::string filename)
{
    m_triangles = tris;
    importFrom(filename);
}

AABB_t BVH::getSceneBounds(void) const
{
    if (m_nodes.size() == 0)
        throw std::runtime_error("Cannot get scene bounds from uninitialized BVH");

    return m_nodes[0].box;
}

void BVH::createSmallNodes()
{
	m_nodes.clear();

	for (BuildNode bn : m_build_nodes)
	{
		Node n;
		n.box = bn.box;
        n.parent = bn.parent;
		if (bn.rightChild == -1) { // leaf node
			n.iStart = bn.iStart;
			U32 sp = bn.spannedTris();
			if (sp > std::numeric_limits<U8>::max())
				throw std::runtime_error("Too many prims to fit into U8!");
			n.nPrims = (U8)(sp);
		}
		else // interior node
		{ 
			n.rightChild = (U32)(bn.rightChild);
		}
		m_nodes.push_back(n);
	}
	std::cout << "Small node vector created" << std::endl;

	// Deallocate build node memory
	m_build_nodes.clear();
	m_build_nodes.shrink_to_fit();
}

void BVH::createIndexList() {
	m_indices.resize(m_refs.size());
	for (int i = 0; i < m_refs.size(); i++)
	{
		m_indices[i] = m_refs[i].ind;
	}

	// No longer needed
	m_refs.clear();
	m_refs.shrink_to_fit();
}

std::vector<U32> importIndices(std::ifstream &in)
{
    U32 size;
    read(in, size);
    std::vector<U32> vec(size);
    
    for (U32 i = 0; i < size; i++)
	{
        U32 ind;
        read(in, ind);
        vec[i] = ind;
    }

    // Return value optimization!
    return vec;
}

std::vector<Node> importNodes(std::ifstream &in)
{
	U32 size;
	read(in, size);
	std::vector<Node> vec(size);

	for (U32 i = 0; i < size; i++)
	{
		Node n;
		fr::float3 bmin, bmax;
		read(in, bmin.x);
		read(in, bmin.y);
		read(in, bmin.z);
		read(in, bmax.x);
		read(in, bmax.y);
		read(in, bmax.z);
		n.box = AABB_t(bmin, bmax);
		read(in, n.iStart);
        read(in, n.parent);
		read(in, n.nPrims);
		vec[i] = n;
	}

	// Return value optimization!
	return vec;
}


void BVH::importFrom(const std::string filename)
{
    std::ifstream infile(filename, std::ios::binary);
    m_indices = importIndices(infile);
    m_nodes = importNodes(infile);
}


void exportNode(std::ofstream &out, const Node &n)
{
	// AABB
    fr::float3 bmin = n.box.min;
    fr::float3 bmax = n.box.max;
	write(out, bmin.x);
	write(out, bmin.y);
	write(out, bmin.z);
	write(out, bmax.x);
	write(out, bmax.y);
	write(out, bmax.z);

	// Parameters
	write(out, n.iStart);
    write(out, n.parent);
	write(out, n.nPrims);
}

/** Write BVH to file for later importing **/
void BVH::exportTo(const std::string filename) const
{
    std::ofstream out(filename, std::ios::binary);

    if (out.good())
    {
		// Index list
		write(out, (U32)m_indices.size());
		for_each(m_indices.begin(), m_indices.end(), [&out](U32 index) { write(out, index); });

		// Node list
		write(out, (U32)m_indices.size());
		for_each(m_nodes.begin(), m_nodes.end(), [&out](const Node &n) { exportNode(out, n); });
    }
    else
    {
        std::cout << "Could not create create file for BVH export!" << std::endl;
    }
}

// Too frequent printing is actually a bottleneck!
void BVH::lazyPrintBuildStatus(F32 progress)
{
	S32 percentage = S32(ceil(progress * 100.0f));
	if (percentage > buildPercentage)
	{
		buildPercentage = percentage;
		printf("\rBVH builder: progress %d%%", percentage);
	}
}

void BVH::build(U32 nInd, U32 depth, F32 progressStart = 0.0f, F32 progressEnd = 1.0f)
{
	lazyPrintBuildStatus(progressStart);

	m_build_nodes[nInd].computeBB(m_refs);
	metrics.depth = std::max(metrics.depth, depth);
	U32 elems = m_build_nodes[nInd].spannedTris();
	if (elems > MaxLeafElems)
	{
		SplitInfo info;
		bool shouldSplit = partition(m_build_nodes[nInd], info);

		if (!shouldSplit) { // parent is cheaper (SAH)
			assert(elems <= std::numeric_limits<U8>::max());
			return;
		}

		metrics.splits++;

		// Statistics for progress bar
		F32 progressMid = lerp(progressStart, progressEnd, (F32)(info.i - m_build_nodes[nInd].iStart) / (F32)(elems));

		// Left child
		m_build_nodes.push_back(BuildNode(m_build_nodes[nInd].iStart, info.i, nInd));
		nodes++;
		build(nodes - 1, depth + 1, progressStart, progressMid); // last pushed

		// Right child
		m_build_nodes.push_back(BuildNode(info.i + 1, m_build_nodes[nInd].iEnd, nInd));
		m_build_nodes[nInd].rightChild = nodes++;
		build(nodes - 1, depth + 1, progressMid, progressEnd);
	}
}

bool BVH::partition(BuildNode &n, SplitInfo &split)
{
	switch (m_mode) {
	case SplitMode::SAH:
		return sahSplit(n, split);
		break;
	case SplitMode::SpatialMedian:
		return spatialMedianSplit(n, split);
		break;
	case SplitMode::ObjectMedian:
		return objectMedianSplit(n, split);
		break;
	default:
		std::cout << "Selected split mode not implemented!" << std::endl;
		throw std::runtime_error("Selected split mode not implemented!");
		break;
	}
}

void BVH::sortReferences(U32 s, U32 e, U32 dim)
{
	auto start = m_refs.begin() + s;
	auto end = m_refs.begin() + e + 1;

	// Sort the range [s, e[ by triangle centroid

	//std::sort(start, end, [dim](TriRef &r1, TriRef &r2) { return r1.pos[dim] < r2.pos[dim]; });

	std::sort(start, end, [dim](TriRef &r1, TriRef &r2) {
		F32 ca = r1.box.min[dim] + r1.box.max[dim];
		F32 cb = r2.box.min[dim] + r2.box.max[dim];
		return (ca < cb || (ca == cb && r1.ind < r2.ind));
	});
}

bool BVH::objectMedianSplit(BuildNode &n, SplitInfo &split)
{
	return objectMedianSplit(n, n.box.maxDim(), split);
}

bool BVH::objectMedianSplit(BuildNode &n, U32 dim, SplitInfo &split)
{
	sortReferences(n.iStart, n.iEnd, dim);
	split.i = n.iStart + n.spannedTris() / 2;
	return true;
}

// Use centroids bounds in spatial median split (reduces bad split amount)
AABB_t BVH::centroudBounds(std::vector<TriRef>::const_iterator begin, std::vector<TriRef>::const_iterator end) const
{
	AABB_t bounds;
	for (auto i = begin; i != end; i++) {
		bounds.min = vmin(bounds.min, i->pos);
		bounds.max = vmax(bounds.max, i->pos);
	}

	return bounds;
}

bool BVH::spatialMedianSplit(BuildNode &n, SplitInfo &info)
{
	auto s = m_refs.begin() + n.iStart;
	auto e = m_refs.begin() + n.iEnd;

	AABB_t centroidBox = centroudBounds(s, e);
	U32 dim = centroidBox.maxDim();
	F32 splitCoord = centroidBox.centroid()[dim];

	// Partition the range [s, e[
	auto it = std::partition(s, e, [dim, splitCoord](TriRef &r) { return r.pos[dim] < splitCoord; });
	info.i = (U32)(it - m_refs.begin()); // Index of split point in index list (first elem. of second group)

	// Fix bad splits (use midpoint)
	if (info.i == n.iStart || info.i == n.iEnd)
	{
		metrics.bad_splits++;
		objectMedianSplit(n, info);
	}

	return true;
}

F32 BVH::sahCost(U32 N1, F32 area1, U32 N2, F32 area2, F32 area_root) const
{
	F32 lcost = N1 * area1 / area_root;
	F32 rcost = N2 * area2 / area_root;
	return 2 * sahParams.costBox + sahParams.costTri * (lcost + rcost);
}

// lookup[n] = AABB with last n + 1 triangles
void BVH::buildBoxLookup(BuildNode &n)
{
	AABB_t box;
	for (U32 i = 0; i < n.spannedTris(); i++)
	{
		box.expand(m_refs[n.iEnd - i].box);
		rightBoxes[i] = box;
	}
}

bool BVH::sahSplit(BuildNode &n, SplitInfo &info)
{
	F32 parentArea = n.box.area();
	assert(parentArea > 0.0f);

	F32 parentCost = sahParams.costBox + n.spannedTris() * sahParams.costTri;

	// Loop over all three axes to find best split
	for (U32 dim = 0; dim < 3; dim++) {
		// Sort along axis
		sortReferences(n.iStart, n.iEnd, dim);

		// Create AABB lookup
		buildBoxLookup(n);

		AABB_t leftBox;
		U32 leftCount = 0;
		U32 spanSize = n.spannedTris();

		// Try different split points along axis
		for (U32 s = n.iStart; s < n.iEnd; s++) // exclude last (all on left side)
		{
			leftBox.expand(m_refs[s].box);
			leftCount++;

			AABB_t &rightBox = rightBoxes[n.iEnd - s - 1];
			F32 areaRight = rightBox.area();
			F32 areaLeft = leftBox.area();

			// New best split?
			F32 cost = sahCost(leftCount, areaLeft,
				spanSize - leftCount, areaRight, parentArea);

			if (cost < info.cost)
			{
				info.cost = cost;
				info.i = s;
				info.leftBounds = leftBox;
				info.rightBounds = rightBox;
				info.dim = dim;
			}
		} // END LOOP SPLIT POINTS
	} // END LOOP AXES

	assert(info.cost != FLT_MAX);
	assert(info.i > -1);

	// Worse than parent?
	if (info.cost > parentCost && n.spannedTris() < MaxLeafElems)
		return false;

	// Re-sort along best axis, if necessary
	if (info.dim != 2)
		sortReferences(n.iStart, n.iEnd, info.dim);

	// Fix indexing if only one triangle on either side
	if (info.i == n.iStart)
	{
		info.i++;
		metrics.bad_splits++;
	}
	else if (info.i == n.iEnd)
	{
		info.i--;
		metrics.bad_splits++;
	}

	return true;
}
