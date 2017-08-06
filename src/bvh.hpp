#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <fstream>
#include <cassert>

#include "triangle.hpp"
#include "bvhnode.hpp"
#include "rtutil.hpp"

template <class A, class B> A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

class BVH
{

friend class CLContext;

public:
    BVH(std::vector<RTTriangle> *tris, SplitMode mode);
    BVH(std::vector<RTTriangle> *tris, const std::string filename);
	BVH(void) {}
	~BVH() {}

    void exportTo(const std::string filename) const;

    AABB_t getSceneBounds(void) const;

private:
	void build(U32 nInd, U32 depth, F32 progressStart, F32 progressEnd);

protected:
	struct SplitInfo;
	
	
    void importFrom(const std::string filename);
	void lazyPrintBuildStatus(F32 percentage);

    // Convert build nodes to small nodes
    void createSmallNodes();
	void createIndexList();

    //bool partition(BuildNode &n, U32 &split); // writes index of first element of second group into split
    //F32 centroidSplit(U32 iStart, U32 iEnd, U32 dimension);

	bool partition(BuildNode &n, SplitInfo &split); // writes index of first element of second group into split
	bool spatialMedianSplit(BuildNode &n, SplitInfo &split);
	bool objectMedianSplit(BuildNode &n, SplitInfo &split);
	bool objectMedianSplit(BuildNode &n, U32 dim, SplitInfo &split);
	bool sahSplit(BuildNode &n, SplitInfo &split);
	void sortReferences(U32 s, U32 e, U32 dim);

	F32 sahCost(U32 N1, F32 area1, U32 N2, F32 area2, F32 area_root) const;
	void buildBoxLookup(BuildNode &n);
	AABB_t centroudBounds(std::vector<TriRef>::const_iterator begin, std::vector<TriRef>::const_iterator end) const;

	std::vector<RTTriangle>* m_triangles;
	std::vector<U32> m_indices;
	std::vector<TriRef> m_refs;
	std::vector<BuildNode> m_build_nodes;
	std::vector<Node> m_nodes;
	std::vector<AABB_t> rightBoxes; // SAH builder optimization
	U32 nodes = 0;
	SplitMode m_mode;

	enum
	{
		MaxLeafElems = 8,
		MaxDepth = 64
	};

	struct
	{
		const F32 costBox = 1.0f;
		const F32 costTri = 1.0f;
	} sahParams;

	struct
	{
		U32 depth = 0;
		U32 bad_splits = 0;
		U32 splits = 0;
	} metrics;

	struct SplitInfo
	{
		S32 i;
		F32 pos; // object split
		S32 dim; // spatial split
		F32 cost;
		AABB_t leftBounds;
		AABB_t rightBounds;

		SplitInfo(void) : i(-1), dim(-1), cost(FLT_MAX) {}
	};

	S32 buildPercentage = -1; // for printing sparingly
};

// Write a simple data type to a stream.
template<class T>
std::ostream &write(std::ostream &stream, const T &x)
{
    return stream.write(reinterpret_cast<const char *>(&x), sizeof(x));
}

// Read a simple data type from a stream.
template<class T>
std::istream &read(std::istream &os, T &x)
{
    return os.read(reinterpret_cast<char *>(&x), sizeof(x));
}
