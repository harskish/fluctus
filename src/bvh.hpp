#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <fstream>

#include "triangle.hpp"
#include "bvhnode.hpp"
#include "rtutil.hpp"

template <class A, class B> A lerp(const A& a, const A& b, const B& t) { return (A)(a * ((B)1 - t) + b * t); }

class BVH {

friend class Tracer;

public:
    BVH(std::vector<RTTriangle> *tris, SplitMode mode);
    BVH(std::vector<RTTriangle> *tris, const std::string filename);
    ~BVH() { }

    void exportTo(const std::string filename) const;

private:
    void build(U32 nInd, U32 depth, F32 progressStart, F32 progressEnd);
    void importFrom(const std::string filename);

    // Convert build nodes to small nodes
    void createSmallNodes();

    bool sortElems(BuildNode &n, U32 &split); // writes index of first element of second group into split
    F32 centroidSplit(U32 iStart, U32 iEnd, U32 dimension);

    bool spatialMedianSplit(BuildNode &n, U32 &split);
    bool objectMedianSplit(BuildNode &n, U32 &split);
    bool objectMedianSplit(BuildNode &n, U32 dim, U32 &split);
    bool sahSplit(BuildNode &n, U32 &split);

    F32 sahCost(U32 N1, F32 area1, U32 N2, F32 area2, F32 area_root) const;

    std::vector<F32> buildAreaLookup(BuildNode &n) const;

    std::vector<RTTriangle> *m_triangles;
    std::vector<U32> m_indices;
    std::vector<BuildNode> m_build_nodes;
    std::vector<Node> m_nodes;
    U32 nodes = 0;
    SplitMode m_mode;

    const U32 MAX_LEAF_ELEMS = 6;

    struct {
        const F32 costBox = 1;
        const F32 costTri = 4;
    } sahParams;

    struct {
        U32 depth = 0;
        U32 leaves = 0;
        U32 bad_splits = 0;
        U32 splits = 0;
    } metrics;
};

// Write a simple data type to a stream.
template<class T>
std::ostream &write(std::ostream &stream, const T &x) {
    return stream.write(reinterpret_cast<const char *>(&x), sizeof(x));
}

// Read a simple data type from a stream.
template<class T>
std::istream &read(std::istream &os, T &x) {
    return os.read(reinterpret_cast<char *>(&x), sizeof(x));
}
