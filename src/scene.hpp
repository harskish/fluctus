#pragma once

#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include "rttriangle.hpp"
#include "math/float3.hpp"

using FireRays::float3;

class Scene {
public:
    Scene(const std::string filename);
    ~Scene() = default;

    std::vector<RTTriangle> &getTriangles() { return triangles; }

private:
    void loadModel(const std::string filename); // load .obj or .ply model
    void loadObjModel(const std::string filename);
    void loadPlyModel(const std::string filename);

    void unpackIndexedData(const std::vector<float3> &positions,
                           const std::vector<float3>& normals,
                           const std::vector<std::array<unsigned, 6>>& faces,
                           bool type_ply);

    std::vector<RTTriangle> triangles;
};
