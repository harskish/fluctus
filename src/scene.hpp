#pragma once

// Suppress sscanf warning on MSVCCompiler
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <string>
#include <vector>
#include <map>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <chrono>

#include "tiny_obj_loader.h"
#include "geom.h"
#include "triangle.hpp"
#include "envmap.hpp"
#include "rtutil.hpp"
#include "math/float3.hpp"
#include "texture.hpp"
#include "settings.hpp"
#include "bxdf_types.h"

using FireRays::float3;
class ProgressView;

class Scene {
public:
    Scene(const std::string filename);
    ~Scene();

    void loadEnvMap(const std::string filename);
    void setEnvMap(std::shared_ptr<EnvironmentMap> envMapPtr);
    void loadModel(const std::string filename, ProgressView *progress); // load .obj or .ply model

    std::vector<RTTriangle> &getTriangles() { return triangles; }
    std::vector<Material> &getMaterials() { return materials; }
    std::vector<Texture*> &getTextures() { return textures; }
    std::shared_ptr<EnvironmentMap> getEnvMap() { return envmap; }

    std::string hashString();

private:
    void loadObjModel(const std::string filename);
    void loadPlyModel(const std::string filename);

    // With tiny_obj_loader
    void loadObjWithMaterials(const std::string filename, ProgressView *progress);
    cl_int tryImportTexture(const std::string path, const std::string name);
    cl_int parseShaderType(std::string &type);

    void unpackIndexedData(const std::vector<float3> &positions,
                           const std::vector<float3>& normals,
                           const std::vector<std::array<unsigned, 6>>& faces,
                           bool type_ply);

  void computeHash(const std::string filename);

  std::shared_ptr<EnvironmentMap> envmap;
  std::vector<RTTriangle> triangles;
  std::vector<Material> materials;
  std::vector<Texture*> textures;
  size_t hash;
};
