#include "scene.hpp"

Scene::Scene(const std::string filename)
{
    loadModel(filename); // Assume file is a model, not a scene. Just for now ;)
}

inline bool endsWith(const std::string s, const std::string end) {
    size_t len = end.size();
    if (len > s.size()) return false;

    std::string substr = s.substr(s.size() - len, len);
    return end == substr;
}

void Scene::loadModel(const std::string filename)
{
    if (endsWith(filename, "obj"))
    {
        std::cout << "Loading OBJ file: " << filename << std::endl;
        this->loadObjModel(filename);
    }
    else if (endsWith(filename, "ply"))
    {
        std::cout << "Loading PLY file: " << filename << std::endl;
        this->loadPlyModel(filename);
    }
    else
    {
        std::cout << "Cannot load file " << filename << ": unknown file format" << std::endl;
        exit(1);
    }
}

void Scene::loadObjModel(const std::string filename)
{
    std::vector<float3> positions, normals;
    std::vector<std::array<unsigned, 6>> faces;

    // Open input file stream for reading.
    std::ifstream input(filename, std::ios::in);

    if(!input)
    {
        std::cout << "Could not open file: " << filename << ", exiting..." << std::endl;
        exit(1);
    }

    // Read the file line by line.
    std::string line;
    while(getline(input, line))
    {
        // Use space as separator
        for (auto& c : line)
        {
            if (c == '/') c = ' ';
        }

        // Temporary objects to read data into
        std::array<unsigned, 6>  f; // Face index array
        std::string              s;

        std::istringstream iss(line);

        // Read object type into s
        iss >> s;

        if (s == "v")
        {
            float x, y, z;
            iss >> x >> y >> z;
            positions.push_back(float3(x, y, z));
        }
        else if (s == "vn")
        {
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            normals.push_back(float3(nx, ny, nz));
        }
        else if (s == "f")
        {
            // Face data is in the format `f v1/vt1/vn1 v2/vt2/vn2 ...`
            // (vertex index, texture index, normal index)
            unsigned sink; // Texture indices ignored for now

            iss >> f[0] >> sink >> f[1] >> f[2] >> sink >> f[3] >> f[4] >> sink >> f[5];

            // Obj indices start from 1, need to be shifted
            for (unsigned& v : f)
            {
                v -= 1;
            }

            faces.push_back(f);
        }
    }

    unpackIndexedData(positions, normals, faces, false);
}

void Scene::loadPlyModel(const std::string filename)
{
    (void)filename;
    std::cout << "Not implemented!" << std::endl;
    exit(1);
}

void Scene::unpackIndexedData(const std::vector<float3> &positions,
                              const std::vector<float3>& normals,
                              const std::vector<std::array<unsigned, 6>>& faces,
                              bool type_ply)
{
    std::cout << "Unpacking mesh" << std::endl;

    std::cout << "Positions: " << positions.size() << std::endl;
    std::cout << "Normals: " << normals.size() << std::endl;
    std::cout << "Faces: " << faces.size() << std::endl;

    VertexPNT v0, v1, v2;

    for (auto& f : faces)
    {
        // f[0] = index of the position of the first vertex
        // f[1] = index of the normal of the first vertex
        // f[2] = index of the position of the second vertex
        // ...

        v0.p = positions[f[0]];
        v1.p = positions[f[2]];
        v2.p = positions[f[4]];

        if (normals.size() == 0)
        {
            // Generate normals
            v0.n = v1.n = v2.n = normalize(cross(v1.p - v0.p, v2.p - v0.p));
        }
        else if(type_ply)
        {
            // PLY-normals have the same indices as their corresponding vertices
            v0.n = normals[f[0]];
            v1.n = normals[f[2]];
            v2.n = normals[f[4]];
        }
        else
        {
            // Use pre-calculated normals for OBJ
            v0.n = normals[f[1]];
            v1.n = normals[f[3]];
            v2.n = normals[f[5]];
        }

        triangles.push_back(RTTriangle(v0, v1, v2));
    }
};