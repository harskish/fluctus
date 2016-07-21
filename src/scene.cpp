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

/* Used for loading PLY meshes */
void Scene::loadPlyModel(const std::string filename)
{
    struct Element
    {
        std::string name;			     // e.g. vertex
        int lines;				         // e.g. 1300
        std::vector<std::string> props;  // e.g. [x, y, z, nx, ny, nz]
    };

    // Keep track of what data, and how meny elements each, to expect
    std::vector<Element> elements;

    // Open input file stream for reading.
    std::ifstream input(filename, std::ios::in);

    std::string line;

    // Data of current element
    std::string type = "none";
    int num_elem = 0;
    std::vector<std::string> currentProps;

    /* READ HEADERS */
    while (getline(input, line))
    {
        std::istringstream iss(line);
        std::string s;
        iss >> s;
        if (s == "element")
        {
            elements.push_back(Element{ type, num_elem, currentProps }); //Push previous element
            currentProps.clear();
            iss >> type >> num_elem;
        }
        else if (s == "property")
        {
            std::string type, name;
            iss >> type >> name;
            currentProps.push_back(name);
        }
        else if (s == "end_header")
        {
            elements.push_back(Element{ type, num_elem, currentProps }); //push last element
            break;
        }
    }

    std::cout << "PLY headers processed" << std::endl;

    std::vector<float3> positions, normals;
    std::vector<std::array<unsigned, 6>> faces;

    /* READ DATA */
    for (Element &e : elements)
    {
        if (e.name == "vertex")
        {
            std::cout << "Reading " << e.lines << " vertices" << std::endl;
            for (int i = 0; i < e.lines; i++)
            {
                getline(input, line);
                std::istringstream iss(line);

                std::map<std::string, float> map;
                float bucket;
                for (std::string name : e.props)
                {
                    iss >> bucket;
                    map[name] = bucket;
                }

                positions.push_back(float3(map["x"], map["y"], map["z"]));
                if (map.find("nx") != map.end()) //contains normals
                    normals.push_back(float3(map["nx"], map["ny"], map["nz"]));
            }
        }
        else if (e.name == "face")
        {
            std::cout << "Reading " << e.lines << " faces" << std::endl;
            for (int i = 0; i < e.lines; i++)
            {
                getline(input, line);
                std::istringstream iss(line);

                std::array<unsigned, 6>  f; // Face index array

                // Face list format: '3 i1 i2 i3' (triangle) or '4 i1 i2 i3 i4' (quad)
                // A quad can be represented with two triangles

                int poly_type;
                iss >> poly_type;

                if (poly_type == 3)
                {
                    iss >> f[0] >> f[2] >> f[4];
                    faces.push_back(f);
                }
                else if (poly_type == 4)
                {
                    int i0, i1, i2, i3;
                    iss >> i0 >> i1 >> i2 >> i3;

                    //triangle 1
                    f[0] = i0; f[2] = i1; f[4] = i2;
                    faces.push_back(f);

                    //triangle 2
                    f[0] = i2; f[2] = i3; f[4] = i0;
                    faces.push_back(f);
                }
            }
        }
        else
        {
            //skip data
            std::cout << "Skipping element of type " << e.name << std::endl;
            for (int i = 0; i < e.lines; i++)
            {
                getline(input, line);
            }
        }
    }

    unpackIndexedData(positions, normals, faces, true); //true = ply format
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