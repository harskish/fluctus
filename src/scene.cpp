#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include "pbrtParser/Scene.h"

#include "scene.hpp"
#include "progressview.hpp"
#include "utils.h"
#include "bxdf_types.h"

#include <set>

Scene::Scene()
{
    // Init default material
    Material def;
    def.Kd = float3(0.64, 0.64, 0.64);
    def.Ni = 1.8f;
    def.Ns = 700.0f;
    def.map_Kd = -1;
    def.map_Ks = -1;
    def.map_N = -1;
    def.type = BXDF_DIFFUSE;
    materials.push_back(def);
    materialTypes |= def.type;
}

Scene::~Scene()
{
    for (Texture *t : textures)
    {
        delete t;
    }
}

void Scene::loadEnvMap(const std::string filename)
{
    envmap.reset(new EnvironmentMap(filename));
}

void Scene::setEnvMap(std::shared_ptr<EnvironmentMap> envMapPtr)
{
    envmap = envMapPtr;
}

std::string Scene::hashString()
{
    std::stringstream ss;
    ss << this->hash;
    return ss.str();
}

void Scene::loadModel(const std::string filename, ProgressView *progress)
{
    // Starting time for model loading
    auto time1 = std::chrono::high_resolution_clock::now();

    if (endsWith(filename, "obj"))
    {
        std::cout << "Loading OBJ file: " << filename << std::endl;
        loadObjWithMaterials(filename, progress);
    }
    else if (endsWith(filename, "ply"))
    {
        std::cout << "Loading PLY file: " << filename << std::endl;
        loadPlyModel(filename);
    }
    else if (endsWith(filename, "pbf"))
    {
        std::cout << "Loading PBRT binary file: " << filename << std::endl;
        loadPBFModel(filename);
    }
    else if (endsWith(filename, "pbrt"))
    {
        const std::string converted = filename.substr(0, filename.length() - 4) + "pbf";

        std::ifstream infile(converted);
        if (!infile.good())
        {
            progress->showMessage("Converting PBRT to binary");
            std::cout << "Converting PBRT file to PBF: " << filename << std::endl;
            convertPBRTModel(filename, converted);
        }        
        
        infile.close();
        progress->showMessage("Loading PBRT binary file");
        std::cout << "Loading PBRT binary file: " << converted << std::endl;
        loadPBFModel(converted);
    }
    else
    {
        std::cout << "Cannot load file " << filename << ": unknown file format" << std::endl;
        waitExit();
    }

    this->hash = fileHash(filename);

    // Print elapsed time
    auto time2 = std::chrono::high_resolution_clock::now();
    std::cout << "Mesh loaded in: "
        << std::chrono::duration<double, std::milli>(time2 - time1).count()
        << " ms" << std::endl;
}


// Possible face data formats include:
//  `f v1/vt1/vn1 v2/vt2/vn2 ...`
//  `f v1//vn1 v2//vn2 ...`
//  `f v1 v2 ...`
//  `f -v1 -v2 ...`
inline void setFaceFormat(int &format, std::string &format_string, bool &negatives, std::istringstream &line)
{
    if (format > -1) return; // Only set once
    
    std::string block1;
    line >> block1;

    bool textures = false;
    bool normals = false;
    int ind = 0;
    for (char c : block1)
    {
        if (c == '-')
            negatives = true;
        if (c == '/')
            ind++;
        else if (ind == 0)
            continue;
        else if (ind == 1)
            textures = true;
        else if (ind == 2)
            normals = true;
        else
            std::cout << "Unknown OBJ face format, too many indices!" << std::endl;
    }

    if (textures && normals)
    {
        format = 0;
        format_string = "f %u/%u/%u %u/%u/%u %u/%u/%u";
    }
    else if (!textures && normals)
    {
        format = 1;
        format_string = "f %u//%u %u//%u %u//%u";
    }
    else if (textures && !normals)
    {
        format = 2;
        format_string = "f %u/%u %u/%u %u/%u";
    }
    else
    {
        format = 3;
        format_string = "f %u %u %u";
    }

    // Replace "%u" with "-%u"
    if (negatives)
    {
        std::stringstream res;
        for (char c : format_string)
        {
            if (c == '%') res << '-';
            res << c;
        }
        format_string = res.str();
    }
}

cl_int Scene::parseShaderType(std::string type)
{
    if (type == "diffuse")
        return BXDF_DIFFUSE;
    if (type == "glossy")
        return BXDF_GLOSSY;
    if (type == "rough_reflection")
        return BXDF_GGX_ROUGH_REFLECTION;
    if (type == "ideal_reflection")
        return BXDF_IDEAL_REFLECTION;
    if (type == "rough_dielectric")
        return BXDF_GGX_ROUGH_DIELECTRIC;
    if (type == "ideal_dielectric")
        return BXDF_IDEAL_DIELECTRIC;
    if (type == "emissive")
        return BXDF_EMISSIVE;
    
    return BXDF_DIFFUSE;
}

void Scene::loadObjWithMaterials(const std::string filePath, ProgressView *progress)
{
    std::vector<tinyobj::shape_t> shapesVec;
    std::vector<tinyobj::material_t> materialsVec;
    tinyobj::attrib_t attrib;
    std::string err;
    
    size_t fileNameStart = filePath.find_last_of("\\"); // assume Windows
    if (fileNameStart == std::string::npos) fileNameStart = filePath.find_last_of("/"); // Linux/MacOS
    std::string folderPath = filePath.substr(0, fileNameStart + 1);
    std::string meshName = filePath.substr(fileNameStart + 1);

    progress->showMessage("Loading mesh", meshName);
    bool ret = tinyobj::LoadObj(&attrib, &shapesVec, &materialsVec, &err, filePath.c_str(), folderPath.c_str());

    if (!err.empty()) // `err` may contain warning message.
    {
        std::cerr << err << std::endl;
    }

    if (!ret)
    {
        std::cout << "OBJ loading failed (tinyobjloader)" << std::endl;
        waitExit();
    }

    const bool hasNormals = attrib.normals.size() > 0;
    const bool hasTexCoords = attrib.texcoords.size() > 0;


    size_t numTris = 0;
    for (tinyobj::shape_t &s : shapesVec)
    {
        numTris += s.mesh.indices.size() / 3;
    }

    // Loop over shapesVec in file
    for (size_t i = 0; i < shapesVec.size(); i++)
    {
        tinyobj::shape_t &shape = shapesVec[i];
        assert((shapesVec[i].mesh.indices.size() % 3) == 0); // properly triangulated

        // Loop over faces in the shape's mesh
        for (size_t f = 0; f < shape.mesh.indices.size() / 3; f++)
        {
            // Progress bar
            size_t N = triangles.size();
            float done = (float)N / numTris;
            if (N % 5000 == 0)
                progress->showMessage("Converting mesh", meshName, done);
            
            VertexPNT V[3];
            
            // Vertices
            bool allNormals = true;
            for (size_t v = 0; v < 3; v++)
            {
                auto ind = shape.mesh.indices[3 * f + v];
                
                // Position
                V[v].p = float3(attrib.vertices[3 * ind.vertex_index + 0],
                                attrib.vertices[3 * ind.vertex_index + 1],
                                attrib.vertices[3 * ind.vertex_index + 2]);

                // Normal
                if (ind.normal_index < 0 || !hasNormals)
                {
                    allNormals = false;
                    V[v].n = float3(0.0f);
                }
                else
                {
                    V[v].n = float3(attrib.normals[3 * ind.normal_index + 0],
                                    attrib.normals[3 * ind.normal_index + 1],
                                    attrib.normals[3 * ind.normal_index + 2]);
                }

                // Tex coord
                if (ind.texcoord_index > -1 && hasTexCoords)
                    V[v].t = float3(attrib.texcoords[2 * ind.texcoord_index + 0], attrib.texcoords[2 * ind.texcoord_index + 1], 0.0f);
                else
                    V[v].t = float3(0.0f);
            }

            if(!allNormals)
                V[0].n = V[1].n = V[2].n = normalize(cross(V[1].p - V[0].p, V[2].p - V[0].p));

            RTTriangle tri(V[0], V[1], V[2]);
            tri.matId = shape.mesh.material_ids[f] + 1; // -1 becomes 0 (default material)
            triangles.push_back(tri);
        }
    }

    // Read materialsVec into own format
    for (tinyobj::material_t &t_mat : materialsVec)
    {
        Material m;
        m.Kd = float3(t_mat.diffuse[0], t_mat.diffuse[1], t_mat.diffuse[2]);
        m.Ks = float3(t_mat.specular[0], t_mat.specular[1], t_mat.specular[2]);
        m.Ke = float3(t_mat.emission[0], t_mat.emission[1], t_mat.emission[2]);
        m.Ns = t_mat.shininess;
        m.Ni = t_mat.ior;
        m.map_Kd = tryImportTexture(unixifyPath(folderPath + t_mat.diffuse_texname), unixifyPath(t_mat.diffuse_texname));
        m.map_Ks = tryImportTexture(unixifyPath(folderPath + t_mat.specular_texname), unixifyPath(t_mat.specular_texname));
        m.map_N = tryImportTexture(unixifyPath(folderPath + t_mat.bump_texname), unixifyPath(t_mat.bump_texname)); // map_bump in mtl treated as normal map
        m.type = parseShaderType(t_mat.unknown_parameter["shader"]);

        materials.push_back(m);
        materialTypes |= m.type;
    }
}

// Import texture if it exists and hasn't been loaded yet, set index in material
cl_int Scene::tryImportTexture(const std::string path, std::string name)
{
    if (name.length() == 0) return -1;

    auto prev = std::find_if(textures.begin(), textures.end(), [name](Texture *t) { return t->getName() == name; });
    if (prev != textures.end())
    {
        return (cl_int)(prev - textures.begin());
    }

    // Texture doesn't exist, load it 
    Texture *tex = new Texture(path, name);
    if (tex->getName() == "error") return -1;

    textures.push_back(tex);
    return (cl_int)(textures.size() - 1);
}

void Scene::loadObjModel(const std::string filename)
{
    std::vector<float3> positions, normals;
    std::vector<std::array<unsigned, 6>> faces;
    
    int face_format = -1;
    std::string format_string = "";
    bool negative_indices = false;

    // Open input file stream for reading.
    std::ifstream input(filename, std::ios::in);

    if(!input)
    {
        std::cout << "Could not open file: " << filename << ", exiting..." << std::endl;
        waitExit();
    }

    // Read the file line by line.
    std::string line;
    while(getline(input, line))
    {
        // Temporary objects to read data into
        std::array<unsigned, 6>  f; // Face index array
        std::string              s;

        std::istringstream iss(line);

        // Read object type into s
        iss >> s;

        // MSVCCompiler has a float cast performance bug
        //   => patch: read into string, cast with atof
        std::string s1, s2, s3;
        if (s == "v")
        {
            iss >> s1 >> s2 >> s3;
            float x = (float)atof(s1.c_str());
            float y = (float)atof(s2.c_str());
            float z = (float)atof(s3.c_str());
            positions.push_back(float3(x, y, z));
        }
        else if (s == "vn")
        {
            iss >> s1 >> s2 >> s3;
            float nx = (float)atof(s1.c_str());
            float ny = (float)atof(s2.c_str());
            float nz = (float)atof(s3.c_str());
            normals.push_back(float3(nx, ny, nz));
        }
        else if (s == "f")
        {
            setFaceFormat(face_format, format_string, negative_indices, iss);

            unsigned sink; // Texture indices ignored for now

            switch (face_format) {
            case 0:
                sscanf(line.c_str(), format_string.c_str(), &f[0], &sink, &f[1], &f[2], &sink, &f[3], &f[4], &sink, &f[5]);
                break;
            case 1:
                sscanf(line.c_str(), format_string.c_str(), &f[0], &f[1], &f[2], &f[3], &f[4], &f[5]);
                break;
            case 2:
                sscanf(line.c_str(), format_string.c_str(), &f[0], &sink, &f[2], &sink, &f[4], &sink);
                break;
            case 3:
                sscanf(line.c_str(), format_string.c_str(), &f[0], &f[2], &f[4]);
                break;
            }

            if (negative_indices)
            {
                // Calculate indices from offsets
                int p = positions.size();
                int n = normals.size();
                f[0] = p - f[0];
                f[1] = n - f[1];
                f[2] = p - f[2];
                f[3] = n - f[3];
                f[4] = p - f[4];
                f[5] = n - f[5];
            }
            else
            {
                // Obj indices start from 1, need to be shifted
                for (unsigned& v : f)
                {
                    v -= 1;
                }
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
        std::string name;                // e.g. vertex
        int lines;                       // e.g. 1300
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

                // MSVCCompiler has a float cast performance bug
                //   => patch: read into string, cast with atof
                std::map<std::string, float> map;
                std::string bucket;
                for (std::string name : e.props)
                {
                    iss >> bucket;
                    map[name] = (float)atof(bucket.c_str());
                }

                positions.push_back(float3(map["x"], map["y"], map["z"]));
                if (map.find("nx") != map.end()) // contains normals
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
                else
                {
                    std::cout << "Unknown polygon type!" << std::endl;
                    waitExit();
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

void Scene::loadPBRTModel(const std::string filename)
{
    try
    {
        auto res = pbrt::importPBRT(filename);
        std::cout << res->toString() << std::endl;
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
}

void Scene::convertPBRTModel(const std::string filenameIn, const std::string filenameOut)
{
    auto res = pbrt::importPBRT(filenameIn);
    res->saveTo(filenameOut);
}

void Scene::loadPBFModel(const std::string filename)
{
    pbrt::Scene::SP scene;
    try
    {
        scene = pbrt::Scene::loadFrom(filename);
        std::cout << scene->toString();
        scene->makeSingleLevel();
    }
    catch (std::runtime_error e)
    {
        std::cerr << "**** ERROR IN PBF PARSING ****" << std::endl << e.what() << std::endl;
        std::cerr << "(this means that either there's something wrong with that PBRT file, or that the parser can't handle it)" << std::endl;
        exit(1);
    }

    size_t fileNameStart = filename.find_last_of("\\"); // assume Windows
    if (fileNameStart == std::string::npos) fileNameStart = filename.find_last_of("/"); // Linux/MacOS
    std::string folderPath = filename.substr(0, fileNameStart + 1);

    auto toFloat3 = [](pbrt::vec3f v) { return float3(v.x, v.y, v.z); };

    //std::set<pbrt::Object::SP> geometries;
    std::vector<pbrt::Material::SP> pbrtMaterials;

    std::function<void(pbrt::Object::SP, pbrt::affine3f xform)> traverse;
    traverse = [&](pbrt::Object::SP object, pbrt::affine3f xform)
    {
        //geometries.insert(object);
        for (auto shape : object->shapes)
        {
            int matId = 0;
            if (shape->material)
            {
                auto prev = std::find(pbrtMaterials.begin(), pbrtMaterials.end(), shape->material);
                if (prev != pbrtMaterials.end())
                {
                    matId = (prev - pbrtMaterials.begin()) + 1;
                }
                else
                {
                    pbrtMaterials.push_back(shape->material);
                    matId = pbrtMaterials.size();
                }
            }

            if (shape->areaLight)
                std::cout << "Skipping area light " << shape->areaLight->toString() << std::endl;
            
            if (pbrt::TriangleMesh::SP mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape))
            {
                const bool hasNormals = mesh->normal.size() > 0;
                const bool hasTexcoord = mesh->texcoord.size() > 0;
                
                // Each triangle
                for (int i = 0; i < mesh->index.size(); i++)
                {
                    VertexPNT V[3];
                    int *inds = &(mesh->index[i].x); // hopefully contiguous...

                    // Each vertex
                    for (int v = 0; v < 3; v++)
                    {
                        int index = inds[v];

                        if (index < 0)
                        {
                            std::cout << "Negative index" << std::endl;
                            index += mesh->index.size();
                        }

                        if (index >= mesh->vertex.size())
                            std::cout << "Mesh index out of range..." << std::endl;

                        if (hasNormals && index >= mesh->normal.size())
                            std::cout << "Normal index out of range..." << std::endl;

                        if (hasTexcoord && index >= mesh->texcoord.size())
                            std::cout << "TexCoord index out of range..." << std::endl;

                        auto P = mesh->vertex[index];
                        auto N = (hasNormals) ? mesh->normal[index] : pbrt::vec3f(0.0f);
                        auto T = (hasTexcoord) ? mesh->texcoord[index] : pbrt::vec2f(0.0f);

                        // Apply transformation
                        P = xform * P;
                        N = pbrt::math::inverse_transpose(xform.l) * N;

                        V[v].p = toFloat3(P);
                        V[v].n = toFloat3(N);
                        V[v].t = float3(T.x, T.y, 0.0f);
                    }

                    if (!hasNormals)
                        V[0].n = V[1].n = V[2].n = normalize(cross(V[1].p - V[0].p, V[2].p - V[0].p));

                    // Apply transform

                    RTTriangle tri(V[0], V[1], V[2]);
                    tri.matId = matId;
                    triangles.push_back(tri);
                }
                
            }
            else if (pbrt::QuadMesh::SP mesh = std::dynamic_pointer_cast<pbrt::QuadMesh>(shape))
            {
                std::cout << "Quads: " << mesh->index.size() << std::endl;
            }
            else if (pbrt::Sphere::SP sphere = std::dynamic_pointer_cast<pbrt::Sphere>(shape))
            {
                std::cout << "Sphere!" << std::endl;
            }
            else if (pbrt::Disk::SP disk = std::dynamic_pointer_cast<pbrt::Disk>(shape))
            {
                std::cout << "Disk!" << std::endl;
            }
            else if (pbrt::Curve::SP curves = std::dynamic_pointer_cast<pbrt::Curve>(shape))
            {
                std::cout << "Curve!" << std::endl;
            }
            else
                std::cout << "unhandled geometry type : " << shape->toString() << std::endl;
        }

        for (auto inst : object->instances)
            traverse(inst->object, xform*inst->xfm);
    };

    traverse(scene->world, pbrt::affine3f::identity());

    // Read xform active when camera was created
    pbrt::Camera::SP cam = scene->cameras[0];
    float3 v = toFloat3(cam->frame.l.vy);
    this->worldUp = (fabs(v.y) > fabs(v.z)) ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, 1.0f);

    auto loadTex = [&](pbrt::Texture::SP tmap)
    {
        if (!tmap)
            return (cl_int)-1;

        if (pbrt::ImageTexture * tex = dynamic_cast<pbrt::ImageTexture*>(tmap.get()))
            return tryImportTexture(unixifyPath(folderPath + tex->fileName), unixifyPath(tex->fileName));
        
        std::cout << "Unsupported texture type" << std::endl;
        return (cl_int)-1;
    };

    // No support for anisitropy at the moment
    auto convertRoughness = [](float r, bool remap = true, float ru = 0.0f, float rv = 0.0f)
    {
        float res = (r > 0.0f) ? r : (0.5f * (ru + rv));
        return (1.0f - res) * ((remap) ? 5000.0f : 1.0f);
    };

    // Read materialsVec into own format
    for (pbrt::Material::SP t_mat : pbrtMaterials)
    {   
        // Use default parameters
        Material m(materials[0]);

        if (auto mat = dynamic_cast<pbrt::PlasticMaterial*>(t_mat.get()))
        {
            // Plastic approximated with substrate for now
            m.type = BXDF_GLOSSY;
            m.Kd = toFloat3(mat->kd);
            m.Ks = toFloat3(mat->ks);
            m.Ns = convertRoughness(mat->roughness, mat->remapRoughness);
            m.map_Kd = loadTex(mat->map_kd);
            m.map_Ks = loadTex(mat->map_ks);
            m.Ni = 1.5; // for Fresnel
            //m.map_N = loadTex();
            //m.map_bump = loadTex(mat->map_bump);

        }
        else if (auto mat = dynamic_cast<pbrt::MatteMaterial*>(t_mat.get()))
        {
            m.type = BXDF_DIFFUSE;
            m.Kd = toFloat3(mat->kd);
            m.map_Kd = loadTex(mat->map_kd);
            //m.map_N = loadTex();
            //m.map_bump = loadTex(mat->map_bump);
        }
        else if (auto mat = dynamic_cast<pbrt::SubstrateMaterial*>(t_mat.get()))
        {
            // Substrate material: diffuse base layer, glossy varnish on top, Fresnel blending
            m.type = BXDF_GLOSSY;
            m.Kd = toFloat3(mat->kd);
            m.Ks = toFloat3(mat->ks);
            m.Ns = convertRoughness(0.0f, mat->remapRoughness, mat->uRoughness, mat->vRoughness);
            m.map_Kd = loadTex(mat->map_kd);
            m.map_Ks = loadTex(mat->map_ks);
            m.Ni = 1.5; // for Fresnel
        }
        else if (auto mat = dynamic_cast<pbrt::UberMaterial*>(t_mat.get()))
        {
            // Substrate material: diffuse base layer, glossy varnish on top, Fresnel blending
            m.type = BXDF_GLOSSY;
            m.Kd = toFloat3(mat->kd);
            m.Ks = toFloat3(mat->ks);
            m.Ns = convertRoughness(mat->roughness, true, mat->uRoughness, mat->vRoughness);
            m.map_Kd = loadTex(mat->map_kd);
            m.map_Ks = loadTex(mat->map_ks);
            m.Ni = mat->index;
        }
        else if (auto mat = dynamic_cast<pbrt::GlassMaterial*>(t_mat.get()))
        {
            m.type = BXDF_IDEAL_DIELECTRIC;
            m.Ks = toFloat3(mat->kt); // m.Ks treated as transmissivity
            m.Ni = (mat->index > 0.0f) ? mat->index : 1.5f;
        }
        else if (auto mat = dynamic_cast<pbrt::MirrorMaterial*>(t_mat.get()))
        {
            m.type = BXDF_IDEAL_REFLECTION;
            m.Ks = toFloat3(mat->kr); // reflectivity
        }
        else if (auto mat = dynamic_cast<pbrt::MetalMaterial*>(t_mat.get()))
        {
            // Very rough approximation of true behavior
            m.type = BXDF_GGX_ROUGH_REFLECTION;
            m.Ni = (mat->eta.x + mat->eta.y + mat->eta.z) / 3.0f;
            m.Ks = toFloat3(mat->k);
            m.Ns = convertRoughness(mat->roughness, mat->remapRoughness, mat->uRoughness, mat->vRoughness);
        }
        else if (auto mat = dynamic_cast<pbrt::FourierMaterial*>(t_mat.get()))
        {
            std::cout << "Unsupported material: FourierMaterial" << std::endl;
        }
        else if (auto mat = dynamic_cast<pbrt::HairMaterial*>(t_mat.get()))
        {
            std::cout << "Unsupported material: HairMaterial" << std::endl;
        }
        else
        {
            std::cout << "Unhandled material type" << std::endl;
        }

        materials.push_back(m);
        materialTypes |= m.type;
    }
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
