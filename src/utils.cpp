#include "utils.h"
#include "xxhash/xxhash.h"
#include "tinyfiledialogs.h"
#include <fstream>
#include <iostream>
#include <vector>

bool endsWith(const std::string s, const std::string end)
{
    size_t len = end.size();
    if (len > s.size()) return false;

    std::string substr = s.substr(s.size() - len, len);
    return end == substr;
}

std::string unixifyPath(std::string path)
{
    size_t index = 0;
    while (true)
    {
        index = path.find("\\", index);
        if (index == std::string::npos) break;

        path.replace(index, 1, "/");
        index += 1;
    }

    return path;
}

std::string openFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter)
{
    char const *selected = tinyfd_openFileDialog(message.c_str(), defaultPath.c_str(), filter.size(), filter.data(), NULL, 0);
    return (selected) ? std::string(selected) : "";
}

std::string saveFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter)
{
    char const *selected = tinyfd_saveFileDialog(message.c_str(), defaultPath.c_str(), filter.size(), filter.data(), NULL);
    return (selected) ? std::string(selected) : "";
}

size_t computeHash(const void* buffer, size_t length)
{
    const size_t seed = 0;
#ifdef ENVIRONMENT64
    size_t const hash = XXH64(buffer, length, seed);
#else
    size_t const hash = XXH32(buffer, length, seed);
#endif
    return hash;
}


size_t fileHash(const std::string filename)
{
    std::ifstream f(filename, std::ios::binary | std::ios::ate);

    if (!f)
    {
        std::cout << "Could not open file " << filename << " for hashing, exiting..." << std::endl;
        waitExit();
    }

    std::ifstream::pos_type pos = f.tellg();
    std::vector<char> data(pos);
    f.seekg(0, std::ios::beg);
    f.read(&data[0], pos);
    f.close();

    return computeHash((const void*)data.data(), (size_t)pos);
}