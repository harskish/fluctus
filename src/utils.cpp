#include "utils.h"
#include "xxhash/xxhash.h"
#include "tinyfiledialogs.h"
#include <fstream>
#include <iostream>
#include <vector>

std::string getAbsolutePath(std::string filename)
{
    const int MAX_LENTH = 4096;
    char resolved_path[MAX_LENTH];
#ifdef _WIN32
    _fullpath(resolved_path, filename.c_str(), MAX_LENTH);
#else
    realpath(filename.c_str(), resolved_path);
#endif
    return std::string(resolved_path);
}

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

std::string getFileName(const std::string path)
{
    const std::string upath = unixifyPath(path);
    size_t idx = path.find("/", 0) + 1;
    return upath.substr(idx, upath.length() - idx);
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

std::string getCLErrorString(int code)
{
    const int SIZE = 64;
    std::string errors[SIZE] =
    {
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE", "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE", "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH", "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE",
        "CL_MISALIGNED_SUB_BUFFER_OFFSET", "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
        "CL_COMPILE_PROGRAM_FAILURE", "CL_LINKER_NOT_AVAILABLE", "CL_LINK_PROGRAM_FAILURE",
        "CL_DEVICE_PARTITION_FAILED", "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
        "-20", "-21", "-22", "-23", "-24", "-25", "-26", "-27", "-28", "-29",
        "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE", "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE", "CL_INVALID_CONTEXT", "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE", "CL_INVALID_HOST_PTR", "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER", "CL_INVALID_BINARY", "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME", "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX", "CL_INVALID_ARG_VALUE", "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE", "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET", "CL_INVALID_EVENT_WAIT_LIST", "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT", "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL", "CL_INVALID_GLOBAL_WORK_SIZE"
    };

    const int ind = -code;
    return (ind >= 0 && ind < SIZE) ? errors[ind] : "unknown";
}