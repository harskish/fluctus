#pragma once

#include <string>
#include <stdlib.h>
#include <glad/glad.h>
#include <vector>
#include "cl2.hpp"
#include "bxdf_types.h"

// Determine target
#if _WIN32 || _WIN64
#if _WIN64
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif

#if __GNUC__
#if __x86_64__ || __ppc64__
#define ENVIRONMENT64
#else
#define ENVIRONMENT32
#endif
#endif

inline void waitExit()
{
#ifdef WIN32
    system("pause");
#endif
    exit(EXIT_FAILURE);
}

inline void GLcheckErrors()
{
    GLenum err = glGetError();
    const char* name;
    switch (err)
    {
    case GL_NO_ERROR:                       name = NULL; break;
    case GL_INVALID_ENUM:                   name = "GL_INVALID_ENUM"; break;
    case GL_INVALID_VALUE:                  name = "GL_INVALID_VALUE"; break;
    case GL_INVALID_OPERATION:              name = "GL_INVALID_OPERATION"; break;
    case GL_OUT_OF_MEMORY:                  name = "GL_OUT_OF_MEMORY"; break;
    case GL_INVALID_FRAMEBUFFER_OPERATION:  name = "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
    default:                                name = "unknown"; break;
    }

    if (name)
    {
        printf("Caught GL error 0x%04x (%s)!", err, name);
        waitExit();
    }
}

inline bool platformIsNvidia(cl::Platform& platform)
{
    std::string name = platform.getInfo<CL_PLATFORM_NAME>();
    return name.find("NVIDIA") != std::string::npos;
}

std::string getAbsolutePath(std::string filename);
std::string getFileName(const std::string path);

bool endsWith(const std::string s, const std::string end);
std::string unixifyPath(std::string path);

std::string openFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter);
std::string saveFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter);

size_t computeHash(const void* buffer, size_t length);
size_t fileHash(const std::string filename);

// Get define string used to compile only relevant material eval logic
std::string getBxdfDefines(unsigned int typeBits);
