#pragma once
#include <stdlib.h>
#include <glad/glad.h>
#include "tinyfiledialogs.h"

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

inline bool endsWith(const std::string s, const std::string end) {
    size_t len = end.size();
    if (len > s.size()) return false;

    std::string substr = s.substr(s.size() - len, len);
    return end == substr;
}

inline std::string unixifyPath(std::string path) {
    size_t index = 0;
    while (true) {
        index = path.find("\\", index);
        if (index == std::string::npos) break;

        path.replace(index, 1, "/");
        index += 1;
    }

    return path;
}

inline std::string openFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter) {
    char const *selected = tinyfd_openFileDialog(message.c_str(), defaultPath.c_str(), filter.size(), filter.data(), NULL, 0);
    return (selected) ? std::string(selected) : "";
}

inline std::string saveFileDialog(const std::string message, const std::string defaultPath, const std::vector<const char*> filter) {
    char const *selected = tinyfd_saveFileDialog(message.c_str(), defaultPath.c_str(), filter.size(), filter.data(), NULL);
    return (selected) ? std::string(selected) : "";
}