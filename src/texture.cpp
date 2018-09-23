#include "texture.hpp"
#include "IL/il.h"
#include "IL/ilu.h"
#include <iostream>

inline void checkILErrors()
{
    ILenum error;
    while ((error = ilGetError()) != IL_NO_ERROR)
    {
        //std::cout << "0x" << std::hex << error << ": " << std::dec << iluErrorString(error) << std::endl;
        printf("%d: %s\n", error, iluErrorString(error));
    }
}

Texture::Texture(const std::string path, const std::string filename)
{
    ILuint ImageName;
    ilGenImages(1, &ImageName);
    ilBindImage(ImageName);
    checkILErrors();

    ILboolean success = ilLoadImage(path.c_str());

    if (success == IL_TRUE)
    {
        width = (cl_uint)ilGetInteger(IL_IMAGE_WIDTH);
        height = (cl_uint)ilGetInteger(IL_IMAGE_HEIGHT);
        data = new cl_uchar[width * height * 4 * 1]; // RGBA: 4 channels, 1 ubyte (uchar) per channel
        ilCopyPixels(0, 0, 0, width, height, 1, IL_RGBA, IL_UNSIGNED_BYTE, data);
        name = filename;
    }
    else
    {
        std::cout << "Texture loading failed for " << filename << std::endl;
        checkILErrors();
        name = "error";
    }

    ilDeleteImages(1, &ImageName);
}