#include "envmap.hpp"

EnvironmentMap::EnvironmentMap(const char *filename) : scale(1.0f)
{
	FILE * f = fopen(filename, "rb");

	if(!f)
	{
		std::cout << "Cannot open file '" << filename << "'" << std::endl;
        exit(1);
	}

	RGBE_ReadHeader(f, &width, &height, NULL);
	data = new float [width * height * 3];
	RGBE_ReadPixels_RLE(f, data, width, height);
	fclose(f);

	std::cout << "Read environment map of size [" << width << ", " << height << "]" << std::endl;
}