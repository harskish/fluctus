#pragma once

/*
	A class that reads an RGBe file and creates an OpenCL image2d_t environment map.
*/

#include <iostream>
#include "rgbe/rgbe.hpp"

class EnvironmentMap
{
public:
	EnvironmentMap() : width(0), height(0), scale(1.0f), data(NULL) {} // default constructor
	EnvironmentMap(const char *filename);
	~EnvironmentMap()
	{
		(void)scale;
		delete [] data; 
	}

	float *getData() { return data; }
	int getWidth() { return width; }
	int getHeight() { return height; }

private:
	int width, height;
	float scale;
	float *data; // used by clcontext to create cl::Image2D
};