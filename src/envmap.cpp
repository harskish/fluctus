#include "envmap.hpp"
#include "rgbe/rgbe.hpp"
#include "utils.h"
#include "geom.h"
#include <iostream>
#include <memory>
#include <stack>

EnvironmentMap::EnvironmentMap(const std::string &filename) : scale(1.0f)
{
	FILE * f = fopen(filename.c_str(), "rb");

	if(!f)
	{
		std::cout << "Cannot open file '" << filename << "'" << std::endl;
        waitExit();
	}

	RGBE_ReadHeader(f, &width, &height, NULL);
	data = new float [width * height * 3];
	RGBE_ReadPixels_RLE(f, data, width, height);
	fclose(f);
    name = filename;

	computeProbabilities();
	std::cout << "Read environment map of size [" << width << ", " << height << "]" << std::endl;
}

// Prepares environment map for importance sampling (using the alias method)
// PBRT chapters 14.2, 13.3
void EnvironmentMap::computeProbabilities()
{
	std::cout << "Processing environment map" << std::endl;

	/* Create scalar representaiton of map (using luminance) */
	std::unique_ptr<float[]> scalars(new float[width * height]);
	
	#pragma omp parallel for
	for (int v = 0; v < height; v++)
	{
		float sinTh = std::sin(PI * float(v + 0.5f) / float(height));
		for (int u = 0; u < width; u++)
		{
			float r = data[3 * (v * width + u) + 0];
			float g = data[3 * (v * width + u) + 1];
			float b = data[3 * (v * width + u) + 2];
			float lum = 0.212671f * r + 0.715160f * g + 0.072169f * b; // sRGB luminance
			
			// SinTh from jacobian of (u,v)->(x,y,z) baked in for IS
            scalars[v * width + u] = lum * sinTh;
		}
	}

	/* Compute 1D flat pdf over whole image */
	pdfTable = new float[width * height];

	// Get integral over whole image
	float I = 0.0f;
	for (int i = 1; i < width*height + 1; i++) I += scalars[i - 1] / (width*height);

	// Calculate pdf
	if (I == 0)
		for (int i = 0; i < width*height; i++) pdfTable[i] = 1.0f / float(width*height); // make integral one
	else
		for (int i = 0; i < width*height; i++) pdfTable[i] = scalars[i] / I;

	/* Compute probability and alias tables */
	/* Stable Vose's algorithm, see http://www.keithschwarz.com/darts-dice-coins/ */
	probTable = new float[width * height];
	aliasTable = new int[width * height];

	// 1D pdf over whole image
	std::stack<std::pair<float, int>> small, large;

	// Distribute probabilities
	for (int i = 0; i < width * height; i++)
	{
		float p = pdfTable[i]; // n pre-divided (stepfunction pdf)
		if (p < 1.0f)
			small.push(std::make_pair(p, i));
		else
			large.push(std::make_pair(p, i));
	}

	while (!small.empty() && !large.empty())
	{
		std::pair<float, int> l = small.top(), g = large.top();
		small.pop();
		large.pop();

		probTable[l.second] = l.first;
		aliasTable[l.second] = g.second;

		float pg = (g.first + l.first) - 1.0f;
		if (pg < 1.0f)
			small.push(std::make_pair(pg, g.second));
		else
			large.push(std::make_pair(pg, g.second));
	}

	while (!large.empty())
	{
		std::pair<float, int> g = large.top();
		large.pop();
		probTable[g.second] = 1.0f;
	}

	while (!small.empty())
	{
		std::pair<float, int> l = small.top();
		small.pop();
		probTable[l.second] = 1.0f;
	}
}
