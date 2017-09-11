#pragma once

/*
	A class that reads an RGBe file and creates an OpenCL image2d_t environment map.
*/

#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <stack>
#include "rgbe/rgbe.hpp"
#include "geom.h"

class EnvironmentMap
{
public:
	EnvironmentMap() : width(0), height(0), scale(1.0f), data(NULL), probTable(NULL), aliasTable(NULL) {} // default constructor
	EnvironmentMap(const char *filename);
	~EnvironmentMap()
	{
		(void)scale;
		delete[] data; 
		delete[] probTable;
		delete[] aliasTable;
        delete[] cdfTable;
        delete[] pdfTable;
	}

	float *getData() { return data; }
	float *getProbTable() { return probTable; }
	int *getAliasTable() { return aliasTable; }
    float *getCdfTable() { return cdfTable; }
    float *getPdfTable() { return pdfTable; }
	int getWidth() { return width; }
	int getHeight() { return height; }

	bool valid() { return data != NULL && probTable != NULL && aliasTable != NULL && width * height > 0; }

private:
	void computeProbabilities();
	
	int width, height;
	float scale;
	float *data; // used by clcontext to create cl::Image2D
	
	// For importance sampling (alias method)
	// One per row (conditional pdfs) + one for marginal pdf
	//    => (h * w + h) entries per table, last column for marginal
	float *probTable;
	int *aliasTable;

    float *cdfTable;
    float *pdfTable;
};