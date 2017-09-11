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
	EnvironmentMap() :
		width(0),
		height(0),
		scale(1.0f),
		data(NULL),
		probTable(NULL),
		aliasTable(NULL),
		cdfTable(NULL),
		pdfTable(NULL),
		pdfTable1D(NULL)
	{} // default constructor

	EnvironmentMap(const char *filename);
	~EnvironmentMap()
	{
		(void)scale;
		delete[] data; 
		delete[] probTable;
		delete[] aliasTable;
        delete[] cdfTable;
        delete[] pdfTable;
		delete[] pdfTable1D;
	}

	float *getData() { return data; }
	float *getProbTable() { return probTable; }
	int *getAliasTable() { return aliasTable; }
    float *getCdfTable() { return cdfTable; }
    float *getPdfTable() { return pdfTable; }
	float *getPdfTable1D() { return pdfTable1D; }
	int getWidth() { return width; }
	int getHeight() { return height; }

	bool valid()
	{ 
		return data != NULL && probTable != NULL && aliasTable != NULL
			&& pdfTable != NULL && pdfTable1D != NULL &&  width * height > 0;
	}

private:
	void computeProbabilities();
	
	int width, height;
	float scale;
	float *data; // used by clcontext to create cl::Image2D
	
	// For importance sampling (alias method)
	float *pdfTable1D;
	float *probTable;
	int *aliasTable;

	// For importance sampling (pbrt binary search method)
	// One per row (conditional pdfs) + one for marginal pdf
	//    => (h * w + h) entries per table, last column for marginal
    float *cdfTable;
    float *pdfTable;
};