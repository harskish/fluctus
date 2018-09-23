#pragma once

/*
	A class that reads an RGBe file and creates an OpenCL image2d_t environment map.
*/

#include <string>

class EnvironmentMap
{
public:
	EnvironmentMap() :
		width(0),
		height(0),
		scale(1.0f),
		data(NULL),
        name(""),
		pdfTable(NULL),
		probTable(NULL),
		aliasTable(NULL)
	{} // default constructor

	EnvironmentMap(const std::string &filename);
	~EnvironmentMap()
	{
		(void)scale;
		delete[] data; 
		delete[] probTable;
		delete[] aliasTable;
		delete[] pdfTable;
	}

    std::string getName() { return name; }
	float *getData() { return data; }
	float *getProbTable() { return probTable; }
	int *getAliasTable() { return aliasTable; }
	float *getPdfTable() { return pdfTable; }
	int getWidth() { return width; }
	int getHeight() { return height; }

	bool valid() { return data != NULL && probTable != NULL && aliasTable != NULL && pdfTable != NULL && width * height > 0; }

private:
	void computeProbabilities();
	
	int width, height;
	float scale;
	float *data; // used by clcontext to create cl::Image2D
    std::string name;
	
	// For importance sampling (alias method)
	float *pdfTable;
	float *probTable;
	int *aliasTable;
};