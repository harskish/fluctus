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

	computeProbabilities();
	std::cout << "Read environment map of size [" << width << ", " << height << "]" << std::endl;
}

// PBRT chapters 14.2, 13.3
void EnvironmentMap::computeProbabilities()
{
	// Create scalar representaiton of map (using luminance)
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

	std::vector<std::vector<float>> conditonalPdfs(height);
	std::vector<float> marginalPdf(height);
	std::vector<float> rowIntegrals(height);

	std::cout << "Computing pdfs" << std::endl;

    cdfTable = new float[(width + 2) * (height + 1)];
    pdfTable = new float[(width + 2) * (height + 1)];

	// Compute 1D conditional distributions
	#pragma omp parallel for
	for (int v = 0; v < height; v++)
	{
		//std::vector<float> cdf(width + 1);
        float *cdf = cdfTable + v * (width + 2);
        float *pdf = pdfTable + v * (width + 2);
		const float* row = scalars.get() + v * width;

		// Fill cdf with unnormalized integrals
		cdf[0] = 0;
		for (int i = 1; i < width + 1; i++) cdf[i] = cdf[i - 1] + row[i - 1] / width;

		// Get step-function integral over row
		float I = cdf[width];
		rowIntegrals[v] = I;

		// Normalize to get actual cdf
		if (I == 0)
			for (int i = 1; i < width + 1; ++i) cdf[i] = float(i) / float(width); // make limit one
		else
			for (int i = 1; i < width + 1; ++i) cdf[i] /= I;

		// Calculate pdf
		if (I == 0)
			for (int i = 0; i < width; i++) pdf[i] = 1.0f / float(width); // make integral one
		else
			for (int i = 0; i < width; i++) pdf[i] = row[i] / I;

		// Save conditional pdf
        std::vector<float> pdfV(pdf, pdf + width);
		conditonalPdfs[v] = pdfV;
	}

	// Compute marginal sampling distribution
	{
		std::vector<float> cdf(height + 1);

		// Fill cdf with unnormalized integrals
		cdf[0] = 0;
		for (int i = 1; i < height + 1; ++i) cdf[i] = cdf[i - 1] + rowIntegrals[i - 1] / height;

		// Get step-function integral over row
		float I = cdf[height];

		// Normalize to get actual cdf
		if (I == 0)
			for (int i = 1; i < height + 1; ++i) cdf[i] = float(i) / float(height);
		else
			for (int i = 1; i < height + 1; ++i) cdf[i] /= I;

        // Write cdf into table (last column)
        for (int i = 0; i < height + 1; i++)
            cdfTable[i * (width + 2) + width + 1] = cdf[i];

		// Calculate pdf
		if (I == 0)
			for (int i = 0; i < height; i++) marginalPdf[i] = 1.0f / float(height);
		else
			for (int i = 0; i < height; i++) marginalPdf[i] = rowIntegrals[i] / I;

        // Write pdf into table (last column)
        for (int i = 0; i < height; i++)
            pdfTable[i * (width + 2) + width + 1] = marginalPdf[i];
	}


	std::cout << "Creating probability and alias tables" << std::endl;

	// Compute probability and alias tables
	// Stable Vose's algorithm, see http://www.keithschwarz.com/darts-dice-coins/
	
	probTable = new float[(width + 1) * height];
	aliasTable = new int[(width + 1) * height];

	// Conditional pdfs
	for (int v = 0; v < height; v++)
	{
		// Last column contains data of marginal distribution
		float* pTable = probTable + (v * (width + 1));
		int* aTable = aliasTable + (v * (width + 1));

		std::stack<std::pair<float, int>> small, large;

		// Distribute probabilities
		std::vector<float>& pdfs = conditonalPdfs[v];
		for (int i = 0; i < pdfs.size(); i++)
		{
			float p = pdfs[i]; // n pre-divided (stepfunction pdf)
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

			pTable[l.second] = l.first;
			aTable[l.second] = g.second;
			
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
			pTable[g.second] = 1.0f;
		}

		while (!small.empty())
		{
			std::pair<float, int> l = small.top();
			small.pop();
			pTable[l.second] = 1.0f;
		}
	}

	// Marginal pdf
	{
		std::stack<std::pair<float, int>> small, large;

		// Distribute probabilities
		std::vector<float>& pdfs = marginalPdf;
		for (int i = 0; i < pdfs.size(); i++)
		{
			float p = pdfs[i]; // n pre-divided (stepfunction pdf)
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

			probTable[l.second * (width + 1) + width] = l.first;
			aliasTable[l.second * (width + 1) + width] = g.second;

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
			probTable[g.second * (width + 1) + width] = 1.0f;
		}

		while (!small.empty())
		{
			std::pair<float, int> l = small.top();
			small.pop();
			probTable[l.second * (width + 1) + width] = 1.0f;
		}
	}
}
