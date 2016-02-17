// Edge.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR * argv[]) {
	using namespace std;
	string inputFile;
	string outputFile;
	unsigned char *h_originalImage, *d_originalImage;
	unsigned char *h_edgeImage, *d_edgeImage;

	if (argc == 3) {
		inputFile = string((char *)argv[1]);
		outputFile = string((char *)argv[2]);
	}
	else {
		cerr << "Usage: Edge inputFile outputFile" << endl;
		exit(1);
	}

	preProcess(&h_originalImage, &h_edgeImage, &d_originalImage, &d_edgeImage, inputFile);

	return 0;
}

void preProcess(unsigned char **inputImage, unsigned char **edgeImage,
	unsigned char **d_originalImage, unsigned char **d_edgeImage,
	const std::string & filename) {
	cudaFree(0);
}
