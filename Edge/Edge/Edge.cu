// Edge.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR * argv[]) {
	using namespace std;
	string inputFile;
	string outputFile;
	uchar4 *h_originalImage, *d_originalImage;
	uchar4 *h_edgeImage, *d_edgeImage;

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

void preProcess(uchar4 **inputImage, uchar4 **edgeImage,
	uchar4 **d_originalImage, uchar4 **d_edgeImage,
	const std::string & filename) {
	cudaFree(0);

	cv::Mat imageOrig;
	cv::Mat imageEdge;
	
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}
	
	cv::cvtColor(image, imageOrig, CV_BGR2RGBA);
	
	// allocate memory for the output
	imageEdge.create(image.rows, image.cols, CV_8UC4);

	if (!imageOrig.isContinuous() || !imageEdge.isContinuous()) {
		std::cerr << "Images aren't continuous!!! Aborting." << std::endl;
		exit(1);
	}
	
	*inputImage = (uchar4 *)imageOrig.ptr<unsigned char>(0);
	*edgeImage = (uchar4 *)imageEdge.ptr<unsigned char>(0);

	const size_t numPixels = imageOrig.rows * imageOrig.cols;

	// allocate device memory
	cudaMalloc(d_originalImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_edgeImage, sizeof(uchar4) * numPixels);
	cudaMemset(*d_edgeImage, 0, sizeof(uchar4) * numPixels);

	// copy inputs to GPU
	cudaMemcpy(*d_originalImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
}
	
