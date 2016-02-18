// Edge.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

size_t numPixels;
int numRows, numCols;

__global__ void edgeDetect(const uchar4 * const origImage,
	uchar4 * const edgeImage,
	int numRows, int numCols) {
	//TODO
}

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
	
	const int threadsInBlock = numPixels < 256 ? numPixels : 256;
	const dim3 threadsPerBlock((int)sqrt(threadsInBlock), (int)sqrt(threadsInBlock));
	const dim3 blocksPerGrid((numPixels + threadsInBlock - 1) / threadsInBlock);

	edgeDetect<<<blocksPerGrid, threadsPerBlock>>>(d_originalImage, d_edgeImage, numRows, numCols);

	cudaDeviceSynchronize();

	cudaMemcpy(h_edgeImage, d_edgeImage, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

	postProcess(outputFile, h_edgeImage);

	cudaFree(d_originalImage);
	cudaFree(d_edgeImage);

	return 0;
}

void preProcess(uchar4 **inputImage, uchar4 **edgeImage,
	uchar4 **d_originalImage, uchar4 **d_edgeImage,
	const std::string & filename) {
	//Pre-process the image
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

	numRows = imageOrig.rows;
	numCols = imageOrig.cols;
	numPixels = imageOrig.rows * imageOrig.cols;

	// allocate device memory
	cudaMalloc(d_originalImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_edgeImage, sizeof(uchar4) * numPixels);
	cudaMemset(*d_edgeImage, 0, sizeof(uchar4) * numPixels);

	// copy inputs to GPU
	cudaMemcpy(*d_originalImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
}
	
void postProcess(const std::string & output_file,uchar4 *edgeImage) {
	cv::Mat output(numRows, numCols, CV_8UC4, (void *)edgeImage);

	cv::imwrite(output_file.c_str(), output);
}