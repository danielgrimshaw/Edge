// Edge.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

size_t numPixels;
int numRows, numCols;

__global__ void edgeDetect(const unsigned char * const greyImage,
	unsigned char * const edgeImage,
	int numRows, int numCols) {
	//TODO
}

__global__ void greyscale(const uchar4 * const origImage,
	unsigned char * const greyImage,
	int numRows, int numCols) {
	int id = blockIdx.x*blockDim.x;
	id += threadIdx.x;
	uchar4 rgba = origImage[id];
	float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
	greyImage[id] = (char)channelSum;
	__syncthreads();
}

int _tmain(int argc, _TCHAR * argv[]) {
	using namespace std;
	string inputFile;
	string outputFile;
	uchar4 *h_originalImage, *d_originalImage;
	unsigned char *h_greyImage, *d_greyImage;
	unsigned char *h_edgeImage, *d_edgeImage;

	if (argc == 3) {
		inputFile = string((char *)argv[1]);
		outputFile = string((char *)argv[2]);
	}
	else {
		cerr << "Usage: Edge inputFile outputFile" << endl;
		exit(1);
	}

	preProcess(&h_originalImage, &h_greyImage, &h_edgeImage, &d_originalImage, &d_greyImage, &d_edgeImage, inputFile);
	
	const int threadsInBlock = numPixels < 256 ? numPixels : 256;
	const dim3 threadsPerBlock((int)sqrt(threadsInBlock), (int)sqrt(threadsInBlock));
	const dim3 blocksPerGrid((numPixels + threadsInBlock - 1) / threadsInBlock);

	greyscale<<<(numPixels +threadsInBlock - 1) / threadsInBlock, threadsInBlock>>>(d_originalImage, d_greyImage, numRows, numCols);
	cudaDeviceSynchronize();

	cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

	edgeDetect<<<blocksPerGrid, threadsPerBlock>>>(d_greyImage, d_edgeImage, numRows, numCols);
	cudaDeviceSynchronize();

	cudaMemcpy(h_edgeImage, d_edgeImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

	postProcess(outputFile, h_edgeImage);

	cudaFree(d_originalImage);
	cudaFree(d_greyImage);
	cudaFree(d_edgeImage);

	return 0;
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage, unsigned char **edgeImage,
	uchar4 **d_originalImage, unsigned char **d_greyImage, unsigned char **d_edgeImage,
	const std::string & filename) {
	//Pre-process the image
	cudaFree(0);

	cv::Mat imageOrig;
	cv::Mat imageGrey;
	cv::Mat imageEdge;
	
	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}
	
	cv::cvtColor(image, imageOrig, CV_BGR2RGBA);
	
	// allocate memory for the output
	imageGrey.create(image.rows, image.cols, CV_8UC1);
	imageEdge.create(image.rows, image.cols, CV_8UC1);

	if (!imageOrig.isContinuous() || !imageGrey.isContinuous || !imageEdge.isContinuous()) {
		std::cerr << "Images aren't continuous!!! Aborting." << std::endl;
		exit(1);
	}
	
	*inputImage = (uchar4 *)imageOrig.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);
	*edgeImage = imageEdge.ptr<unsigned char>(0);

	numRows = imageOrig.rows;
	numCols = imageOrig.cols;
	numPixels = imageOrig.rows * imageOrig.cols;

	// allocate device memory
	cudaMalloc(d_originalImage, sizeof(uchar4) * numPixels);
	cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels);
	cudaMalloc(d_edgeImage, sizeof(unsigned char) * numPixels);

	cudaMemset(*d_greyImage, 0, sizeof(unsigned char) * numPixels);
	cudaMemset(*d_edgeImage, 0, sizeof(unsigned char) * numPixels);

	// copy inputs to GPU
	cudaMemcpy(*d_originalImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
}
	
void postProcess(const std::string & output_file, unsigned char *edgeImage) {
	cv::Mat output(numRows, numCols, CV_8UC1, (void *)edgeImage);

	cv::imwrite(output_file.c_str(), output);
}