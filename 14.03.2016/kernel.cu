#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <cuda.h>
#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <string>
#include <time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <GL/glut.h>
#include <GL/glui.h>
using namespace std;
using namespace cv;

//***********************************************************************************************************************

Mat img;
unsigned char *pixelPtr, *pixelPtr2, *pixelPtr3, *pixelPtr4, *pixelPtr5, *pixelPtr6, *pixelPtr7, *pixelPtr8;
double **filter, **filter2, **filter3, **filter4, **filter5, **filter6, **filter7;
int n = 1;
double **globalFilter;
int height, width;

//***********************************************************************************************************************

double **readFilter(int);  //Reading filter from text file
//CPU calculations
void mainFunction(Mat, unsigned char *, unsigned char *, int, int, double**);
void filterFunction(Mat, unsigned char *, unsigned char *, int, int, int, int, double**);
//GPU calculations
void mainFunctionForCUDA(Mat, unsigned char *, unsigned char *, unsigned int, unsigned int, double**);
__global__ void imageFilterForCUDA(Mat, unsigned char *, unsigned char *, unsigned int, unsigned int, double*, int, int);
int gridToBlock(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
//Interface GLUİ process
void control_cb(int control);
void myGlui();
void mainEngine(int);
void menu();

//**********************************************************************

#define N_ID				200
#define Image_ID			500
#define Text_ID				600
#define Processor_ID		700
#define Type_ID				800
#define Video_ID			900

//**********************************************************************

GLUI *glui1, *glui2;
GLUI_String text1;
int processorFlag = -1;
int typeFlag = -1;
int videoFlag = -1;
int imageFlag = -1;
int buttonFlag = -1;
int main_window = 0;
int readFlag = 0;

//**********************************************************************

#define TILE_W 16
#define TILE_H 16
#define RADIUS floor(height/2)
#define DIAMETER (RADIUS*2+1)
#define APRON (height-1)
#define FILTER_SIZE (DIAMETER*DIAMETER)
#define BLOCK_W (TILE_W + (2*RADIUS))
#define BLOCK_H (TILE_H + (2*RADIUS))
#define BLOCK_DIM 32
#define DIM 512

//**********************************************************************

//**********************************************************************

int main(int argc, const char** argv) {

	//img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	//Mat gray_img;
	////imshow("Original Image", img);
	////cvtColor(img, img, CV_BGR2GRAY);
	//int a = img.cols;
	//int b = img.rows;

	//unsigned char* pixelPtr2, *pixelPtr3, *pixelPtr4, *pixelPtr5, *pixelPtr6, *pixelPtr7, *pixelPtr8;

	//pixelPtr2 = new unsigned char[img.rows*img.cols * 3];
	/*pixelPtr2 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr3 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr4 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr5 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr6 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr7 = new unsigned char[img.rows*img.cols * 3];
	pixelPtr8 = new unsigned char[img.rows*img.cols * 3];*/

	//outputAY POINTERS TO HOLD DIFFERENT FILTERED OUTPUT IMAGES
	/*
	pixelPtr3 = new uint8_t[img.rows*img.cols * 3];
	pixelPtr4 = new uint8_t[img.rows*img.cols * 3];
	pixelPtr5 = new uint8_t[img.rows*img.cols * 3];
	pixelPtr6 = new uint8_t[img.rows*img.cols * 3];
	pixelPtr7 = new uint8_t[img.rows*img.cols * 3];
	pixelPtr8 = new uint8_t[img.rows*img.cols * 3];
	*/

	/*unsigned char* *///pixelPtr = (unsigned char*)img.data;
	int cn = img.channels();

	//file reading 
	//double **filter, **filter2, **filter3, **filter4, **filter5, **filter6, **filter7;


	////*************************************************************************************
	//// Filtering with GPU 

	////Filter 1
	//filter = readFilter(5);
	//mainFunctionForCUDA(img, pixelPtr, pixelPtr2, img.rows, img.cols, filter);
	//Mat img2 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr2);
	//imshow("Original Image", img);
	//imshow("CUDA Discrete appx. to the Laplacian filter(Filter1)", img2);

	////Filter 2
	//filter2 = readFilter(2);
	//mainFunctionForCUDA(img, pixelPtr, pixelPtr3, img.rows, img.cols, filter2);
	//Mat img3 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr3);
	//imshow("CUDA 3x3 Simple filter(Filter2)", img3);

	////Filter 3
	//filter3 = readFilter(4);
	//mainFunctionForCUDA(img, pixelPtr, pixelPtr4, img.rows, img.cols, filter3);
	//Mat img4 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr4);
	//imshow("CUDA Filter 3", img4);

	//// Filtering with GPU
	////**************************************************************************************
	//
	////**************************************************************************************
	////Filtering with CPU

	////Filter 4
	//filter4 = readFilter(4);
	//mainFunction(img, pixelPtr, pixelPtr5, height, width, filter4);
	//Mat img5 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr5);
	//imshow("CPU Discrete appx. to the Laplacian filter 2(Filter4)", img5);

	////Filter 5
	//filter5 = readFilter(5);
	//mainFunction(img, pixelPtr, pixelPtr6, height, width, filter5);
	//Mat img6 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr6);
	//imshow("CPU Gaussian Function(Filter5)", img6);

	////Filter 6
	//filter6 = readFilter(6);
	//mainFunction(img, pixelPtr, pixelPtr7, height, width, filter6);
	//Mat img7 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr7);
	//imshow("CPU Discrete appx to LoG function with Gaussian(Filter6)", img7);

	//filter7 = readFilter(7);
	//mainFunction(img, pixelPtr, pixelPtr8, height, width, filter7);
	//img7 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr8);
	//imshow("CPU More blur", img7);
	//
	////Filtering with CPU
	////************************************************************************************

	//**************************************************************************************
	menu();
	return 0;
}

double **readFilter(int i) {
	char x;
	string line;
	const char space = ' ';
	fstream myFilter;
	int flag = 0, a = 0, b = 0;
	double **filter = 0;
	string txt1, txt2;
	double sayi;
	char k[10];
	x = i + '0';
	txt1 = "filter";
	txt1 += x;
	txt2 = ".txt";
	txt1 += txt2;
	myFilter.open(txt1, ios_base::in);
	cout << txt1;

	//system("pause");

	if (myFilter.is_open()) {  //checks the file if it is opened or not
		while (!myFilter.eof()) {  //loop until end of the file
			while (flag != 1) {  //checks if height and width are initialized
				myFilter >> line;
				if (line == "height" && flag == 0) {
					myFilter >> x;
					height = x - '0';
				}
				if (line == "width" && flag == 0) {
					myFilter >> x;
					width = x - '0';
					flag = 1;
					filter = new double *[width];
					for (int i = 0; i < width; i++) {
						filter[i] = new double[height];
					}
				}
			}
			myFilter >> line;
			filter[a][b] = stod(line.c_str());

			b++;
			if (b == width) {
				a++;
				b = 0;
			}
		}
		myFilter.close();
	}
	else {
		cout << "Unable to open file." << endl;
		return 0;
	}

	cout << "\n";
	cout << "height: " << height;
	cout << "witdh: " << width << endl;

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			cout << filter[i][j] << "  ";
		}
		cout << "\n";
	}

	return filter;
}

void mainFunction(Mat img, uint8_t *pixelPtr, uint8_t *output, int height, int width, double **filter) { //Main fucntion for CPU Calculations
	int n, m;
	double sum = 0;
	for (n = 0; n<height; n++){
		for (m = 0; m<width; m++){
			sum += filter[n][m];
		}
	}
	//cout << "sum: " << sum << endl;

	if (sum != 0) {
		for (n = 0; n < height; n++){
			for (m = 0; m < width; m++){
				filter[n][m] = filter[n][m] * (1 / sum);
			}
		}
	}
	/*cout << "filter after summing the kernel" << endl;
	for (n = 0; n<height; n++){
	for (m = 0; m<width; m++){
	cout << filter[n][m] << "  ";
	}
	cout << endl;
	}*/


	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			filterFunction(img, pixelPtr, output, i, j, height, width, filter);
		}
	}
	//cout << endl;


	/*for (int i = 0; i < width; i++)
	{
	for (int j = 0; j < height; j++)
	{
	cout << filter[i][j] << "  ";
	}
	cout << endl;
	}*/
}

void filterFunction(Mat img, uint8_t *pixelPtr, uint8_t *output, int index_x, int index_y, int height, int width, double **filter) {	//Filter function for CPU calculations
	int i = floor(-height / 2), j = floor(-width / 2);
	int a = floor(height / 2), b = floor(width / 2), c = floor(height / 2), d = floor(width / 2);
	double pixel_b = 0, pixel_g = 0, pixel_r = 0;
	int cn = img.channels();
	int n, m;

	if (index_x <= a && index_y >= b) { i = -index_x; c = index_x; }
	if (index_x >= a && index_y <= b) { j = -index_y; d = index_y; }
	if (index_x <= a && index_y <= b) { i = -index_x; j = -index_y; c = index_x; d = index_y; }
	if (index_x <= img.rows - a && index_y >= img.cols - b) { b = img.cols - index_y; }
	if (index_x >= img.rows - a && index_y <= img.cols - b) { a = img.rows - index_x; }
	if (index_x >= img.rows - a && index_y >= img.cols - b) { a = img.rows - index_x; b = img.cols - index_y; }

	for (i = -c; i <= a; i++) {
		for (j = -d; j <= b; j++) {
			//pixel_b  += filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 0] + filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 1] + filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 2];
			pixel_b += filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 0];
			pixel_g += filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 1];
			pixel_r += filter[i + height / 2][j + width / 2] * pixelPtr[((index_x + i)*img.cols*cn) + (index_y + j)*cn + 2];
		}
	}

	if (pixel_b < 0) pixel_b = 0;
	if (pixel_g < 0) pixel_g = 0;
	if (pixel_r < 0) pixel_r = 0;
	if (pixel_b > 255) pixel_b = 255;
	if (pixel_g > 255) pixel_g = 255;
	if (pixel_r > 255) pixel_r = 255;

	output[((index_x)*img.cols*cn) + (index_y)*cn + 0] = pixel_b;
	output[((index_x)*img.cols*cn) + (index_y)*cn + 1] = pixel_g;
	output[((index_x)*img.cols*cn) + (index_y)*cn + 2] = pixel_r;

}

void mainFunctionForCUDA(Mat img, unsigned char* input, unsigned char* output, unsigned int imageHeight, unsigned int imageWidth, double **filter) {
	unsigned char* dev_input;
	unsigned char* dev_output;
	double *filter_1d;
	double *filter_1dForCuda;
	int index = 0;

	filter_1d = (double*)malloc(height * width * sizeof(double));	//Allocating host memory to copy input filter

	cudaMalloc((void**)&dev_input, imageWidth*imageHeight * 3 * sizeof(unsigned char));		//Allocating device memory to copy input image
	cudaMemcpy(dev_input, input, imageWidth*imageHeight * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);	//Copying input image to allocated memory in device memory

	cudaMalloc((void**)&dev_output, imageWidth*imageHeight * 3 * sizeof(unsigned char));	//Allocating device memory to copy output image

	int n, m;
	double sum = 0;
	for (n = 0; n<height; n++){
		for (m = 0; m<width; m++){
			sum += filter[n][m];
		}
	}
	//cout << "sum: " << sum << endl;

	if (sum != 0) {
		for (n = 0; n < height; n++){
			for (m = 0; m < width; m++){
				filter[n][m] = filter[n][m] * (1 / sum);
			}
		}
	}

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			filter_1d[index++] = filter[i][j];
		}
	}

	cudaMalloc((void**)&filter_1dForCuda, height * width * sizeof(double));	//Allocating device memory to copy filter for device
	cudaMemcpy(filter_1dForCuda, filter_1d, height * width * sizeof(double), cudaMemcpyHostToDevice);	//Copying filter to device memory

	/*dim3 blockDims(16, 16 , 1);
	dim3 gridDims((unsigned int)ceil((double)(width*height / blockDims.x)), 1, 1);*/

	const dim3 gridDims(gridToBlock(imageWidth, BLOCK_DIM), gridToBlock(imageHeight, BLOCK_DIM), 1);
	const dim3 blockDims(BLOCK_DIM, BLOCK_DIM, 1);

	size_t blockSize = (BLOCK_DIM + APRON) * (BLOCK_DIM + APRON) * 3 * sizeof(float); //***** Değişiklik yapıldı
	//cout << "radius: " << RADIUS << endl;
	//cout << "apron: " << APRON << endl;
	imageFilterForCUDA <<< gridDims, blockDims, blockSize >>>(img, dev_input, dev_output, imageWidth, imageHeight, filter_1dForCuda, RADIUS, APRON);//*******değişiklik yapıldı

	cudaMemcpy(output, dev_output, imageHeight*imageWidth * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(dev_input);
	cudaFree(dev_output);
	cudaFree(filter_1dForCuda);
}

__global__ void imageFilterForCUDA(Mat img, unsigned char *inputImage, unsigned char *outputImage, unsigned int imageWidth, unsigned int imageHeight, double *filter, int radius, int apron) {

	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	unsigned int Idx = yIndex*imageWidth + xIndex;

	__shared__ float shMem[BLOCK_DIM + 2][BLOCK_DIM + 2][3];

	//-----------------------------------------------------------------------------------------------------------

	//int radius = floor(width / 2);
	extern __shared__ float shDynmcMem[];
	//float *A = shDynmcMem;
	shDynmcMem[(BLOCK_DIM + apron) * (BLOCK_DIM + apron) * 3]; //******değişiklik yapıldı

	//-----------------------------------------------------------------------------------------------------------

	unsigned int shY = threadIdx.y + radius;
	unsigned int shX = threadIdx.x + radius;

	shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + (shX * 3) + 0] = inputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 0]; //***** değişiklik yapıldı
	shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + (shX * 3) + 1] = inputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 1];	//***** değişiklik yapıldı
	shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + (shX * 3) + 2] = inputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 2];	//***** değişiklik yapıldı
	/*
	if (threadIdx.x == 0) {
	shDynmcMem[((shY) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 0] = 0;    // --- left border
	shDynmcMem[((shY) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 1] = 0;    // --- left border
	shDynmcMem[((shY) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 2] = 0;    // --- left border
	}
	else if (threadIdx.x == BLOCK_DIM - 1) {
	shDynmcMem[((shY)* BLOCK_DIM * 3) + ((shX + 1) * 3) + 0] = 0;    // --- right border
	shDynmcMem[((shY)* BLOCK_DIM * 3) + ((shX + 1) * 3) + 1] = 0;    // --- right border
	shDynmcMem[((shY)* BLOCK_DIM * 3) + ((shX + 1) * 3) + 2] = 0;    // --- right border
	}
	if (threadIdx.y == 0) {
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX) * 3) + 0] = 0;    // --- upper border
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX) * 3) + 1] = 0;    // --- upper border
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX) * 3) + 2] = 0;    // --- upper border
	if (threadIdx.x == 0) {
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX - 1)* 3) + 0] = 0;    // --- top-left corner
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX - 1)* 3) + 1] = 0;    // --- top-left corner
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX - 1)* 3) + 2] = 0;    // --- top-left corner
	}
	else if (threadIdx.x == BLOCK_DIM - 1) {
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 0] = 0;    // --- top-right corner
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 1] = 0;    // --- top-right corner
	shDynmcMem[((shY - 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 2] = 0;    // --- top-right corner
	}
	}
	else if ( threadIdx.y == BLOCK_DIM - 1) {
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX) * 3) + 0] = 0;    // --- bottom border
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX) * 3) + 1] = 0;    // --- bottom border
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX) * 3) + 2] = 0;    // --- bottom border
	if (threadIdx.x == 0) {
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 0] = 0;    // --- bottom-left corder
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 1] = 0;    // --- bottom-left corder
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX - 1) * 3) + 2] = 0;    // --- bottom-left corder
	}
	else if (threadIdx.x == BLOCK_DIM - 1) {
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 0] = 0;    // --- bottom-right corner
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 1] = 0;    // --- bottom-right corner
	shDynmcMem[((shY + 1) * BLOCK_DIM * 3) + ((shX + 1) * 3) + 2] = 0;    // --- bottom-right corner
	}
	}*/

	//*******************************************************************************************************************************************

	if (threadIdx.x < radius && xIndex >= radius) {
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 1] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - radius) * 3) + 1]; //*left edge
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 2] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - radius) * 3) + 2]; //*left edge
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 0] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - radius) * 3) + 0]; //*left edge
	}
	else if (threadIdx.x >= BLOCK_DIM - radius && xIndex < imageWidth - radius && threadIdx.x != BLOCK_DIM) {
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 0] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + radius) * 3) + 0]; //*right edge
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 1] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + radius) * 3) + 1]; //*right edge
		shDynmcMem[(shY * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 2] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + radius) * 3) + 2]; //*right edge
	}
	if (threadIdx.y < radius && yIndex >= radius) {
		shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 0] = inputImage[((yIndex - radius)*imageWidth * 3) + ((xIndex)* 3) + 0]; //*top edge
		shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 1] = inputImage[((yIndex - radius)*imageWidth * 3) + ((xIndex)* 3) + 1]; //*top edge
		shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 2] = inputImage[((yIndex - radius)*imageWidth * 3) + ((xIndex)* 3) + 2]; //*top edge
		if (threadIdx.x < radius && xIndex >= radius) {
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 0] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 0]; //*top left corner
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 1] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 1]; //*top left corner
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 2] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 2]; //*top left corner
		}
		else if (threadIdx.x >= BLOCK_DIM - radius && xIndex < imageWidth - radius && threadIdx.x != BLOCK_DIM) {
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 0] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 0]; //*top right corner
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 1] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 1]; //*top right corner
			shDynmcMem[((shY - radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 2] = inputImage[((yIndex - radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 2]; //*top right corner
		}
	}
	else if (threadIdx.y >= BLOCK_DIM - radius && yIndex < imageHeight - radius && threadIdx.y != BLOCK_DIM) {
		shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 0] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex)* 3) + 0]; //*bottom edge
		shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 1] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex)* 3) + 1]; //*bottom edge
		shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + (shX * 3) + 2] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex)* 3) + 2]; //*bottom edge
		if (threadIdx.x < radius && xIndex >= radius)	{
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 0] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 0]; //*bottom left corner
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 1] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 1]; //*bottom left corner
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX - radius) * 3) + 2] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex - radius) * 3) + 2]; //*bottom left corner
		}
		else if (threadIdx.x >= BLOCK_DIM - radius && xIndex < imageWidth - radius && threadIdx.x != BLOCK_DIM) {
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 0] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 0]; //*bottom right corner
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 1] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 1]; //*bottom right corner
			shDynmcMem[((shY + radius) * (BLOCK_DIM + apron) * 3) + ((shX + radius) * 3) + 2] = inputImage[((yIndex + radius) * imageWidth * 3) + ((xIndex + radius) * 3) + 2]; //*bottom right corner
		}
	}
	//*******************************************************************************************************************************************

	/*if (threadIdx.x == 0 && xIndex > 0)	{
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 0] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - 1) * 3) + 0]; //*left edge
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 1] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - 1) * 3) + 1]; //*left edge
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 2] = inputImage[(yIndex*imageWidth * 3) + ((xIndex - 1) * 3) + 2]; //*left edge
	}
	else if (threadIdx.x == BLOCK_DIM - 1 && xIndex < imageWidth - 1) {
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 0] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + 1) * 3) + 0]; //*right edge
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 1] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + 1) * 3) + 1]; //*right edge
	shDynmcMem[(shY * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 2] = inputImage[(yIndex*imageWidth * 3) + ((xIndex + 1) * 3) + 2]; //*right edge
	}
	if (threadIdx.y == 0 && yIndex > 0) {
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + (shX * 3) + 0] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex) * 3) + 0]; //*top edge
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + (shX * 3) + 1] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex) * 3) + 1]; //*top edge
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + (shX * 3) + 2] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex) * 3) + 2]; //*top edge
	if (threadIdx.x == 0 && xIndex > 0) {
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 0] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex - 1) * 3) + 0]; //*top left corner
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 1] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex - 1) * 3) + 1]; //*top left corner
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 2] = inputImage[((yIndex - 1)*imageWidth * 3) + ((xIndex - 1) * 3) + 2]; //*top left corner
	}
	else if (threadIdx.x == BLOCK_DIM - 1 && xIndex < imageWidth - 1) {
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 0] = inputImage[((yIndex - 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 0]; //*top right corner
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 1] = inputImage[((yIndex - 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 1]; //*top right corner
	shDynmcMem[((shY - 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 2] = inputImage[((yIndex - 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 2]; //*top right corner
	}
	}*/
	/*
	else if (threadIdx.y == BLOCK_DIM - 1 && yIndex < imageHeight - 1) {
	shDynmcMem[((shY + 1)* (BLOCK_DIM+2) * 3) + (shX * 3) + 0] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex) * 3) + 0]; //*bottom edge
	shDynmcMem[((shY + 1)* (BLOCK_DIM+2) * 3) + (shX * 3) + 1] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex) * 3) + 1]; //*bottom edge
	shDynmcMem[((shY + 1)* (BLOCK_DIM+2) * 3) + (shX * 3) + 2] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex) * 3) + 2]; //*bottom edge
	if (threadIdx.x == 0 && xIndex > 0)	{
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 0] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex - 1) * 3) + 0]; //*bottom left corner
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 1] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex - 1) * 3) + 1]; //*bottom left corner
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX - 1) * 3) + 2] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex - 1) * 3) + 2]; //*bottom left corner
	}
	else if (threadIdx.x == BLOCK_DIM - 1 && xIndex < imageWidth - 1) {
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 0] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 0]; //*bottom right corner
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 1] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 1]; //*bottom right corner
	shDynmcMem[((shY + 1) * (BLOCK_DIM+2) * 3) + ((shX + 1) * 3) + 2] = inputImage[((yIndex + 1) * imageWidth * 3) + ((xIndex + 1) * 3) + 2]; //*bottom right corner
	}
	}
	*/
	//-----------------------------------------------------------------------------------------------------------

	__syncthreads();


	//------------------------------------------

	int sumR = 0;
	int sumG = 0;
	int sumB = 0;
	int a = 0;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			sumR += filter[a] * shDynmcMem[((shY + i)*(BLOCK_DIM + apron) * 3) + ((shX + j) * 3) + 0];
			sumG += filter[a] * shDynmcMem[((shY + i)*(BLOCK_DIM + apron) * 3) + ((shX + j) * 3) + 1];
			sumB += filter[a] * shDynmcMem[((shY + i)*(BLOCK_DIM + apron) * 3) + ((shX + j) * 3) + 2];
			a++;
		}
	}

	if (sumR < 0) sumR = 0;
	if (sumG < 0) sumG = 0;
	if (sumB < 0) sumB = 0;
	if (sumR > 255) sumR = 255;
	if (sumG > 255) sumG = 255;
	if (sumB > 255) sumB = 255;

	/*outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 0] = shDynmcMem[((shY)*(BLOCK_DIM+2) * 3) + ((shX) * 3) + 0];
	outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 1] = shDynmcMem[((shY)*(BLOCK_DIM+2) * 3) + ((shX) * 3) + 1];
	outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 2] = shDynmcMem[((shY)*(BLOCK_DIM+2) * 3) + ((shX) * 3) + 2];*/


	outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 0] = sumR;
	outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 1] = sumG;
	outputImage[(yIndex*imageWidth * 3) + (xIndex * 3) + 2] = sumB;

	//**************************************************************************

}

void control_cb(int control)
{
	switch (control)
	{
	case N_ID:
		globalFilter = readFilter(n);
		/*if (glutGetWindow() != main_window) glutSetWindow(main_window);
		glutPostRedisplay();*/
		break;
	case Processor_ID:
		//mainEngine(processorFlag, typeFlag);
		cout << "processorFlag: " << processorFlag << endl;
		/*if (glutGetWindow() != main_window) glutSetWindow(main_window);
		glutPostRedisplay();*/
		break;
	case Type_ID:
		//mainEngine(processorFlag, typeFlag);
		cout << "rotationFlag: " << typeFlag << endl;
		/*if (glutGetWindow() != main_window) glutSetWindow(main_window);
		glutPostRedisplay();*/
		break;
	case Text_ID:
		//mainEngine(processorFlag, typeFlag);
		cout << "File Explore String: " << text1 << endl;
		break;
	case Video_ID:
		//mainEngine(processorFlag, typeFlag);
		cout << "videoFlag: " << videoFlag << endl;
		break;
	case Image_ID:
		//mainEngine(processorFlag, typeFlag);
		cout << "imageFlag: " << imageFlag << endl;
	}
}

void myGlui()
{
	GLUI *glui1 = GLUI_Master.create_glui("GLUI");

	GLUI_Panel *processorPanel = glui1->add_panel("CPU or GPU");
	GLUI_RadioGroup *showWhatRadiobutton = glui1->add_radiogroup_to_panel
		(processorPanel, &processorFlag, Processor_ID, control_cb);
	glui1->add_radiobutton_to_group(showWhatRadiobutton, "None");
	glui1->add_radiobutton_to_group(showWhatRadiobutton, "CPU");
	glui1->add_radiobutton_to_group(showWhatRadiobutton, "GPU");

	glui1->add_statictext("");

	GLUI_Panel *textPanel = glui1->add_panel("File");
	GLUI_EditText *textBox = glui1->add_edittext_to_panel
		(textPanel, "Explore File", GLUI_EDITTEXT_TEXT, text1, NULL, control_cb);

	glui1->add_statictext("");

	GLUI_Panel *imagePanel = glui1->add_panel("Lena or Another");
	GLUI_RadioGroup *showWhatRadiobutton3 = glui1->add_radiogroup_to_panel
		(imagePanel, &imageFlag, Image_ID, control_cb);
	glui1->add_radiobutton_to_group(showWhatRadiobutton3, "None");
	glui1->add_radiobutton_to_group(showWhatRadiobutton3, "Lena");
	glui1->add_radiobutton_to_group(showWhatRadiobutton3, "Another");

	glui1->add_statictext("");

	GLUI_Panel *filePanel = glui1->add_panel("Webcam or File");
	GLUI_RadioGroup *showWhatRadiobutton2 = glui1->add_radiogroup_to_panel
		(filePanel, &videoFlag, Video_ID, control_cb);
	glui1->add_radiobutton_to_group(showWhatRadiobutton2, "None");
	glui1->add_radiobutton_to_group(showWhatRadiobutton2, "Webcam");
	glui1->add_radiobutton_to_group(showWhatRadiobutton2, "File");

	glui1->add_statictext("");

	GLUI_Panel *typePanel = glui1->add_panel("Video or Image");
	GLUI_RadioGroup *showWhatRadiobutton1 = glui1->add_radiogroup_to_panel
		(typePanel, &typeFlag, Type_ID, control_cb);
	glui1->add_radiobutton_to_group(showWhatRadiobutton1, "None");
	glui1->add_radiobutton_to_group(showWhatRadiobutton1, "Image");
	glui1->add_radiobutton_to_group(showWhatRadiobutton1, "Video");

	glui1->add_statictext("");

	GLUI_Panel *filterPanel = glui1->add_panel("Filter");
	GLUI_Spinner *nSpinner = glui1->add_spinner_to_panel
		(filterPanel, "Number", GLUI_SPINNER_INT, &n, N_ID, control_cb);
	nSpinner->set_int_limits(1, 7);
	nSpinner->set_alignment(GLUI_ALIGN_RIGHT);

	GLUI_Button *button = glui1->add_button("Start", buttonFlag, mainEngine);
}

void mainEngine(int control) {
	if (typeFlag == 1) { //Image
		if (imageFlag == 1) {
			img = 0;
			img = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
			if (img.data == 0) //if not success, break loop
			{
				cout << "Cannot read the image" << endl;
				readFlag = -1;
			}
			else {
				readFlag = 0;
			}
			pixelPtr = 0;
			pixelPtr5 = 0;
			if (readFlag == 0) {
				pixelPtr = (unsigned char *)img.data;
				//pixelPtr = new unsigned char[img.cols * img.rows * 3];
				pixelPtr5 = new unsigned char[img.cols * img.rows * 3];
			}
		}
		else if (imageFlag == 2) {
			img = 0;
			img = imread(text1.string, CV_LOAD_IMAGE_COLOR);
			if (img.data == 0) //if not success, break loop
			{
				cout << "Cannot read the image" << endl;
				readFlag = -1;
			}
			else {
				readFlag = 0;
			}
			pixelPtr = 0;
			pixelPtr5 = 0;
			if (readFlag == 0) {
				pixelPtr = (unsigned char *)img.data;
				//pixelPtr = new unsigned char[img.cols * img.rows * 3];
				pixelPtr5 = new unsigned char[img.cols * img.rows * 3];
			}
		}
		else {
			cout << "Lena or File not chosen" << endl;
			readFlag = -1;
		}
		if (readFlag == 0) {
			if (processorFlag == 1) {
				filter4 = readFilter(n);
				mainFunction(img, pixelPtr, pixelPtr5, height, width, filter4);
				Mat img5 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr5);
				imshow("Window", img5);
				imshow("Original Image", img);
				waitKey(0);
				destroyAllWindows();
			}
			else if (processorFlag == 2) {
				filter4 = readFilter(n);
				mainFunctionForCUDA(img, pixelPtr, pixelPtr5, img.rows, img.cols, filter4);
				Mat img5 = Mat(img.rows, img.cols, CV_8UC3, pixelPtr5);
				imshow("Window", img5);
				imshow("Original Image", img);
				waitKey(0);
				destroyAllWindows();
			}
		}
	}
	else if (typeFlag == 2) {  //Video
		VideoCapture *cap;
		cap = new VideoCapture("C:/Users/Erce/Videos/video.mp4");
		if (videoFlag == 1) {
			cap = new VideoCapture(0);
		}
		else if (videoFlag == 2) {
			cap = new VideoCapture(text1.string);
			cout << "videoFlag=2 , text1: " << text1 << endl;
		}
		double fps = (*cap).get(CV_CAP_PROP_FPS);
		cout << "Frame Per Seconds: " << fps << endl;

		int videoHeight = (*cap).get(CV_CAP_PROP_FRAME_HEIGHT);
		int videoWidth = (*cap).get(CV_CAP_PROP_FRAME_WIDTH);

		unsigned char* pixelPtrVideo;
		unsigned char* pixelPtrVideo2;
		unsigned char* pixelPtrVideo3 = new unsigned char[videoHeight * videoWidth * 3];
		unsigned char* pixelPtrVideo4 = new unsigned char[videoHeight * videoWidth * 3];

		globalFilter = readFilter(n);

		Mat img2;
		Mat img5;

		while (1)
		{
			int capFlag = 0;
			Mat frame;
			(*cap).retrieve(frame);
			bool bSuccess = (*cap).read(frame); // read a new frame from video

			pixelPtrVideo = (unsigned char*)frame.data;
			pixelPtrVideo2 = (unsigned char*)frame.data;

			if (!bSuccess) //if not success, break loop
			{
				cout << "Cannot read the frame from video file" << endl;
				capFlag = -1;
				break;
			}

			if (processorFlag == 1 && capFlag == 0) {
				//CPU example************************************************************************************************
				namedWindow("CPU Video", CV_WINDOW_AUTOSIZE);
				mainFunction(frame, pixelPtrVideo, pixelPtrVideo3, height, width, globalFilter);
				img5 = Mat(frame.rows, frame.cols, CV_8UC3, pixelPtrVideo3);
				imshow("CPU Video", img5); //show the frame in "MyVideo" window
				imshow("Original Video", frame);
				//***********************************************************************************************************
			}
			else if (processorFlag == 2 && capFlag == 0) {
				//GPU example************************************************************************************************
				namedWindow("GPU Video", CV_WINDOW_AUTOSIZE);
				mainFunctionForCUDA(frame, pixelPtrVideo2, pixelPtrVideo4, frame.rows, frame.cols, globalFilter);
				img2 = Mat(frame.rows, frame.cols, CV_8UC3, pixelPtrVideo4);
				imshow("GPU Video", img2);
				imshow("Original Video", frame);
				//***********************************************************************************************************
			}
			if (waitKey(30) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			{
				destroyAllWindows();
				cout << "esc key is pressed by user" << endl;
				break;
			}
		}
	}
}

void menu(){
	myGlui();
	//glui1->set_main_gfx_window(main_window);
	glutMainLoop();
}