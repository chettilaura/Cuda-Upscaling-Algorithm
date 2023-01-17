#pragma once

#include "../standlib/stdCu.h"

#define MAX_KERNEL_LENGTH 225
#define MAX_DIM 15

__constant__ float d_kernel[MAX_KERNEL_LENGTH];

__global__ void gaussianKernelGPU(const int gaussLength, const float gaussSigma, const char dimension, float *kernel);

__global__ void convGPU(const char *input, char *output, const int dim, const int dimKernel, const int dimB, const int tileDim);

__global__ void scaleGPU(char *cutout, char *scaled, int dimCut, int dimScaled, int dimSS, int offset);

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY);

void loadKernel(const float *kernel,const int dimKernel);