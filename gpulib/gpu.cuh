#pragma once

#include "../standlib/stdCu.h"

#define MAX_KERNEL_LENGTH 225
#define MAX_DIM 15

__constant__ float d_kernel[MAX_KERNEL_LENGTH];

__global__ void tilingCudaUpscaling(const char *input, char *output, const int inWidth, const int inHeight, const int outWidth, const int outHeight, const int tileWidth, const int tileHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing);

__global__ void globalCudaUpscaling(const char *input, char *output, const int inWidth, const int inHeight, const int outWidth, const int outHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing);

void loadKernel(const float *kernel, const int dimKernel);


/* DEPRECATED FUNCTIONS */

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY);
// Get cutout from image

__global__ void scaleGPU(const char *input, char *output, const int dimImgIn, const int dimImgMid, const int dimImgOut, const int offset);
// Scale image from cutout

__global__ void scaleImage(const char *input, char *output, const int dimImgIn, const int dimImgMid, const int dimImgW, const int dimImgOut, const int offsetCut, const int offsetScaled, const int stuffing, const int limit);
// Scale image from full image

__global__ void basicConvGPU(const char *input, char *output, const int dimImgIn, const int dimImgOut, const int dimKernel);
// Global convolution bewteen input already created and kernel

__global__ void convGPU(const char *input, char *output, const int dimImgIn, const int dimImgOut, const int dimKernel, const int dimTileIn, const int dimTileOut);
// Tiled convolution bewteen input already created and kernel
