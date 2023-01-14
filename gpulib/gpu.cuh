#pragma once

#include "../standlib/stdCu.h"

__global__ void zero_order_zoomingGPU(char *img, char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim, int stuffing);

__global__ void gaussianKernelGPU(const int gaussLength, const float gaussSigma, const char dimension, float *kernel);

__global__ void convGPU(const char *input, char *output, const int dim);

__global__ void scaleGPU(char *cutout, char *scaled, int dimCut, int dimScaled);

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY);