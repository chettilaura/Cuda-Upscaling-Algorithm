#pragma once

#include "../standlib/stdCu.h"

__global__ void zero_order_zoomingGPU(char *img, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height);

__global__ void gaussianKernelGPU(const int gaussLength, const float gaussSigma, const char dimension, float *kernel);

__global__ void convGPU(const char *input, char *output, const int dim);

