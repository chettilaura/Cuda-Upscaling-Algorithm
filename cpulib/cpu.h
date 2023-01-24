#pragma once

#include "../imglib/img.h"
#include "../standlib/stdCu.h"

void convCPU(char *input, char *output, char *kernel, const int width, const int heigth);
void gaussianKernelCPU(const int gaussLength, const float gaussSigma, float *outkernel);
void zero_order_zoomingCPU(unsigned char *img, unsigned char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim);
bool checkTiling(const int width, const int height, int *dimTilesX, int *dimTilesY);
int getNumTilesPerBlock(int maxElem, const int dim);