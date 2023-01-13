#pragma once

void convCPU(char *input, char *output, char *kernel, const int width, const int heigth);
void gaussianKernelCPU(const int gaussLength, const float gaussSigma, float *outkernel);
void zero_order_zoomingCPU(int *img, int *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height);