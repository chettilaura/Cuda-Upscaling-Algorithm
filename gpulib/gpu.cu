#include "gpu.cuh"

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // in the index calculus the first part shows the line, the second the column the third the color
    if(idx < dimCutX*dimCutY*3)
        cutout[idx] = img[(stpntY * width + stpntX)*3 + idx/(dimCutX*3) * width * 3 + idx % (dimCutX*3)];

    __syncthreads();
}

__global__ void scaleGPU(char *cutout, char *scaled, int dimCut, int dimScaled)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = dimScaled / dimCut * 3;

    // In the index calculus the first part shows the line, the second the column the third the color
    if(idx < dimScaled*dimScaled*3)   
        scaled[idx] = cutout[idx/dimScaled/stuffing*dimCut*3 + (idx/3 % dimScaled)/stuffing*9 + idx % 3];
    
    __syncthreads();
}


__global__ void zero_order_zoomingGPU(char *img, char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int stpntX, int stpntY, int width, int height, int outDim, int stuffing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outDim * 3* outDim)
        return;

    if(idx < dimZoomX*dimZoomY*3)
        zoomed[idx] = img[(stpntY * width + stpntX)*3 + idx/(dimZoomX*3) * width * 3 + idx % (dimZoomX*3)];

    __syncthreads();

    zoomed_out[idx] = zoomed[ idx/outDim/stuffing*dimZoomX*3 + (idx/3 % outDim)/stuffing*9 + idx % 3 ];
}

__global__ void gaussianKernelGPU(const int gaussLength, const float gaussSigma, const char dimension, float *kernel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dimension * dimension)
        return;

    float sum = 0;

    kernel[idx] = exp(-((idx / dimension - gaussLength / 2) * (idx / dimension - gaussLength / 2) + (idx % dimension - gaussLength / 2) * (idx % dimension - gaussLength / 2)) / (2 * gaussSigma * gaussSigma));
    sum += kernel[idx];

    __syncthreads();

    kernel[idx] /= sum;
}

__global__ void convGPU(const char *input, char *output, const int dim, const char *mask)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= dim || idy >= dim)  
        return;

    int sum = 0;
    for (int i = 0; i < DIMKERNEL; i++)
    {
        for (int j = 0; j < DIMKERNEL; j++)
        {
            sum += input[(idx + i) + (idy + j) * dim] * mask[i * DIMKERNEL + j];
        }
    }
    output[idx + idy * dim] = sum;
}