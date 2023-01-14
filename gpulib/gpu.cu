#include "gpu.cuh"

__global__ void zero_order_zoomingGPU(char *img, char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outDim * 3* outDim)
        return;

    if(idx < dimZoomX*dimZoomY*3)
        zoomed[idx] = img[ (idx/3 % dimZoomX)*3 + (idx/3 / dimZoomX)*dimZoomX*3 + idx % 3 ];



    int stuffing = outDim / dimZoomX;
    __syncthreads();

    zoomed_out[idx] = zoomed[ idx/outDim/stuffing*dimZoomX*3 + (idx/3 % outDim)/stuffing*9 + idx % 3 ];

}

//ritaglio avvenuto lato CPU
__global__ void scaleGPU(char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = outDim / dimZoomX * 3;

    // La prima parte di zoomed indica la riga, la seconda la colonna la terza il colore
    if(idx < outDim*3*outDim)   
        zoomed_out[idx] = zoomed[ idx/outDim/stuffing*dimZoomX*3 + (idx/3 % outDim)/stuffing*9 + idx % 3 ];
    

    __syncthreads();
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