#include "gpu.cuh"

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // in the index calculus the first part shows the line, the second the column the third the color
    if (idx < dimCutX * dimCutY * 3)
        cutout[idx] = img[(stpntY * width + stpntX) * 3 + idx / (dimCutX * 3) * width * 3 + idx % (dimCutX * 3)];

    __syncthreads();
}

__global__ void scaleGPU(char *cutout, char *scaled, int dimCut, int dimScaled, int dimSS, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = dimScaled / dimCut * 3;
    if (idx >= dimScaled * dimScaled * 3)
        return;
    // In the index calculus the first part shows the line, the second the column the third the color
    const char value = cutout[idx / dimScaled / stuffing * dimCut * 3 + (idx / 3 % dimScaled) / stuffing * 9 + idx % 3];
    const int position = offset * 3 + offset * dimSS * 3 + idx / 3 / dimScaled * dimSS * 3 + idx % 3 + idx / 3 % dimScaled * 3;

    __syncthreads();
    scaled[position] = value;
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

__global__ void convGPU(const char *input, char *output, const int dim, const int dimKernel, const int dimB, const int tileDim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (dimB * dimB))
        return;
    
    int sum = 0;
    char col = blockIdx.x % 3;

    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    // Load the input image into shared memory
    in_img_shared[threadIdx.x] = input[blockIdx.x/3*blockDim.x + threadIdx.x/tileDim*dimB + threadIdx.x * 3 + col];  // Either this one
    __syncthreads();

    // Compute the convolution if the thread is inside the image
    if(threadIdx.x < tileDim*tileDim ){
        for (int i = 0; i < dimKernel; i++)
            for (int j = 0; j < dimKernel; j++)
                sum += in_img_shared[threadIdx.x + i * tileDim + j] * d_kernel[i * dimKernel + j];

        if(idx == 0) printf("%d", sum);
        __syncthreads();
        output[col + blockIdx.x/3*blockDim.x + threadIdx.x/tileDim*dim + threadIdx.x * 3] = sum; // Or this one are wrong
    }  
}

void loadKernel(const float *kernel, const int dimKernel)
{
    cudaMemcpyToSymbol(d_kernel, kernel, dimKernel * dimKernel * sizeof(float));
}