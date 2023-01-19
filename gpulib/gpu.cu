#include "gpu.cuh"

__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // in the index calculus the first part shows the line, the second the column the third the color
    if (idx < dimCutX * dimCutY * 3)
    {
        cutout[idx] = img[(stpntY * width + stpntX) * 3 + idx / (dimCutX * 3) * width * 3 + idx % (dimCutX * 3)];
    }
    __syncthreads();
}

__global__ void scaleGPU(char *cutout, char *scaled, int dimCut, int dimScaled, int dimSS, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = dimScaled / dimCut * 3;
    if (idx >= dimScaled * dimScaled * 3)
    {
        return;
    }
    // In the index calculus the first part shows the line, the second the column the third the color
    const char value = cutout[idx / dimScaled / stuffing * dimCut * 3 + (idx / 3 % dimScaled) / stuffing * 9 + idx % 3];
    const int position = offset * 3 + offset * dimSS * 3 + idx / 3 / dimScaled * dimSS * 3 + idx % 3 + idx / 3 % dimScaled * 3;

    __syncthreads();
    scaled[position] = value;
}

__global__ void basicConvGPU(const char *input, char *output, const int dim, const int dimKernel, const int dimB)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (dim * dim))
    {
        return;
    }

    float sum = 0;

    // Compute the convolution
    for (int i = 0; i < dimKernel; i++)
    {
        for (int j = 0; j < dimKernel; j++)
        {
            sum += input[(idx / dim + i) * dim + idx % dim + j + 3] * d_kernel[i * dimKernel + j];
        }
    }

    __syncthreads();
    output[idx] = (unsigned char)sum;
}

__global__ void convGPU(const char *input, char *output, const int dim, const int dimKernel, const int dimB, const int tileDim, const int bigTileDim)
{
    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    int ty = threadIdx.y; // t_row
    int row_o = blockIdx.y * tileDim + ty;
    int row_i = row_o - dimKernel / 2;
    
    int tx = threadIdx.x; // t_col
    int color = blockIdx.x % 3;
    int col_o = blockIdx.x / 3 * (tileDim*3) + tx * 3 + color;
    int col_i = col_o - dimKernel / 2;

    // Load the input image into shared memory
    in_img_shared[tx + ty * bigTileDim] = input[row_o * dimB + col_o];
    __syncthreads();

    float sum = 0;
    if (ty < tileDim && tx < tileDim)
    {
        for (int m_row = 0; m_row < dimKernel; m_row++)
            for (int m_col = 0; m_col < dimKernel; m_col++)
                sum += in_img_shared[(ty + m_row) * bigTileDim + tx + m_col] * d_kernel[m_row * dimKernel + m_col];
        
        if (sum < 0)
            sum = 0;
        if (sum > 256)
            sum = 255;
        if (row_o < dimB && col_o < dimB)output[row_o * dim + col_o] = sum;
    }
}

void loadKernel(const float *kernel, const int dimKernel)
{
    cudaMemcpyToSymbol(d_kernel, kernel, dimKernel * dimKernel * sizeof(float));
}