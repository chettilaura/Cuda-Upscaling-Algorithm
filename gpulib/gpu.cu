#include "gpu.cuh"
#include <cmath>

__global__ void tilingCudaUpscaling(const unsigned char *input, unsigned char *output, const int inWidth, const int inHeight, const int outWidth, const int outHeight, const int tileWidth, const int tileHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing)
{
    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    int ty = threadIdx.y; // t_row
    int row = blockIdx.y * tileHeight + ty;
    int row_i = row / stuffing + offsetCutY;

    int tx = threadIdx.x; // t_col
    int color = blockIdx.z % 3;
    int col = blockIdx.x * tileWidth + tx;
    int col_i = col / stuffing + offsetCutX;

    //  Load the input image into shared memory
    in_img_shared[ty * blockDim.x + tx] = input[(((row_i < 0) ? 0 : ((row_i < inHeight) ? row_i : (inHeight - 1))) * inWidth + ((col_i < 0) ? 0 : ((col_i < inWidth) ? col_i : (inWidth - 1)))) * 3 + color];
    __syncthreads();

    float sum = 0;
    if (ty < tileHeight && tx < tileWidth)
    {
        for (int m_row = 0; m_row < maskLength; m_row++)
            for (int m_col = 0; m_col < maskLength; m_col++)
                sum += in_img_shared[(ty + m_row) * blockDim.x + tx + m_col] * d_kernel[m_row * maskLength + m_col];

        if (sum < 0)
            sum = 0;
        if (sum > 255)
            sum = 255;
        if (row < outHeight && col < outWidth)
            output[(row * outWidth + col) * 3 + color] = sum;
    }
}


__global__ void globalCudaUpscaling(const unsigned char *input, unsigned char *output, const int inWidth, const int inHeight, const int outWidth, const int outHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing)
{    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outWidth * outHeight * 3)
    {
        return;
    }
    int color = idx % 3;
    int linePos = idx / 3;
    int row_i, col_i;

    float sum = 0;
    for (int m_row = 0; m_row < maskLength; m_row++)
        for (int m_col = 0; m_col < maskLength; m_col++){
            row_i = offsetCutY + (linePos / outWidth + m_row) / stuffing;
            col_i = offsetCutX + (linePos % outWidth + m_col) / stuffing;
            sum += ((row_i >= 0 ) && (row_i < inHeight) && (col_i >= 0) && (col_i < inWidth)) ? (input[((row_i) * inWidth + col_i) * 3 + color] * d_kernel[m_row * maskLength + m_col]) : 0;
        }
            
    if (sum < 0)
        sum = 0;
    if (sum > 255)
        sum = 255;
    output[idx] = sum;
    __syncthreads();
}


void loadKernel(const float *kernel, const int dimKernel)
{
    cudaMemcpyToSymbol(d_kernel, kernel, dimKernel * dimKernel * sizeof(float));
}


/* DEPRECATED FUNCTIONS */

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

__global__ void scaleImage(const char *input, char *output, const int dimImgIn, const int dimImgMid, const int dimImgW, const int dimImgOut, const int offsetCut, const int offsetScaled, const int stuffing, const int limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit)
    {
        return;
    }
    // In the index calculus the first part shows the offset from the starting points, then the line, the third the column and lastly the color
    const char value = input[offsetCut + idx / dimImgW / stuffing * dimImgIn + ((idx % dimImgW) / (stuffing * 3)) * 3 + idx % 3];
    const int position = offsetScaled + idx / dimImgW * dimImgOut + idx % dimImgW;

    __syncthreads();
    output[position] = value;
}

__global__ void scaleGPU(const char *cutout, char *scaled, const int dimImgIn, const int dimImgMid, const int dimImgOut, const int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = dimImgMid / dimImgIn * 3;
    if (idx >= dimImgMid * dimImgMid * 3)
    {
        return;
    }
    // In the index calculus the first part shows the line, the second the column the third the color
    const char value = cutout[idx / dimImgMid / stuffing * dimImgIn * 3 + (idx / 3 % dimImgMid) / stuffing * 9 + idx % 3];
    const int position = offset * 3 + offset * dimImgOut * 3 + idx / 3 / dimImgMid * dimImgOut * 3 + idx % 3 + idx / 3 % dimImgMid * 3;

    __syncthreads();
    scaled[position] = value;
}

__global__ void basicConvGPU(const char *input, char *output, const int dimImgIn, const int dimImgOut, const int dimKernel)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (dimImgOut * dimImgOut))
    {
        return;
    }

    float sum = 0;

    // Compute the convolution
    for (int i = 0; i < dimKernel; i++)
    {
        for (int j = 0; j < dimKernel; j++)
        {
            sum += input[(idx / dimImgIn + i) * dimImgIn + idx % dimImgIn + j + 3] * d_kernel[i * dimKernel + j];
        }
    }

    __syncthreads();
    output[idx] = (unsigned char)sum;
}

__global__ void convGPU(const char *input, char *output, const int dimImgIn, const int dimImgOut, const int dimKernel, const int dimTileIn, const int dimTileOut)
{
    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    int ty = threadIdx.y; // t_row
    int row = blockIdx.y * dimTileOut + ty;

    int tx = threadIdx.x; // t_col
    int color = blockIdx.x % 3;
    int col = blockIdx.x / 3 * (dimTileOut * 3) + tx * 3 + color;

    // Load the input image into shared memory
    in_img_shared[tx + ty * dimTileIn] = input[row * dimImgIn + col];
    __syncthreads();

    float sum = 0;
    if (ty < dimTileOut && tx < dimTileOut)
    {
        for (int m_row = 0; m_row < dimKernel; m_row++)
            for (int m_col = 0; m_col < dimKernel; m_col++)
                sum += in_img_shared[(ty + m_row) * dimTileIn + tx + m_col] * d_kernel[m_row * dimKernel + m_col];

        if (sum < 0)
            sum = 0;
        if (sum > 256)
            sum = 255;
        if (row < dimImgIn && col < dimImgIn)
            output[row * dimImgOut + col] = sum;
    }
}
