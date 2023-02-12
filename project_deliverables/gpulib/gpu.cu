#include "gpu.cuh"
#include <cmath>

/**
 * @brief Every thread loads a byte of the input image into shared memory and if it is a thread assigned to the output image computes the convolution with the mask.
 *         The result is stored in the output image.
 * @param [in] input: input image (full)
 * @param [out] output: output image
 * @param [in] inWidth: input image width
 * @param [in] inHeight: input image height
 * @param [in] outWidth: output image width
 * @param [in] outHeight: output image height
 * @param [in] tileWidth: width of the tile
 * @param [in] tileHeight: height of the tile
 * @param [in] maskLength: length of the mask
 * @param [in] offsetCutX: offset of the cutout in the x direction
 * @param [in] offsetCutY: offset of the cutout in the y direction
 * @param [in] stuffing: stuffing factor
 *
 */
__global__ void tilingCudaUpscaling(const unsigned char *input, unsigned char *output, const size_t inWidth, const size_t inHeight, const size_t outWidth, const size_t outHeight, const int tileWidth, const int tileHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing)
{
    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    int ty = threadIdx.y; // t_row
    int row = blockIdx.y * tileHeight + ty;
    int row_i = row / stuffing + offsetCutY;

    int tx = threadIdx.x; // t_col
    int colour = blockIdx.z % 3;
    int col = blockIdx.x * tileWidth + tx;
    int col_i = col / stuffing + offsetCutX;

    //  Load the input image into shared memory
    in_img_shared[ty * blockDim.x + tx] = input[(((row_i < 0) ? 0 : ((row_i < inHeight) ? row_i : (inHeight - 1))) * inWidth + ((col_i < 0) ? 0 : ((col_i < inWidth) ? col_i : (inWidth - 1)))) * 3 + colour];
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
            output[(row * outWidth + col) * 3 + colour] = sum;
    }
}

/**
 * @brief Every thread if it is assigned to the output image computes the convolution with the mask.
 * @param [in] input: input image (full)
 * @param [out] output: output image
 * @param [in] inWidth: input image width
 * @param [in] inHeight: input image height
 * @param [in] outWidth: output image width
 * @param [in] outHeight: output image height
 * @param [in] maskLength: length of the mask
 * @param [in] offsetCutX: offset of the cutout in the x direction
 * @param [in] offsetCutY: offset of the cutout in the y direction
 * @param [in] stuffing: stuffing factor
 *
 */
__global__ void globalCudaUpscaling(const unsigned char *input, unsigned char *output, const size_t inWidth, const size_t inHeight, const size_t outWidth, const size_t outHeight, const int maskLength, const int offsetCutX, const int offsetCutY, const int stuffing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= outWidth * outHeight * 3)
    {
        return;
    }
    int colour = idx % 3;
    int linePos = idx / 3;
    int row_i, col_i;
    int rowOffset = linePos / outWidth;
    int colOffset = linePos % outWidth;

    float sum = 0;
    for (int m_row = 0; m_row < maskLength; m_row++)
    {
        for (int m_col = 0; m_col < maskLength; m_col++)
        {
            row_i = offsetCutY + (rowOffset + m_row) / stuffing;
            col_i = offsetCutX + (colOffset + m_col) / stuffing;
            sum += input[((((row_i < 0) ? 0 : ((row_i < inHeight) ? row_i : (inHeight - 1))) * inWidth + ((col_i < 0) ? 0 : ((col_i < inWidth) ? col_i : (inWidth - 1)))) * 3 + colour)] * d_kernel[m_row * maskLength + m_col];
        }
    }
    if (sum < 0)
    {
        sum = 0;
    }
    if (sum > 255)
    {
        sum = 255;
    }
    output[idx] = sum;
    __syncthreads();
}

/**
 * @brief Copies the kernel to the device's constant memory
 * @param [in] kernel: kernel to be loaded
 * @param [in] dimKernel: dimension of the kernel
 *
 */
void loadKernel(const float *kernel, const int dimKernel)
{
    cudaMemcpyToSymbol(d_kernel, kernel, dimKernel * dimKernel * sizeof(float));
}

/* DEPRECATED FUNCTIONS */

/**
 * @brief Saves to output the cutout of the image img starting from the point (stpntY, stpntX) with dimension (dimCutX, dimCutY)
 * @param [in] img: input image (full)
 * @param [out] cutout: output image (cutout)
 * @param [in] stpntY: starting point in the y direction
 * @param [in] stpntX: starting point in the x direction
 * @param [in] width: width of the image
 * @param [in] dimCutX: dimension of the cutout in the x direction
 * @param [in] dimCutY: dimension of the cutout in the y direction
 *
 */
__global__ void getCutout(char *img, char *cutout, int stpntY, int stpntX, int width, int dimCutX, int dimCutY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // in the index calculus the first part shows the line, the second the column the third the colour
    if (idx < dimCutX * dimCutY * 3)
    {
        cutout[idx] = img[(stpntY * width + stpntX) * 3 + idx / (dimCutX * 3) * width * 3 + idx % (dimCutX * 3)];
    }
    __syncthreads();
}

/**
 * @brief Does cutout and scaling in one step
 * @param [in] input: input image (full)
 * @param [out] output: output image (scaled)
 * @param [in] dimImgIn: dimension of the input image
 * @param [in] dimImgMid: dimension of the intermediate image
 * @param [in] dimImgW: dimension of the image in the width direction
 * @param [in] dimImgOut: dimension of the output image
 * @param [in] offsetCut: offset of the cutout
 * @param [in] offsetScaled: offset of the scaled image
 * @param [in] stuffing: stuffing factor
 * @param [in] limit: limit of the loop
 *
 */
__global__ void scaleImage(const char *input, char *output, const int dimImgIn, const int dimImgMid, const int dimImgW, const int dimImgOut, const int offsetCut, const int offsetScaled, const int stuffing, const int limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= limit)
    {
        return;
    }
    // In the index calculus the first part shows the offset from the starting points, then the line, the third the column and lastly the colour
    const char value = input[offsetCut + idx / dimImgW / stuffing * dimImgIn + ((idx % dimImgW) / (stuffing * 3)) * 3 + idx % 3];
    const int position = offsetScaled + idx / dimImgW * dimImgOut + idx % dimImgW;

    __syncthreads();
    output[position] = value;
}

/**
 * @brief Scales the image from the cutout
 * @param [in] cutout: input image (cutout)
 * @param [out] scaled: output image (scaled)
 * @param [in] dimImgIn: dimension of the input image
 * @param [in] dimImgMid: dimension of the intermediate image
 * @param [in] dimImgOut: dimension of the output image
 * @param [in] offset: offset of the scaled image
 *
 */
__global__ void scaleGPU(const char *cutout, char *scaled, const int dimImgIn, const int dimImgMid, const int dimImgOut, const int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = dimImgMid / dimImgIn * 3;
    if (idx >= dimImgMid * dimImgMid * 3)
    {
        return;
    }
    // In the index calculus the first part shows the line, the second the column the third the colour
    const char value = cutout[idx / dimImgMid / stuffing * dimImgIn * 3 + (idx / 3 % dimImgMid) / stuffing * 9 + idx % 3];
    const int position = offset * 3 + offset * dimImgOut * 3 + idx / 3 / dimImgMid * dimImgOut * 3 + idx % 3 + idx / 3 % dimImgMid * 3;

    __syncthreads();
    scaled[position] = value;
}

/**
 * @brief Does the global memory convolution of the image with the kernel starting from the scaled image
 * @param [in] input: input image (scaled)
 * @param [out] output: output image
 * @param [in] dimImgIn: dimension of the input image
 * @param [in] dimImgOut: dimension of the output image
 * @param [in] dimKernel: dimension of the kernel
 *
 */
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

/**
 * @brief Does the shared memory convolution of the image with the kernel starting from the scaled image
 * @param [in] input: input image (scaled)
 * @param [out] output: output image
 * @param [in] dimImgIn: dimension of the input image
 * @param [in] dimImgOut: dimension of the output image
 * @param [in] dimKernel: dimension of the kernel
 * @param [in] dimTileIn: dimension of the smaller tile
 * @param [in] dimTileOut: dimension of the outwards tile
 *
 */
__global__ void convGPU(const char *input, char *output, const int dimImgIn, const int dimImgOut, const int dimKernel, const int dimTileIn, const int dimTileOut)
{
    // Alloccate shared memory
    extern __shared__ unsigned char in_img_shared[];

    int ty = threadIdx.y; // t_row
    int row = blockIdx.y * dimTileOut + ty;

    int tx = threadIdx.x; // t_col
    int colour = blockIdx.x % 3;
    int col = blockIdx.x / 3 * (dimTileOut * 3) + tx * 3 + colour;

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
