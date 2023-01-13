#include "gpu.cuh"

__global__ void zero_order_zoomingGPU(char *img, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width)
        return;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= height)
        return;

    int x_range_max;
    int y_range_max;

    int *zoomed;
    cudaMalloc(&zoomed, dimZoomX * dimZoomY * sizeof(int));


    if (x < 0 || y < 0 || x > width || y > height)
    {
        printf("Errore: coordinate fuori dai bordi dell'immagine");
        return;
    }

    if (width - x < x)
    {
        x_range_max = width - x;
    }
    else
    {
        x_range_max = x;
    }

    if (height - y < y)
    {
        y_range_max = height - y;
    }
    else
    {
        y_range_max = y;
    }

    if (x_range_max < dimZoomX || y_range_max < dimZoomY)
    {
        printf("Errore: dimensione della maschera fuori dai bordi dell'immagine");
        return;
    }
    if (idx < dimZoomX && idy < dimZoomY)
        zoomed[idy * dimZoomX + idx] = img[x + idx + (y + idy) * width];

    float stuffing_bits_x = width / dimZoomX;
    float stuffing_bits_y = height / dimZoomY;
    printf("prima di synch thread");

    int stuffing_x = (int)stuffing_bits_x;
    int stuffing_y = (int)stuffing_bits_y;

    int x_float_stuff = (int)100 / (stuffing_bits_x * 100 - stuffing_x * 100);
    int y_float_stuff = (int)100 / (stuffing_bits_y * 100 - stuffing_y * 100);

    int x_float_stuff_counter = 0;
    int y_float_stuff_counter = 0;
    __syncthreads();

    if (idx < dimZoomX && idy < dimZoomY)
    {
        zoomed_out[idx * width + idy] = zoomed[(idx / stuffing_y) * dimZoomX + (idy / stuffing_x)];
        if (x_float_stuff_counter == x_float_stuff)
        {
            zoomed_out[idx * width + idy] = zoomed[(idx / stuffing_y) * dimZoomX + (idy / stuffing_x) + 1];
            x_float_stuff_counter = 0;
        }
        if (y_float_stuff_counter == y_float_stuff)
        {
            zoomed_out[idx * width + idy] = zoomed[((idx / stuffing_y) + 1) * dimZoomX + (idy / stuffing_x)];
            y_float_stuff_counter = 0;
        }
        x_float_stuff_counter++;
        y_float_stuff_counter++;
    }
    printf("arrivato alla fine");
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