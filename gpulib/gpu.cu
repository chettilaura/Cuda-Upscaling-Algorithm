#include "gpu.cuh"

__global__ void zero_order_zoomingGPU(char *img, char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width)
        return;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= height)
        return;

    if (idx < dimZoomX*3 && idy < dimZoomY*3)
        zoomed[(idy + 1) * dimZoomX + (idx + 1)] = img[(x + idx) + (y + idy) * width];



    int stuffing = outDim / dimZoomX;
    __syncthreads();
    //printf("%d\n", img[(idx + x)+ dimZoomX * (idy + y)]);
    //printf("%d\n", zoomed[(idx / stuffing + 1) * dimZoomX * 3 + (idy / stuffing + 1) * 3 + idx % 3]);
    //zoomed_out[idx * outDim * 3 + idy * 3 + idx % 3 ] = zoomed[(idx / stuffing + 1) * dimZoomX * 3 + (idy / stuffing + 1) * 3 + idx % 3];

    zoomed_out[idy * outDim + idx] = zoomed[(idy / stuffing + 1) * dimZoomX + (idx / stuffing + 1)];
    //printf("%d\n", zoomed_out[idy * outDim + idx]);
   //zoomed_out[idx + idy * outDim] = zoomed[(idx / stuffing + 1) + (idy / stuffing + 1) * dimZoomX];

}

//ritaglio avvenuto lato CPU
__global__ void scaleGPU(char *zoomed, char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stuffing = outDim / dimZoomX * 3;
    // DimZoom is the dimension of small image, outDim is the dimension of the big image, stuffing is the number of bytes to repeat
    if(idx < outDim*3*outDim)   // idx in zoomed out 1200 deve essere uguale a 300
        
        
        zoomed_out[idx] = zoomed[ idx/stuffing*3 + idx % 3  ];
        
            //ispiration da cpu:
           // zoomed_out[i * outDim * 3 + j * 3] = zoomed[(i / stuffing ) * dimZoomX * 3 + (j / stuffing +) * 3];

            //relazione
           //i->idx
           //j->idx/(outDim*3)

    //printf("idx: %d, idy: %d", idx, idy);
    /*if( idx<(outDim*3*outDim) )
        zoomed_out[idx * outDim * 3 + j * 3] = zoomed[(i / stuffing + 1) * dimZoomX * 3 + (j / stuffing + 1) * 3];*/
        //zoomed_out[idx] = zoomed[(idx / stuffing + 1) * dimZoomX + (idx / stuffing + 1) + idx % 3];
        //zoomed_out[idx] = zoomed[idx];
        //zoomed_out[idx] = 255;//zoomed[idy*dimZoomX+idx];
        //zoomed_out[idy * outDim + idx] = zoomed[(idy / stuffing + 1) * dimZoomX + (idx / stuffing + 1)];
    
    
    __syncthreads();
    //zoomed_out[idy * outDim + idx] = zoomed[(idy / stuffing + 1) * dimZoomX + (idx / stuffing + 1)];
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