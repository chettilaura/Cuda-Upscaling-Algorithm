#include "imglib/img.h"
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

#define N 9

signed char sharpness[N] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
__constant__ float mask[N];

__global__ void convGPU(char *input, char *output, const int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim*dim)
        return;

    output[idx] = input[idx] * mask [0] + input[idx + 1] * mask [1] + input[idx + 2] * mask [2] + input[idx + dim] * mask [3] + input[idx + dim + 1] * mask [4] + input[idx + dim + 2] * mask [5] + input[idx + dim*2] * mask [6] + input[idx + dim*2 + 1] * mask [7] + input[idx + dim*2 + 2] * mask [8];
}

void zero_order_zooming(int *img, int *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height)
{

    int x_range_max;
    int y_range_max;

    int zoomed = malloc(dimZoomX * dimZoomY * sizeof(int));
    int zoomed_out = malloc(width * height * sizeof(int));

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

    for (int i = 0; i < dimZoomY; i++)
        for (int j = 0; j < dimZoomX; j++)
            zoomed[i * dimZoomX + j] = img[x + j + (y + i) * width];

    float stuffing_bits_x = width / dimZoomX;
    float stuffing_bits_y = height / dimZoomY;

    int stuffing_x = (int)stuffing_bits_x;
    int stuffing_y = (int)stuffing_bits_y;

    int x_float_stuff = (int)100 / (stuffing_bits_x * 100 - stuffing_x * 100);
    int y_float_stuff = (int)100 / (stuffing_bits_y * 100 - stuffing_y * 100);

    int x_float_stuff_counter = 0;
    int y_float_stuff_counter = 0;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            zoomed_out[i * width + j] = zoomed[(i / stuffing_y) * dimZoomX + (j / stuffing_x)];
            if (x_float_stuff_counter == x_float_stuff)
            {
                zoomed_out[i * width + j] = zoomed[(i / stuffing_y) * dimZoomX + (j / stuffing_x) + 1];
                x_float_stuff_counter = 0;
            }
            if (y_float_stuff_counter == y_float_stuff)
            {
                zoomed_out[i * width + j] = zoomed[((i / stuffing_y) + 1) * dimZoomX + (j / stuffing_x)];
                y_float_stuff_counter = 0;
            }
            x_float_stuff_counter++;
            y_float_stuff_counter++;
        }
    }

}

__global__ void scaleGPU(char *input, char *output, const int dim, const int dimSmall)
{    
}

int main(int argc, char **argv)
{
    // Filter Setup
    switch (argc)
    {
    // Help
    case 2:
        if (argv[1] == std::string("-h") || argv[1] == std::string("--help"))
        {
            printf("Usage: %s inputFile.ppm DimX DimY dimZoom mode [GaussLength GaussSigma]\n\nWhere:\n\tDimX: X coordinate of the center of the selection zone\n\tDimY: Y coordinate of the center of the selection zone\n\tdimZoom: Length of the side of the selection mask which has a square shape\n\tmode: 0 for Sharpness filter, 1 for Gaussian*, 2 for custom 3x3 Kernel*\n*Input required from the user\n", argv[0]);
            return 0;
        }
        else
        {
            printf("Wrong command line input. Use -h or --help for more information\n");
            return -1;
        }
        break;

    // Sharpness or Custom Kernel
    case 6:{
        int mode = (int)strtol(argv[5], NULL, 10);
        if (mode == 0)
            cudaMemcpyToSymbol(mask, sharpness, N * sizeof(char));
        else if (mode == 2)
        {
            printf("Insert the 3x3 kernel values, from left to right, from top to bottom to a single line where each value is separated by a space\n");
            const char M = N * 4;
            char buff[M];
            fgets(buff, M, stdin);

            // Check input
            int bLength = strlen(buff);
            if (bLength < 18) // 9 values, 8 spaces
            {
                printf("Wrong input. Use -h or --help for more information\n");
                return -1;
            }
            float *kernel = (float *)malloc(N * sizeof(float));
            if (sscanf(buff, "%f %f %f %f %f %f %f %f %f", &kernel[0], &kernel[1], &kernel[2], &kernel[3], &kernel[4], &kernel[5], &kernel[6], &kernel[7], &kernel[8]) != 9)
            {
                printf("Wrong input. Use -h or --help for more information\n");
                return -1;
            }
            cudaMemcpyToSymbol(mask, kernel, N * sizeof(float));
            free(kernel);
        }
        else
        {
            printf("Wrong command line input. Use -h or --help for more information\n");
            return -1;
        }
    }
        break;

    // Gaussian
    case 8:{
        int mode = (int)strtol(argv[5], NULL, 10);
        if (mode != 1)
        {
            printf("Wrong command line input. Do not input gaussian data for non-gaussian matrices. Use -h or --help for more information\n");
            return -1;
        }
        int gaussLength = (int)strtol(argv[6], NULL, 10);
        float gaussSigma = (float)strtof(argv[7], NULL);
        if (gaussLength < 3 || gaussLength > 15 || gaussSigma < 0.5 || gaussSigma > 5)
        {
            printf("Wrong Gaussian values:\nACCEPTED VALUES:\n\t 3 <= gaussLength <= 15\n\t 0.5 <= gaussSigma <= 5\nAborting...\n");
            return -1;
        }
        float *gaussKernel = (float *)malloc(N * sizeof(float));
        // TODO: implementare gaussianKernel
        // gaussKernel = gaussianKernel(gaussLength, gaussSigma);
        cudaMemcpyToSymbol(mask, gaussKernel, N * sizeof(float));
        free(gaussKernel);
    }
        break;
    
    default:
        printf("Wrong command line input. Use -h or --help for more information\n");
        return -1;
    }

    dimX = (int) strtol(argv[2], NULL, 10);
    dimY = (int) strtol(argv[3], NULL, 10);
    dimZoom = (int) strtol(argv[4], NULL, 10);

    printf("DimX: %d, DimY: %d, dimZoom: %d\n", dimX, dimY, dimZoom);


    GrayImage *img = readPGM(argv[1]);
    if (img == NULL)
    {
        printf("Errore: dimZoom deve essere pari\n");
        return -1;
    }

    printf("DimX: %d, DimY: %d, dimZoom: %d\n", dimX, dimY, dimZoom);

    RGBImage *img = readPPM(argv[1]);

    // Check per Y
    if (dimY > img->height || dimY < 0)
    {
        printf("Errore: Y fuori dai bordi dell'immagine");
        return -1;
    }
    {
        const int boardYup = dimY - dimZoom / 2;
        const int boardYdown = dimY + dimZoom / 2;
        if (boardYdown > img->height - 1 || boardYup < 1)
        {
            printf("Errore: Maschera Y fuori dai bordi dell'immagine");
            return -1;
        }
    }

    // Check per X
    if (dimX > img->width || dimX < 0)
    {
        printf("Errore: X fuori dai bordi dell'immagine");
        return -1;
    }
    {
        const int boardXup = dimX + dimZoom / 2;
        const int boardXdown = dimX - dimZoom / 2;
        if (boardXup > img->width - 1 || boardXdown < 1)
        {
            printf("Errore: Maschera X fuori dai bordi dell'immagine");
            return -1;
        }
    }

    // Selezione
    // What is the order between scaling and convolution?
    
    const int inConvDim = dimZoom + 2;
    const int outScaleDim = (img->width >= img->height) ? img->width : img->height;    
    const int pxCount = outScaleDim * outScaleDim * 3;
    RGBImage *imgConvWorked = createPPM(inConvDim, inConvDim);
    RGBImage *imgScaled = createPPM(outScaleDim, outScaleDim);
    
    unsigned char *startingMatrix = (unsigned char *)malloc(inConvDim * inConvDim * 3 * sizeof(unsigned char));    
    const int pointX = dimX - dimZoom / 2;
    const int pointY = dimY - dimZoom / 2;

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
        {
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3];
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3 + 1] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3 + 1];
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3 + 2] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3 + 2];
        }

    destroyPPM(img);

    char *d_start, *d_Scale, *d_Conv;    

    cudaMalloc((void **)&d_start, inConvDim * inConvDim * 3 * sizeof(char));
    cudaMalloc((void **)&d_Scale, outScaleDim * outScaleDim * 3 * sizeof(char));
    cudaMalloc((void **)&d_Conv, dimZoom * dimZoom * 3 * sizeof(char));
    cudaMemcpy(d_start, startingMatrix, inConvDim * inConvDim * 3 * sizeof(char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(startingMatrix);

    // Check GPU
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }


    
    int *d_imgCenter, *d_imgLeft, *d_imgRight, *d_imgUp, *d_imgDown, *d_imgConv;
    cudaMalloc((void **)&d_imgCenter, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgLeft, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgRight, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgUp, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgDown, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgConv, dimZoom * dimZoom * sizeof(int));


    // Convoluzione
    GrayImage *imgConv = createPGM(dimZoom, dimZoom);
    int thread= dimZoom*dimZoom;
    superSampler<<<1, 1>>>(d_imgCenter, d_imgLeft, d_imgRight, d_imgUp, d_imgDown, d_imgConv, dimZoom);
    cudaDeviceSynchronize();


    // Scale
    const int blockCeilingScale = (pxCount / maxThreads) + 1;
    scaleGPU<<<blockCeilingScale, maxThreads>>>(d_Conv, d_Scale, outScaleDim, dimZoom);    
    cudaDeviceSynchronize();

    //cudaMemcpy(imgFinalWorked->data, d_endScale, pxCount * sizeof(char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_start);
    cudaFree(d_Conv);
    cudaFree(d_Scale);
    printf("well done");

    // Stampa
    //writePPM("output.ppm", imgFinalWorked);
    destroyPPM(imgScaled);
    destroyPPM(imgConvWorked);

    return 0;
}