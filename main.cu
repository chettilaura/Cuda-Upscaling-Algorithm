#include "imglib/img.h"
#include <stdio.h>
#include <stdlib.h>
//#include <helper_cuda.h>
#include <string>

#define N 9

signed char sharpness[N] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
__constant__ float mask[N];

__global__ void resizer(char *start,char *output, int dim_lato_start, int dim_lato_output, int dim_lato_mask, int dim_tot_output){

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dim_tot_output)
        return;

}

/*
__global__ void convGPU(char *input, char *output, const int dimS, const int dimB)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= dimB)
        return;

    //output[idx] = input[idx] * mask [0] + input[idx + 1] * mask [1] + input[idx + 2] * mask [2] + input[idx + dim] * mask [3] + input[idx + dim + 1] * mask [4] + input[idx + dim + 2] * mask [5] + input[idx + dim*2] * mask [6] + input[idx + dim*2 + 1] * mask [7] + input[idx + dim*2 + 2] * mask [8];
}
*/

float* gaussianKernel(const int gaussLength, const float gaussSigma){
    const char dimension = 3;
    float *gaussKernel = (float *) malloc(dimension * dimension * sizeof(float));
    float sum = 0;
    for(int i = 0; i < dimension; i++){
        for(int j = 0; j < dimension; j++){
            gaussKernel[i * dimension + j] = exp(-((i - gaussLength / 2) * (i - gaussLength / 2) + (j - gaussLength / 2) * (j - gaussLength / 2)) / (2 * gaussSigma * gaussSigma));
            sum += gaussKernel[i * dimension + j];
        }
    }
    for(int i = 0; i < dimension; i++){
        for(int j = 0; j < dimension; j++){
            gaussKernel[i * dimension + j] /= sum;
        }
    }
    return gaussKernel;
}

void zero_order_zooming(int *img, int *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height)
{

    int x_range_max;
    int y_range_max;

    int *zoomed = (int *) malloc(dimZoomX * dimZoomY * sizeof(int));
    zoomed_out = (int *) malloc(width * height * sizeof(int));

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
        gaussKernel = gaussianKernel(gaussLength, gaussSigma);
        cudaMemcpyToSymbol(mask, gaussKernel, N * sizeof(float));
        free(gaussKernel);
    }
        break;
    
    default:
        printf("Wrong command line input. Use -h or --help for more information\n");
        return -1;
    }

    //fine switch case
    const int dimX = (int)strtol(argv[2], NULL, 10);        // X coordinate of the center of the selection zone
    const int dimY = (int)strtol(argv[3], NULL, 10);        // Y coordinate of the center of the selection zone
    const int dimZoom = (int)strtol(argv[4], NULL, 10);     // Length of the side of the selection mask which has a square shape

    if (dimZoom % 2 != 0)
    {
        printf("Error: dimZoom must be even\n");
        return -1;
    }

    printf("DimX: %d, DimY: %d, dimZoom: %d\n", dimX, dimY, dimZoom);
    if(std::string(argv[1]).size() < 4 || std::string(argv[1]).substr(std::string(argv[1]).size() - 4) != ".ppm")
    {
        printf("Error: input file must be a .ppm file\n");
        return -1;
    }
    RGBImage *img = readPPM(argv[1]);

    // Y boundaries check and mask check
    const int pointY = dimY - dimZoom / 2;
    if (dimY > img->height || dimY < 0)
    {
        printf("Error: dimY outside image boundaries");
        return -1;
    }
    if ((dimY + dimZoom / 2) > img->height - 1 || pointY < 1)
    {
        printf("Error: Y mask outside image boundaries");
        return -1;
    }

    // X boundaries check and mask check
    const int pointX = dimX - dimZoom / 2;
    if (dimX > img->width || dimX < 0)
    {
        printf("Error: dimX outside image boundaries");
        return -1;
    }
    if ((dimX + dimZoom / 2) > img->width - 1 || pointX < 1)
    {
        printf("Error: X mask outside image boundaries");
        return -1;
    }
    
    // Selection    
    const int inConvDim = dimZoom + 2;
    const int outScaleDim = (img->width >= img->height) ? img->width : img->height;    
    const int pxCount = outScaleDim * outScaleDim * 3;
    RGBImage *imgScaled = createPPM(outScaleDim, outScaleDim);
    unsigned char *startingMatrix = (unsigned char *)malloc(inConvDim * inConvDim * 3 * sizeof(unsigned char));    

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
        {
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3];
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3 + 1] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3 + 1];
            startingMatrix[(i + 1) * inConvDim * 3 + (j + 1) * 3 + 2] = img->data[(pointX + j) * 3 + (pointY + i) * img->width * 3 + 2];
        }
    destroyPPM(img);

    char *d_start, *d_scale;
    cudaMalloc((void **)&d_start, inConvDim * inConvDim * 3 * sizeof(char));
    cudaMalloc((void **)&d_scale, outScaleDim * outScaleDim * 3 * sizeof(char));
    cudaMemcpy(d_start, startingMatrix, inConvDim * inConvDim * 3 * sizeof(char), cudaMemcpyHostToDevice);
    free(startingMatrix);

    // Check GPU
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess) {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int maxThreads = prop.maxThreadsPerBlock;
    //calcolo numero blocchi necessari 
    const int blockCeiling = (dimZoom * 3 * dimZoom / maxThreads) + 1;
    //controllo numero blocchi utilizzabili
    if (blockCeiling > prop.maxGridSize[0]){
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }

    //chiamata kernel:passiamo ritaglio iniziale, output, dim ritaglio iniziale, dim output, dim maschera, numero pixel output (non passiamo maschera perchè è già in memoria costante)
    resizer<<<blockCeiling, maxThreads>>>(d_start,d_scale, inConvDim, outScaleDim, dimZoom, pxCount);
    //convGPU<<<blockCeiling, maxThreads>>>(d_start, d_scale, dimZoom*dimZoom*3, pxCount);
    cudaMemcpy(imgScaled->data, d_scale, pxCount * sizeof(char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_start);
    cudaFree(d_scale);
    printf("well done");

    // Print output
    writePPM("output.ppm", imgScaled);
    destroyPPM(imgScaled);

    return 0;
}