#include "imglib/img.h"
#include "cpulib/cpu.h"
#include "gpulib/gpu.cuh"
#include "standlib/stdCu.h"

#define N 9

#define DEBUG 1

signed char sharpness[N] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
__constant__ float mask[N];

int main(int argc, char **argv)
{
    // Check GPU
    int nDevices;
    cudaError_t err = cudaGetDeviceCount(&nDevices);
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }
    if (nDevices == 0)
    {
        printf("No CUDA device found\n");
        return -1;
    }
    #if DEBUG  
    printf("Number of CUDA devices: %d\n", nDevices);
    #endif

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

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
    case 6:
    {
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
    case 8:
    {
        int mode = (int)strtol(argv[5], NULL, 10);
        if (mode != 1)
        {
            printf("Wrong command line input. Do not input gaussian data for non-gaussian matrices. Use -h or --help for more information\n");
            return -1;
        }
        int gaussLength = (int)strtol(argv[6], NULL, 10);
        float gaussSigma = (float)strtof(argv[7], NULL);
        if (gaussLength < 3 || gaussLength > 15 || gaussLength % 2 == 1 || gaussSigma < 0.5 || gaussSigma > 5)
        {
            printf("Wrong Gaussian values:\nACCEPTED VALUES:\n\t 3 <= gaussLength (odd) <= 15\n\t 0.5 <= gaussSigma <= 5\nAborting...\n");
            return -1;
        }
        float *gaussKernel = (float *)malloc(N * sizeof(float));
        gaussianKernelCPU(3, gaussSigma, gaussKernel);
        //gaussianKernelCPU(gaussLength, gaussSigma, gaussKernel);   // needed when dynamic approach used
        cudaMemcpyToSymbol(mask, gaussKernel, N * sizeof(float));
        free(gaussKernel);
    }
    break;

    default:
        printf("Wrong command line input. Use -h or --help for more information\n");
        return -1;
    }

    // fine switch case
    const int dimX = (int)strtol(argv[2], NULL, 10);    // X coordinate of the center of the selection zone
    const int dimY = (int)strtol(argv[3], NULL, 10);    // Y coordinate of the center of the selection zone
    const int dimZoom = (int)strtol(argv[4], NULL, 10); // Length of the side of the selection mask which has a square shape

    // check dimZoom is even
    if (dimZoom % 2 != 0)
    {
        printf("Error: dimZoom must be even\n");
        return -1;
    }

    #if DEBUG
    printf("DimX: %d, DimY: %d, dimZoom: %d\n", dimX, dimY, dimZoom);
    #endif

    // Check input file ends with .ppm
    if (std::string(argv[1]).size() < 4 || std::string(argv[1]).substr(std::string(argv[1]).size() - 4) != ".ppm")
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

    
    const int neededThreads = dimZoom * 3 * dimZoom;
    const short usedThreads = (neededThreads > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : neededThreads; 
    // calcolo numero blocchi necessari
    const int usedBlocks = (neededThreads / prop.maxThreadsPerBlock) + 1;
    // controllo numero blocchi utilizzabili
    if (usedBlocks > prop.maxGridSize[0])
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }
    #if DEBUG
    printf("Used Threads: %d - Used Blocks: %d\n", usedThreads, usedBlocks);
    #endif
    #if DEBUG
    printf("END OF CPU INSTRUCTIONS\n\n");
    #endif

    // chiamata kernel:passiamo ritaglio iniziale, output, dim ritaglio iniziale, dim output, dim maschera, numero pixel output (non passiamo maschera perchè è già in memoria costante)
    //resizer<<<usedBlocks, maxThreads>>>(d_start, d_scale, inConvDim, outScaleDim, dimZoom, pxCount);
    // convGPU<<<usedBlocks, maxThreads>>>(d_start, d_scale, dimZoom*dimZoom*3, pxCount);
    zero_order_zoomingGPU<<<usedBlocks, usedThreads>>>(d_start, d_scale, dimZoom, dimZoom, dimX, dimY, outScaleDim, outScaleDim);
    cudaDeviceSynchronize();
    //convGPU<<<usedBlocks, usedThreads>>>(d_scale, d_start, outScaleDim * 3);
    //cudaDeviceSynchronize();
    cudaMemcpy(imgScaled->data, d_scale, pxCount * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_start);
    cudaFree(d_scale);
    printf("well done");

    // Print output
    writePPM("output.ppm", imgScaled);
    destroyPPM(imgScaled);

    return 0;
}