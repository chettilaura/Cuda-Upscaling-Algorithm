#include "imglib/img.h"
#include "cpulib/cpu.h"
#include "gpulib/gpu.cuh"
#include "standlib/stdCu.h"

#define N 9

#define DEBUG 1

signed char sharpness[N] = {0, -1, 0, -1, 4, -1, 0, -1, 0};

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

    // Mask variables
    float *d_mask;
    char maskDim = 0;

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
        {
            cudaMemcpy(&d_mask, sharpness, N * sizeof(float), cudaMemcpyHostToDevice);
            maskDim = N;
        }
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
            cudaMemcpy(&d_mask, kernel, N * sizeof(float), cudaMemcpyHostToDevice);
            maskDim = N;
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
            printf("Wrong Gaussian values:\nACCEPTED VALUES:\n\t 3 <= gaussLength (must be odd) <= 15\n\t 0.5 <= gaussSigma <= 5\nAborting...\n");
            return -1;
        }
        float *gaussKernel = (float *)malloc(gaussLength * gaussLength * sizeof(float));
        gaussianKernelCPU(gaussLength, gaussSigma, gaussKernel);
        cudaMemcpy(&d_mask, gaussKernel, gaussLength * gaussLength * sizeof(float), cudaMemcpyHostToDevice);
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

    const int outScaleDim = (img->width >= img->height) ? ((int)img->height / dimZoom) * dimZoom : ((int)img->width / dimZoom) * dimZoom;
    const int pxCount = outScaleDim * outScaleDim * 3;

    char *d_start, *d_scale, *d_zoom;
    cudaMalloc((void **)&d_start, pxCount * sizeof(char));
    cudaMalloc((void **)&d_scale, pxCount * sizeof(char));
    cudaMalloc((void **)&d_zoom, dimZoom * 3 * dimZoom * sizeof(char));
    cudaMemcpy(d_start, img->data, pxCount * sizeof(char), cudaMemcpyHostToDevice);

    int neededThreads = dimZoom * dimZoom * 3;
    int usedThreads = (neededThreads > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : neededThreads;
    // calcolo numero blocchi necessari
    int usedBlocks = (neededThreads / prop.maxThreadsPerBlock) + 1;
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

    // Get the cutout of the image
    getCutout<<<usedBlocks, usedThreads>>>(d_start, d_zoom, pointY, pointX, img->width, dimZoom, dimZoom);
    cudaDeviceSynchronize();
    destroyPPM(img);
    cudaFree(d_start);
#if DEBUG
    RGBImage *imgCut = createPPM(dimZoom, dimZoom);
    cudaMemcpy(imgCut->data, d_zoom, dimZoom * dimZoom * 3 * sizeof(char), cudaMemcpyDeviceToHost);
    writePPM("DEBUG_cutout.ppm", imgCut);
    destroyPPM(imgCut);
    printf("\tDone Scaling\n");
#endif

    // Zooming
    usedThreads = (pxCount > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : pxCount;
    usedBlocks = (pxCount / prop.maxThreadsPerBlock) + 1;
    if (usedBlocks > prop.maxGridSize[0])
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }
    scaleGPU<<<usedBlocks, usedThreads>>>(d_zoom, d_scale, dimZoom, outScaleDim);
    cudaDeviceSynchronize();
    cudaFree(d_zoom);
#if DEBUG
    RGBImage *imgScale = createPPM(outScaleDim, outScaleDim);
    cudaMemcpy(imgScale->data, d_scale, pxCount * sizeof(char), cudaMemcpyDeviceToHost);
    writePPM("DEBUG_scaled.ppm", imgScale);
    destroyPPM(imgScale);
    printf("\tDone Zooming\n");
#endif

    // Convolution
    char *d_out;
    cudaMalloc((void **)&d_out, pxCount * sizeof(char));
    // convGPU<<<usedBlocks, usedThreads>>>(d_scale, d_out, outScaleDim * 3);
    cudaDeviceSynchronize();
#if DEBUG
    printf("\tDone Convoluting\nEND OF GPU INSTRUCTIONS\n\n");
#endif

    RGBImage *imgOut = createPPM(outScaleDim, outScaleDim);
    cudaMemcpy(imgOut->data, d_out, pxCount * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_scale);
    cudaFree(d_out);

    // Print output
    writePPM("output.ppm", imgOut);
    destroyPPM(imgOut);

#if DEBUG
    printf("END OF THE PROGRAM\n\n");
#endif
    return 0;
}