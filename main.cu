#include "imglib/img.h"
#include "cpulib/cpu.h"
#include "gpulib/gpu.cuh"
#include "standlib/stdCu.h"

#define N 9
#define DEBUG 1

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
    char maskDim = 0;

    // Filter Setup
    switch (argc)
    {
    // Help
    case 2:
        if (argv[1] == std::string("-h") || argv[1] == std::string("--help"))
        {
            printf("Usage: %s inputFile.ppm DimX DimY dimZoom mode (GaussLength GaussSigma) OR InputKernelFile.txt\n\nWhere:\n\tDimX: X coordinate of the center of the selection zone\n\tDimY: Y coordinate of the center of the selection zone\n\tdimZoom: Length of the side of the selection mask which has a square shape\n\tmode: 0 for Gaussian filter, 1  for custom kernel loaded from file\n\n\tGaussLength: must be an odd value from 3 to 15 sides include\n\tGaussSigma: must be a value from 0.5 to 5 sides included\n\n\tInputKernelFile.txt: formatted as such\n\n\t\t\tmatrixSide'sLength\n\t\t\tFirstElement SecondElement ...\n\t\t\tRowElement ...\n\t\t\t...\n", argv[0]);
            return 0;
        }
        else
        {
            printf("Wrong command line input. Use -h or --help for more information\n");
            return -1;
        }
        break;

    // Custom Kernel from file
    case 7:
    {
        int mode = (int)strtol(argv[5], NULL, 10);
        if (mode != 1)
        {
            printf("Wrong command line input. Do not input gaussian data for non-gaussian matrices. Use -h or --help for more information\n");
            return -1;
        }
        FILE *kernelFile = fopen(argv[6], "r");
        if (kernelFile == NULL)
        {
            printf("Error opening file %s\n", argv[6]);
            return -1;
        }

        // Read file
        char buff[120];
        fgets(buff, 100, kernelFile);
        maskDim = (char)strtol(buff, NULL, 10);
        if (maskDim < 3 || maskDim > 15 || maskDim % 2 == 0)
        {
            printf("Wrong mask dimension. Use -h or --help for more information\n");
            return -1;
        }

        // Allocate memory
        float *kernel = (float *)malloc(maskDim * maskDim * sizeof(float));
        for (int i = 0; i < maskDim; i++)
        {
            fgets(buff, 120, kernelFile);
            if (sscanf(buff, "%f %f %f %f %f %f %f %f %f", &kernel[i * maskDim], &kernel[i * maskDim + 1], &kernel[i * maskDim + 2], &kernel[i * maskDim + 3], &kernel[i * maskDim + 4], &kernel[i * maskDim + 5], &kernel[i * maskDim + 6], &kernel[i * maskDim + 7], &kernel[i * maskDim + 8]) != maskDim)
            {
                printf("Wrong input. Use -h or --help for more information\n");
                return -1;
            }
        }
        fclose(kernelFile);

        // Copy to device
        loadKernel(kernel, maskDim);
        free(kernel);
    }
    break;

    // Gaussian
    case 8:
    {
        int mode = (int)strtol(argv[5], NULL, 10);
        if (mode != 0)
        {
            printf("Wrong command line input. Do not input gaussian data for non-gaussian matrices. Use -h or --help for more information\n");
            return -1;
        }
        int gaussLength = (int)strtol(argv[6], NULL, 10);
        float gaussSigma = (float)strtof(argv[7], NULL);
        if (gaussLength < 3 || gaussLength > 15 || gaussLength % 2 == 0 || gaussSigma < 0.5 || gaussSigma > 5)
        {
            printf("Wrong Gaussian values:\nACCEPTED VALUES:\n\t 3 <= gaussLength (must be odd) <= 15\n\t 0.5 <= gaussSigma <= 5\nAborting...\n");
            return -1;
        }
        float *gaussKernel = (float *)malloc(gaussLength * gaussLength * sizeof(float));
        gaussianKernelCPU(gaussLength, gaussSigma, gaussKernel);
#if DEBUG
        printf("\nGaussLength: %d\nGaussSigma: %f\n", gaussLength, gaussSigma);
        printf("\nGaussian kernel:\n");
        for (int i = 0; i < gaussLength; i++)
        {
            for (int j = 0; j < gaussLength; j++)
            {
                printf("%f ", gaussKernel[i * gaussLength + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
        maskDim = gaussLength;
        loadKernel(gaussKernel, maskDim);
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
    const int newOutSDim = outScaleDim + maskDim - 1;
    const int pxCount = outScaleDim * outScaleDim * 3;
    const int pxCnt = newOutSDim * newOutSDim * 3;

    char *d_start, *d_cutout, *d_scale;
    cudaMalloc((void **)&d_start, pxCount * sizeof(char));
    cudaMalloc((void **)&d_scale, pxCnt * sizeof(char));
    cudaMalloc((void **)&d_cutout, dimZoom * 3 * dimZoom * sizeof(char));
    cudaMemcpy(d_start, img->data, pxCount * sizeof(char), cudaMemcpyHostToDevice);

    int neededThreads = dimZoom * dimZoom * 3;
    dim3 usedThreads = (neededThreads > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : neededThreads;

    // calcolo numero blocchi necessari
    dim3 usedBlocks = (neededThreads / prop.maxThreadsPerBlock) + 1;
    // controllo numero blocchi utilizzabili
    if (usedBlocks.x > prop.maxGridSize[0])
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }
#if DEBUG
    printf("END OF CPU INSTRUCTIONS\n\n");
#endif

    // Get the cutout of the image
#if DEBUG
    printf("\nCutout:\nUsed Threads: %d - Used Blocks: %d\n", usedThreads.x, usedBlocks.x);
#endif
    getCutout<<<usedBlocks, usedThreads>>>(d_start, d_cutout, pointY, pointX, img->width, dimZoom, dimZoom);
    cudaDeviceSynchronize();
    destroyPPM(img);
    cudaFree(d_start);
#if DEBUG
    RGBImage *imgCut = createPPM(dimZoom, dimZoom);
    cudaMemcpy(imgCut->data, d_cutout, dimZoom * dimZoom * 3 * sizeof(char), cudaMemcpyDeviceToHost);
    writePPM("DEBUG_cutout.ppm", imgCut);
    destroyPPM(imgCut);
    printf("\tDone Scaling\n");
#endif

    // Zooming
    usedThreads = (pxCount > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : pxCount;
    usedBlocks = (pxCount / prop.maxThreadsPerBlock) + 1;
    if (usedBlocks.x > prop.maxGridSize[0])
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }
#if DEBUG
    printf("\nZooming:\nUsed Threads: %d - Used Blocks: %d\n", usedThreads.x, usedBlocks.x);
#endif
    scaleGPU<<<usedBlocks, usedThreads>>>(d_cutout, d_scale, dimZoom, outScaleDim, newOutSDim, maskDim / 2);
    cudaDeviceSynchronize();
    cudaFree(d_cutout);
#if DEBUG
    RGBImage *imgScale = createPPM(newOutSDim, newOutSDim);
    cudaMemcpy(imgScale->data, d_scale, pxCnt * sizeof(char), cudaMemcpyDeviceToHost);
    writePPM("DEBUG_scaled.ppm", imgScale);
    destroyPPM(imgScale);
    printf("\tDone Zooming\n");
#endif

    // Convolution
    char *d_out;
    cudaMalloc((void **)&d_out, pxCount * sizeof(char));
    const int elementsPerTile = ((int)sqrt(usedThreads.x) - (maskDim - 1));
    const int numTilesPerBlock = getNumTilesPerBlock(elementsPerTile, outScaleDim);
    if (numTilesPerBlock == 0 || ((pxCount % numTilesPerBlock) != 0))
    {
        printf("Error: Cannot divide the image into tiles\nThe final image dimension must be a multiple of < %d for the system to use the tiling approach\nSwitching to naive solution\n", elementsPerTile);
#if DEBUG
        printf("\nBasic Convolution:\nUsed Threads: %d - Used Blocks: %d\n", usedThreads.x, usedBlocks.x);
#endif
        basicConvGPU<<<usedBlocks, usedThreads>>>(d_scale, d_out, outScaleDim * 3, maskDim, newOutSDim * 3);
    }
    else
    {
        const int biggerTilesPerBlock = numTilesPerBlock + maskDim - 1;
        usedThreads.x = biggerTilesPerBlock;
        usedThreads.y = biggerTilesPerBlock;
        const int res = outScaleDim / numTilesPerBlock * 3;
        usedBlocks.x = res;
        usedBlocks.y = res;
        const int sharedMem = biggerTilesPerBlock * biggerTilesPerBlock * sizeof(char);

        if (sharedMem > prop.sharedMemPerBlock)
        {
            printf("Error: shared memory too small for the operation\n");
            return -1;
        }

#if DEBUG
        printf("\nConvolution:\nUsed Threads: %d - Used Blocks: %d - Shared Memory: %d - Tiles per block: %d - Wider Tiles's side: %d\n", usedThreads.x*usedThreads.y, usedBlocks.x*usedBlocks.y, sharedMem, numTilesPerBlock, biggerTilesPerBlock);
#endif
        convGPU<<<usedBlocks, usedThreads, sharedMem>>>(d_scale, d_out, outScaleDim * 3, maskDim, newOutSDim * 3, numTilesPerBlock, biggerTilesPerBlock);
    }
    checkCudaErrors(cudaDeviceSynchronize());
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