#include "imglib/img.h"
#include "cpulib/cpu.h"
#include "gpulib/gpu.cuh"
#include "standlib/stdCu.h"

#define N 9

bool replace(std::string &str, char c, char r)
{
    bool found = false;
    int len = str.length();
    for (int i = 0; i < len; i++)
    {
        printf("%c\n", str[i]);
        if (str[i] == c)
        {
            str[i] = r;
            printf("std::string: %s\n", str.c_str());
            found = true;
        }
    }
    return found;
}

int main(int argc, char **argv)
{
    // Go to help
    if (argc == 1 || (argc == 2 && (argv[1] == std::string("-h") || argv[1] == std::string("--help"))))
    {
        printf(
            "CUDA Upscale\n\n"
            "WARNING: This program is made only for educational purposes and is not intended to be used in production.\n\n"
            "Usage:\n\n"
            "    %s [Filtering Matrix generation's commands] inputFile.ppm cutOutCenterX cutOutCenterY cutOutWidth cutOutHeight zoomLevel [Matrix generation's parameters]\n\n"
            "  - Filtering Matrix generation's commands\n"
            "\t -c[v] --custom[v]: Generate a custom matrix from the file passed as an argument in the Matrix generation's parameters\n"
            "\t -g[v] --gauss[v]: Generate a gaussian matrix\n"
            "\t Optional: v character to allow verbose mode and print debug information\n\n"
            "  - inputFile.ppm: A valid .ppm P6 input image\n"
            "  - cutOutCenterX: X coordinate of the center of the selection zone\n"
            "  - cutOutCenterY: Y coordinate of the center of the selection zone\n"
            "  - cutOutWidth: Length of the side of the selection\n\n"
            "  - cutOutHeight: Length of the side of the selection\n\n"
            "  - zoomLevel: Zoom level of the output image, must be a INT value from 1 to 32\n"
            "               If 1 is inserted, only the convolution will be performed\n\n"
            "  - Matrix generation's parameters\n"
            "\t GaussLength: must be an odd value from 3 to 15 sides included\n"
            "\t GaussSigma: must be a value from 0.5 to 5 sides included\n"
            "\t InputKernelFile.txt: formatted as such\n\n"
            "\t\t\tmatrixSide'sLength (must be odd)\n"
            "\t\t\tFirstElement SecondElement ...\n"
            "\t\t\tRowElement ...\n"
            "\t\t\t...\n",
            argv[0]);
    }

    if (argc < 9 || argc > 10)
    {
        printf("Wrong command line input. Use -h or --help for more information\n");
        return -1;
    }

    // Check if verbose mode is enabled
    std::string arg1 = argv[1];
    bool verbose = arg1.find('v') != std::string::npos;

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

    if (verbose)
        printf("Number of CUDA devices: %d\n", nDevices);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Mask length
    char maskDim = 0;

    // Check mode and load kernel
    if (verbose)
        printf("Loading kernel...\n");

    // Custom Kernel from file
    if (argc == 9 && ((arg1.rfind("-c", 0) == 0) || (arg1.rfind("--custom", 0) == 0)))
    {
        FILE *kernelFile = fopen(argv[8], "r");
        if (kernelFile == NULL)
        {
            printf("Error opening file %s\n", argv[8]);
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
            else if (verbose)
                printf("%f %f %f %f %f %f %f %f %f \n", kernel[i * maskDim], kernel[i * maskDim + 1], kernel[i * maskDim + 2], kernel[i * maskDim + 3], kernel[i * maskDim + 4], kernel[i * maskDim + 5], kernel[i * maskDim + 6], kernel[i * maskDim + 7], kernel[i * maskDim + 8]);
        }
        fclose(kernelFile);

        // Copy to device
        loadKernel(kernel, maskDim);
        free(kernel);
    }
    // Gaussian kernel
    else if (argc == 10 && ((arg1.rfind("-g", 0) == 0) || (arg1.rfind("--gauss", 0) == 0)))
    {
        int gaussLength = (int)strtol(argv[8], NULL, 10);
        float gaussSigma = (float)strtof(argv[9], NULL);
        if (gaussLength < 3 || gaussLength > 15 || gaussLength % 2 == 0 || gaussSigma < 0.5 || gaussSigma > 5)
        {
            printf("Wrong Gaussian values:\nACCEPTED VALUES:\n\t 3 <= gaussLength (must be odd) <= 15\n\t 0.5 <= gaussSigma <= 5\nAborting...\n");
            return -1;
        }
        float *gaussKernel = (float *)malloc(gaussLength * gaussLength * sizeof(float));
        gaussianKernelCPU(gaussLength, gaussSigma, gaussKernel);
        if (verbose)
        {
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
        }
        maskDim = gaussLength;
        loadKernel(gaussKernel, maskDim);
        free(gaussKernel);
    }
    // Display Error
    else
    {
        printf("Wrong command line input. Use -h or --help for more information\n");
        printf("arg1: %s, argc: %d\n", arg1.c_str(), argc);
        return -1;
    }
    if (verbose)
        printf("Kernel loaded\n"
               "Mask dimension: %d\n"
               "Proceeding with checks for scaling...\n",
               maskDim);

    // fine switch case
    const int cutOutCenterX = (int)strtol(argv[3], NULL, 10); // X coordinate of the center of the selection zone
    const int cutOutCenterY = (int)strtol(argv[4], NULL, 10); // Y coordinate of the center of the selection zone
    const int cutOutWidth = (int)strtol(argv[5], NULL, 10);   // Length of the side of the selection mask
    const int cutOutHeight = (int)strtol(argv[6], NULL, 10);  // Length of the side of the selection mask
    const int zoomLevel = (int)strtol(argv[7], NULL, 10);

    // check cutOutWidth is even
    if (cutOutWidth % 2 != 0)
    {
        printf("Error: cutOutWidth must be even\n");
        return -1;
    }

    // check cutOutHeight is even
    if (cutOutHeight % 2 != 0)
    {
        printf("Error: cutOutHeight must be even\n");
        return -1;
    }

    if (zoomLevel < 1 || zoomLevel > 32)
    {
        printf("Error: zoomLevel must be between 1 and 32\n");
        return -1;
    }

    if (verbose)
        printf("cutOutCenterX: %d, cutOutCenterY: %d, cutOutWidth: %d, cutOutHeight: %d, zoomLevel: %d\n", cutOutCenterX, cutOutCenterY, cutOutWidth, cutOutHeight, zoomLevel);

    // Check input file ends with .ppm
    if (std::string(argv[2]).size() < 4 || std::string(argv[2]).substr(std::string(argv[2]).size() - 4) != ".ppm")
    {
        printf("Error: input file must be a .ppm file\n");
        return -1;
    }
    RGBImage *img = readPPM(argv[2]);

    // Y boundaries check and mask check
    const int pointY = cutOutCenterY - cutOutHeight / 2;
    if (cutOutCenterY > img->height || cutOutCenterY < 0)
    {
        printf("Error: cutOutCenterY outside image boundaries");
        return -1;
    }
    if ((cutOutCenterY + cutOutHeight / 2) > img->height - 1 || pointY < 1)
    {
        printf("Error: Y mask outside image boundaries");
        return -1;
    }

    // X boundaries check and mask check
    const int pointX = cutOutCenterX - cutOutWidth / 2;
    if (cutOutCenterX > img->width || cutOutCenterX < 0)
    {
        printf("Error: cutOutCenterX outside image boundaries");
        return -1;
    }
    if ((cutOutCenterX + cutOutWidth / 2) > img->width - 1 || pointX < 1)
    {
        printf("Error: X mask outside image boundaries");
        return -1;
    }

    const int outWidthDim = cutOutWidth * zoomLevel;
    const int outHeightDim = cutOutHeight * zoomLevel;
    const int scaleWidthDim = outWidthDim + maskDim - 1;
    const int scaleHeightDim = outHeightDim + maskDim - 1;
    const int outPx = outWidthDim * outHeightDim * 3;
    const int scalePx = scaleWidthDim * scaleHeightDim * 3;

    char *d_start, *d_scale;
    cudaMalloc((void **)&d_start, img->height * img->width * 3 * sizeof(char));
    cudaMalloc((void **)&d_scale, scalePx * sizeof(char));
    cudaMemcpy(d_start, img->data, img->height * img->width * 3 * sizeof(char), cudaMemcpyHostToDevice);

    // int neededThreads = 32 / zoomLevel; // numero di thread necessari per ogni byte di output
    dim3 usedThreads = (outPx > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : outPx;

    // calcolo numero blocchi necessari
    dim3 usedBlocks = (outPx / prop.maxThreadsPerBlock) + 1;
    // controllo numero blocchi utilizzabili
    if (usedBlocks.x > prop.maxGridSize[0])
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }

    // Scale image from d_start to d_scale
    if (verbose)
        printf("Scaling image...\n"
               "Used Threads: %d - Used Blocks: %d\n"
               "dimImgIn: %d - dimImgMid: %d - dimImgW: %d - dimImgOut: %d - offsetCut: %d - offsetScale: %d - stuffing: %d - limit: %d\n",
               usedThreads.x, usedBlocks.x, img->width * 3, cutOutWidth * 3, outWidthDim * 3, scaleWidthDim * 3, (pointY * img->width + pointX) * 3, (scaleWidthDim + 1) * 3 * (maskDim / 2), zoomLevel, outPx);

    scaleImage<<<usedBlocks, usedThreads>>>(d_start, d_scale, img->width * 3, cutOutWidth * 3, outWidthDim * 3, scaleWidthDim * 3, (pointY * img->width + pointX) * 3, (scaleWidthDim + 1) * 3 * (maskDim / 2), zoomLevel, outPx);
    cudaDeviceSynchronize();
    if (verbose)
    {
        printf("Scaled image created on disk\n");
        RGBImage *scaled = createPPM(scaleWidthDim, scaleHeightDim);
        cudaMemcpy(scaled->data, d_scale, scaleWidthDim * scaleHeightDim * 3 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaFree(d_scale);
        writePPM("VERB_scaled.ppm", scaled);
        destroyPPM(scaled);
    }

    // Convolution
    char *d_out;
    cudaMalloc((void **)&d_out, outPx * sizeof(char));
    const int elementsPerTile = ((int)sqrt(usedThreads.x) - (maskDim - 1));
    const int numTilesPerBlock = getNumTilesPerBlock(elementsPerTile, outWidthDim);
    if (numTilesPerBlock == 0 || ((outPx % numTilesPerBlock) != 0))
    {
        printf("Error: Cannot divide the image into tiles\nThe final image dimension must be a multiple of < %d for the system to use the tiling approach\nSwitching to naive solution\n", elementsPerTile);
        if (verbose)
            printf("\nBasic Convolution:\nUsed Threads: %d - Used Blocks: %d\n", usedThreads.x, usedBlocks.x);

        basicConvGPU<<<usedBlocks, usedThreads>>>(d_scale, d_out, scaleWidthDim * 3, outWidthDim * 3, maskDim);
    }
    else
    {
        const int biggerTilesPerBlock = numTilesPerBlock + maskDim - 1;
        usedThreads.x = biggerTilesPerBlock;
        usedThreads.y = biggerTilesPerBlock;
        const int res = outWidthDim / numTilesPerBlock * 3;
        usedBlocks.x = res;
        usedBlocks.y = res;
        const int sharedMem = biggerTilesPerBlock * biggerTilesPerBlock * sizeof(char);

        if (sharedMem > prop.sharedMemPerBlock)
        {
            printf("Error: shared memory too small for the operation\n");
            return -1;
        }

        if (verbose)
            printf("\nConvolution:\nUsed Threads: %d - Used Blocks: %d - Shared Memory: %d - Tiles per block: %d - Wider Tiles's side: %d\n", usedThreads.x * usedThreads.y, usedBlocks.x * usedBlocks.y, sharedMem, numTilesPerBlock, biggerTilesPerBlock);

        convGPU<<<usedBlocks, usedThreads, sharedMem>>>(d_scale, d_out, scaleWidthDim * 3, outWidthDim * 3, maskDim, biggerTilesPerBlock, numTilesPerBlock);
    }
    cudaDeviceSynchronize();
    if (verbose)
        printf("\tDone Convoluting\nEND OF GPU INSTRUCTIONS\n\n");

    RGBImage *imgOut = createPPM(outWidthDim, outHeightDim);
    cudaMemcpy(imgOut->data, d_out, outPx * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(d_scale);
    cudaFree(d_out);

    // Print output
    writePPM("output.ppm", imgOut);
    destroyPPM(imgOut);

    if (verbose)
        printf("END OF THE PROGRAM\n\n");

    return 0;
}