#include "imglib/img.h"
#include "cpulib/cpu.h"
#include "gpulib/gpu.cuh"
#include "standlib/stdCu.h"

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
            "\t -c[v][f] --custom[v][f]: Generate a custom matrix from the file passed as an argument in the Matrix generation's parameters\n"
            "\t -g[v][f] --gauss[v][f]: Generate a gaussian matrix\n"
            "\t Optional: v character to allow verbose mode and print debug information\n"
            "\t Optional: f character to force the use of the global memory\n\n"
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
    bool forceGlobal = arg1.find('f') != std::string::npos;

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

    const int cutOutCenterX = (int)strtol(argv[3], NULL, 10); // X coordinate of the center of the selection zone
    const int cutOutCenterY = (int)strtol(argv[4], NULL, 10); // Y coordinate of the center of the selection zone
    const int cutOutWidth = (int)strtol(argv[5], NULL, 10);   // Length of the side of the selection mask
    const int cutOutHeight = (int)strtol(argv[6], NULL, 10);  // Length of the side of the selection mask
    const int zoomLevel = (int)strtol(argv[7], NULL, 10);     // Zoom level

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
        printf("Error: cutOutCenterY outside image boundaries\n");
        return -1;
    }
    if ((cutOutCenterY + cutOutHeight / 2) > img->height - 1 || pointY < 1)
    {
        printf("Error: Y mask outside image boundaries\n");
        return -1;
    }

    // X boundaries check and mask check
    const int pointX = cutOutCenterX - cutOutWidth / 2;
    if (cutOutCenterX > img->width || cutOutCenterX < 0)
    {
        printf("Error: cutOutCenterX outside image boundaries\n");
        return -1;
    }
    if ((cutOutCenterX + cutOutWidth / 2) > img->width - 1 || pointX < 1)
    {
        printf("Error: X mask outside image boundaries\n");
        return -1;
    }

    const int widthImgIn = img->width;
    const int heightImgIn = img->height;
    const int widthImgOut = cutOutWidth * zoomLevel;
    const int heightImgOut = cutOutHeight * zoomLevel;
    const int outPx = widthImgOut * heightImgOut * 3;

    unsigned char *d_start, *d_out;
    cudaMalloc((void **)&d_start, img->height * img->width * 3 * sizeof(char));
    cudaMemcpy(d_start, img->data, img->height * img->width * 3 * sizeof(char), cudaMemcpyHostToDevice);
    destroyPPM(img);
    cudaMalloc((void **)&d_out, outPx * sizeof(char));

    if (verbose)
        printf("Image loaded\n"
               "Image width: %d px\n"
               "Image height: %d px\n"
               "Image size: %d bytes\n"
               "Output width: %d px\n"
               "Output height: %d px\n"
               "Output size: %d bytes\n",
               widthImgIn, img->height, img->height * img->width * 3, widthImgOut, heightImgOut, outPx);

    int widthTile = ((int)sqrt(prop.maxThreadsPerBlock) - (maskDim - 1));
    int heightTile = widthTile;
    if (checkTiling(widthImgOut, heightImgOut, &widthTile, &heightTile) && !forceGlobal)
    // TRUE: Tiling approach doable
    {
        // Number of threads per block
        dim3 usedThreads = dim3(widthTile + maskDim - 1, heightTile + maskDim - 1, 1);
        // Number of blocks
        dim3 usedBlocks = dim3(widthImgOut / widthTile, heightImgOut / heightTile, 3);
        if(usedBlocks.x > prop.maxGridSize[0] || usedBlocks.y > prop.maxGridSize[1] || usedBlocks.z > prop.maxGridSize[2])
        {
            printf("Error: Blocks overflow\n");
            return -1;
        }
        
        // Bytes of shared memory per block
        int sharedMemSize = (widthTile + maskDim - 1) * (heightTile + maskDim - 1) * sizeof(char);
        if (sharedMemSize > prop.sharedMemPerBlock)
        {
            printf("Error: Shared memory overflow\n");
            return -1;
        }

        if (verbose)
            printf("Tiling approach executing...\n"
                   "Threads per block: %d, %d, %d\n"
                   "Blocks: %d, %d, %d\n"
                   "Shared memory size: %d bytes\n"
                   "Launching kernel...\n"
                   "Parameters:\n"
                   "",
                   usedThreads.x, usedThreads.y, usedThreads.z, usedBlocks.x, usedBlocks.y, usedBlocks.z, sharedMemSize);

        tilingCudaUpscaling<<<usedBlocks, usedThreads, sharedMemSize>>>(d_start, d_out, widthImgIn, heightImgIn, widthImgOut, heightImgOut, widthTile, heightTile, maskDim, (pointX - (maskDim / 2 / zoomLevel)), (pointY - (maskDim / 2 / zoomLevel)), zoomLevel);
    }
    else
    // FALSE: Global memory approach is used
    {
        // Number of threads per block
        dim3 usedThreads = (outPx > prop.maxThreadsPerBlock) ? prop.maxThreadsPerBlock : outPx;
        // Number of blocks
        dim3 usedBlocks = (outPx / prop.maxThreadsPerBlock) + 1;
        if (usedBlocks.x > prop.maxGridSize[0])
        {
            printf("%s\n", cudaGetErrorString(err));
            return -1;
        }

        if (verbose)
            printf("Global memory approach executing...\n"
                   "Threads per block: %d\n"
                   "Blocks: %d\n"
                   "Launching kernel...\n",
                   usedThreads.x, usedBlocks.x);

        globalCudaUpscaling<<<usedBlocks, usedThreads>>>(d_start, d_out, widthImgIn, heightImgIn, widthImgOut, heightImgOut, maskDim, (pointX - (maskDim / 2 / zoomLevel)), (pointY - (maskDim / 2 / zoomLevel)), zoomLevel);
    }
    cudaDeviceSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("%s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back to host
    RGBImage *out = createPPM(widthImgOut, heightImgOut);
    cudaMemcpy(out->data, d_out, outPx * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(d_start);
    cudaFree(d_out);

    // Write output file
    writePPM("output.ppm", out);
    destroyPPM(out);

    if (verbose)
        printf("END OF THE PROGRAM\n\n");

    return 0;
}