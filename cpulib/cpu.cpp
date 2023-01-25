#include "cpu.h"
#include <cmath>


void gaussianKernelCPU(const int gaussLength, const float gaussSigma, float *kernel)
{
    float sum = 0;
    for (int i = 0; i < gaussLength; i++)
    {
        for (int j = 0; j < gaussLength; j++)
        {
            kernel[i * gaussLength + j] = exp(-((i - gaussLength / 2) * (i - gaussLength / 2) + (j - gaussLength / 2) * (j - gaussLength / 2)) / (2 * gaussSigma * gaussSigma));
            sum += kernel[i * gaussLength + j];
        }
    }
    for (int i = 0; i < gaussLength; i++)
    {
        for (int j = 0; j < gaussLength; j++)
        {
            kernel[i * gaussLength + j] /= sum;
        }
    }
}

bool checkTiling(const int width, const int height, int *dimTilesX, int *dimTilesY)
{
    const int back = *dimTilesY * 2;

    for (; *dimTilesX > 0; (*dimTilesX)--)
        if (width % *dimTilesX == 0)
            for (*dimTilesY = back - *dimTilesX; *dimTilesY > 0; (*dimTilesY)--)
                if (height % *dimTilesY == 0)
                    return true;
    
    return false;
}


/* DEPRECATED FUNCTIONS */

void convCPU(char *input, char *output, char *kernel, const int width, const int heigth)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < heigth; j++)
        {
            int sum = 0;
            for (int k = 0; k < DIMKERNEL; k++)
            {
                for (int l = 0; l < DIMKERNEL; l++)
                {
                    sum += input[(i + k) + (j + l) * width] * kernel[k * DIMKERNEL + l];
                }
            }
            output[i + j * width] = sum;
        }
    }
}

void zero_order_zoomingCPU(unsigned char *img, unsigned char *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height, int outDim)
{

    int x_range_max;
    int y_range_max;

    int *zoomed = (int *)malloc(dimZoomX * 3 * dimZoomY * sizeof(int));

    /*if (x < 0 || y < 0 || x > width || y > height)
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
    }*/

    for (int i = 0; i < dimZoomY; i++)
        for (int j = 0; j < dimZoomX; j++)
        {
            zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3] = img[(x + j) * 3 + (y + i) * width * 3];
            zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3 + 1] = img[(x + j) * 3 + (y + i) * width * 3 + 1];
            zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3 + 2] = img[(x + j) * 3 + (y + i) * width * 3 + 2];
        }

    // Second version for CPU, works, better parallelizable.
    // It refers to the bigger matrix and exploit the rounding down to refers to the same byte to copy to different positions
    int stuffing = outDim / dimZoomX;
    for (int i = 0; i < outDim; i++)
        for (int j = 0; j < outDim; j++)
        {
            zoomed_out[i * outDim * 3 + j * 3] = zoomed[(i / stuffing + 1) * dimZoomX * 3 + (j / stuffing + 1) * 3];
            zoomed_out[i * outDim * 3 + j * 3 + 1] = zoomed[(i / stuffing + 1) * dimZoomX * 3 + (j / stuffing + 1) * 3 + 1];
            zoomed_out[i * outDim * 3 + j * 3 + 2] = zoomed[(i / stuffing + 1) * dimZoomX * 3 + (j / stuffing + 1) * 3 + 2];
        }
    free(zoomed);
    // First version, works but not parallelizable, it refers to the smaller matrix and copies the same values cnt times
    /*for (int i = 0; i < dimZoomY; i++)
        for (int j = 0; j < dimZoomX; j++)
        {
            for (int cnt = 0; cnt < stuffing_y; cnt++)
            {
                for (int cnt2 = 0; cnt2 < stuffing_y; cnt2++)
                {
                    zoomed_out[(i * stuffing_y + cnt) * outDim * 3 + (j * stuffing_y + cnt2) * 3] = zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3];
                    zoomed_out[(i * stuffing_y + cnt) * outDim * 3 + (j * stuffing_y + cnt2) * 3 + 1] = zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3 + 1];
                    zoomed_out[(i * stuffing_y + cnt) * outDim * 3 + (j * stuffing_y + cnt2) * 3 + 2] = zoomed[(i + 1) * dimZoomX * 3 + (j + 1) * 3 + 2];
                }
            }
        }*/
}
