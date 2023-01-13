#include "cpu.h"
#include "../standlib/stdCu.h"

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


void gaussianKernelCPU(const int gaussLength, const float gaussSigma, float *outkernel)
{
    float *kernel = (float *)malloc(gaussLength * gaussLength * sizeof(float));
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
    memcpy(outkernel, kernel, gaussLength * gaussLength * sizeof(float));
    free(kernel);
}



void zero_order_zoomingCPU(int *img, int *zoomed_out, int dimZoomX, int dimZoomY, int x, int y, int width, int height)
{

    int x_range_max;
    int y_range_max;

    int *zoomed = (int *)malloc(dimZoomX * dimZoomY * sizeof(int));
    zoomed_out = (int *)malloc(width * height * sizeof(int));

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