#include "imglib/img.h"
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

/*
    includere più di un vettore img in un kernel non è possibile, crasha il programma
*/
__global__ void superSampler(int *d_imgCenter, int *d_imgLeft, int *d_imgRight, int *d_imgUp, int *d_imgDown, int *d_imgConv, int dimZoom)
{
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int j = threadIdx.y + blockIdx.y * blockDim.y;
    /* if (i < dimZoom && j < dimZoom)
     {
         int sum = 0;
         //sum += d_imgCenter[i * dimZoom + j] * 4;
         sum += d_imgLeft[j * dimZoom + i] * -1;
         sum += d_imgRight[j * dimZoom + i] * -1;
         sum += d_imgUp[j * dimZoom + i] * -1;
         sum += d_imgDown[j * dimZoom + i] * -1;
         //d_imgConv[i * dimZoom + j] = sum;
     }*/
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

int main(int argc, char **argv)
{
    int dimX = 0;    // Coordinata centro X maschera per selezione
    int dimY = 0;    // Coordinata centro Y maschera per selezione
    int dimZoom = 0; // Dimensione della maschera per selezione

    // Inizializzazione
    if (argc != 5)
    {
        printf("Uso: %s inputFile.? DimX DimY dimZoom\n", argv[0]);
        return -1;
    }

    dimX = (int)strtol(argv[2], NULL, 10);
    dimY = (int)strtol(argv[3], NULL, 10);
    dimZoom = (int)strtol(argv[4], NULL, 10);

    printf("DimX: %d, DimY: %d, dimZoom: %d\n", dimX, dimY, dimZoom);

    GrayImage *img = readPGM(argv[1]);
    if (img == NULL)
    {
        printf("Errore nell'apertura dell'immagine");
        return -1;
    }

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
    GrayImage *imgCenter = createPGM(dimZoom, dimZoom);
    GrayImage *imgLeft = createPGM(dimZoom, dimZoom);
    GrayImage *imgRight = createPGM(dimZoom, dimZoom);
    GrayImage *imgUp = createPGM(dimZoom, dimZoom);
    GrayImage *imgDown = createPGM(dimZoom, dimZoom);

    const int pointX = dimX - dimZoom / 2;
    const int pointY = dimY - dimZoom / 2;

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
            imgCenter->data[i * dimZoom + j] = img->data[pointX + j + (pointY + i) * img->width];

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
            imgLeft->data[i * dimZoom + j] = img->data[pointX + j - 1 + (pointY + i) * img->width];

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
            imgRight->data[i * dimZoom + j] = img->data[pointX + j + 1 + (pointY + i) * img->width];

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
            imgUp->data[i * dimZoom + j] = img->data[pointX + j + (pointY + i - 1) * img->width];

    for (int i = 0; i < dimZoom; i++)
        for (int j = 0; j < dimZoom; j++)
            imgDown->data[i * dimZoom + j] = img->data[pointX + j + (pointY + i + 1) * img->width];

    int *d_imgCenter, *d_imgLeft, *d_imgRight, *d_imgUp, *d_imgDown, *d_imgConv;
    cudaMalloc((void **)&d_imgCenter, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgLeft, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgRight, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgUp, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgDown, dimZoom * dimZoom * sizeof(int));
    cudaMalloc((void **)&d_imgConv, dimZoom * dimZoom * sizeof(int));

    cudaMemcpy(d_imgCenter, imgCenter->data, dimZoom * dimZoom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgLeft, imgLeft->data, dimZoom * dimZoom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgRight, imgRight->data, dimZoom * dimZoom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgUp, imgUp->data, dimZoom * dimZoom * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imgDown, imgDown->data, dimZoom * dimZoom * sizeof(int), cudaMemcpyHostToDevice);

    // Convoluzione
    GrayImage *imgConv = createPGM(dimZoom, dimZoom);
    int thread = dimZoom * dimZoom;
    superSampler<<<1, 1>>>(d_imgCenter, d_imgLeft, d_imgRight, d_imgUp, d_imgDown, d_imgConv, dimZoom);
    cudaDeviceSynchronize();
    cudaMemcpy(imgConv->data, d_imgConv, dimZoom * dimZoom * sizeof(int), cudaMemcpyDeviceToHost);
    printf("well done");
    // Stampa

    writePGM("output.pgm", imgConv);
    destroyPGM(img);
    destroyPGM(imgCenter);
    destroyPGM(imgLeft);
    destroyPGM(imgRight);
    destroyPGM(imgUp);
    destroyPGM(imgDown);
    destroyPGM(imgConv);
    cudaFree(d_imgCenter);
    cudaFree(d_imgLeft);
    cudaFree(d_imgRight);
    cudaFree(d_imgUp);
    cudaFree(d_imgDown);
    cudaFree(d_imgConv);

    return 0;
}