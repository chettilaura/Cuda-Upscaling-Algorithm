#include "imglib/img.h"
#include <stdio.h>

__global__ void superSampler(int *d_imgCenter, int *d_imgLeft, int *d_imgRight, int *d_imgUp, int *d_imgDown, int *d_imgConv, int dimZoom){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < dimZoom && j < dimZoom)
    {
        int sum = 0;
        sum += d_imgCenter[i * dimZoom + j] * 4;
        sum += d_imgLeft[i * dimZoom + j] * -1;
        sum += d_imgRight[i * dimZoom + j] * -1;
        sum += d_imgUp[i * dimZoom + j] * -1;
        sum += d_imgDown[i * dimZoom + j] * -1;
        d_imgConv[i * dimZoom + j] = sum;
    }
}

int main(int argc, char **argv)
{
    int dimX = 250;    // Coordinata centro X maschera per selezione
    int dimY = 200;    // Coordinata centro Y maschera per selezione
    int dimZoom = 100; // Dimensione della maschera per selezione

    // Inizializzazione
    if (argc != 5)
    {
        printf("Uso: %s inputFile.? DimX DimY dimZoom", argv[0]);
        return -1;
    }

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
    superSampler<<<1, 10000>>>(d_imgCenter, d_imgLeft, d_imgRight, d_imgUp, d_imgDown, d_imgConv, dimZoom);

    cudaMemcpy(imgConv->data, d_imgConv, dimZoom * dimZoom * sizeof(int), cudaMemcpyDeviceToHost);
    // Stampa
    writePGM("output.pgm", imgConv);
    destroyPGM(img);
    destroyPGM(imgCenter);
}