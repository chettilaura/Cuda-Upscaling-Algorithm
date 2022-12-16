#include "imglib/img.h"
#include <stdio.h>

int main(int argc, char **argv){
    int dimX= 250;      //Coordinata centro X maschera per selezione
    int dimY= 200;      //Coordinata centro Y maschera per selezione
    int dimZoom = 100;    //Dimensione della maschera per selezione

    //Inizializzazione
    if(argc != 5){
        printf("Uso: %s inputFile.? DimX DimY dimZoom", argv[0]);
        return -1;
    }

    GrayImage* img = readPGM(argv[1]);
    if (img == NULL) {
        printf("Errore nell'apertura dell'immagine");
        return -1;
    }

    //Check per Y
    if(dimY > img->height || dimY < 0){
        printf("Errore: Y fuori dai bordi dell'immagine");
        return -1;
    }
    {
        const int boardYup = dimY - dimZoom/2;
        const int boardYdown = dimY + dimZoom/2;
        if (boardYdown > img->height || boardYup < 0){
            printf("Errore: Maschera Y fuori dai bordi dell'immagine");
            return -1;
        }

    }

    //Check per X
    if(dimX > img->width || dimX < 0){
        printf("Errore: X fuori dai bordi dell'immagine");
        return -1;
    }
    {
        const int boardXup = dimX + dimZoom/2;
        const int boardXdown = dimX - dimZoom/2;
        if (boardXup > img->width || boardXdown < 0){
            printf("Errore: Maschera X fuori dai bordi dell'immagine");
            return -1;
        }

    }


    //Selezione
    GrayImage* img2 = createPGM(dimZoom, dimZoom);
    GrayImage* imgLeft = createPGM(dimZoom, dimZoom);
    GrayImage* imgRight = createPGM(dimZoom, dimZoom);
    GrayImage* imgUp = createPGM(dimZoom, dimZoom);
    GrayImage* imgDown = createPGM(dimZoom, dimZoom);

    const int pointX =  dimX - dimZoom/2;
    const int pointY =  dimY - dimZoom/2;


    for(int i = 0; i < dimZoom; i++){
        for(int j = 0; j < dimZoom; j++){
            img2->data[i * dimZoom + j] = img->data[pointX + j + (pointY + i) * img->width];
        }
    }

    //Stampa
    writePGM("output.pgm", img2);
    destroyPGM(img);
    destroyPGM(img2);
}