//
// Created by Corrado on 11/17/2022.
//
#define MAX_COLOR_DEPTH 255
#include "img.h"
#include <cstdio>
#include <stdlib.h>


RGBImage* readPPM(const char* filename) {
    char buff[16];
    RGBImage* img;
    FILE* fp;
    int c, rgb_comp_color;
    int w, h;

    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }


    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n');
        c = getc(fp);
    }

    ungetc(c, fp);

    if (fscanf(fp, "%d %d", &w, &h) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }


    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
        exit(1);
    }

    if (rgb_comp_color != MAX_COLOR_DEPTH) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    img = createPPM(w, h);

    while (fgetc(fp) != '\n');

    if (fread(img->data, 3 * img->width, img->height, fp) != img->height) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(const char* filename, RGBImage* img)
{
    FILE* fp;

    fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }


    fprintf(fp, "P6\n");

    fprintf(fp, "%d %d\n", img->width, img->height);

    fprintf(fp, "%d\n", MAX_COLOR_DEPTH);

    fwrite(img->data, 3 * img->width, img->height, fp);
    fclose(fp);
}

RGBImage* createPPM(int width, int height) {

    RGBImage* img;

    img = (RGBImage*)malloc(sizeof(RGBImage));

    if (!img) {
        fprintf(stderr, "malloc failure\n");
        exit(1);
    }

    img->width = width;
    img->height = height;
    img->data = (unsigned char*)malloc(img->width * img->height * 3 * sizeof(unsigned char));

    if (!img->data) {
        fprintf(stderr, "malloc failure\n");
        exit(1);
    }
    return img;
}

void destroyPPM(RGBImage* img) {

    free(img->data);
    free(img);
}
