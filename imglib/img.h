#pragma once
//
// Created by Corrado on 11/17/2022.
//




typedef struct {
    int width, height;
    // data is a vector of dimension width*height*channels (RGB has 3 channels)
    // pixels are stored as RGB one after the other ( r_0,g_0,b_0,r_1,g_1,b_1...) 
    unsigned char * data; 
} RGBImage;

typedef struct {
    int width, height;
    // data is a vector of dimension width*height*channels (GrayImage has 1 channels) 
    // pixels are stored one after the other ( gray_0, gray_1, gray_2...) 
    unsigned char* data; 
} GrayImage;

RGBImage* readPPM(const char* filename);
GrayImage* readPGM(const char* filename);

GrayImage* createPGM(int width, int height);
RGBImage* createPPM(int width, int height);

void destroyPGM(GrayImage* img);
void destroyPPM(RGBImage* img);

void writePGM(const char* filename, GrayImage* img);
void writePPM(const char* filename, RGBImage* img);

