#ifndef _INC_IMG // include guard for 3rd party interop
#define _INC_IMG

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

RGBImage* readPPM(const char* filename);

RGBImage* createPPM(int width, int height);

void destroyPPM(RGBImage* img);

void writePPM(const char* filename, RGBImage* img);

#endif //IMG_H

