# Cuda_SuperSampling Project

# TEAM
- s294767 Marchetti Laura
- s298846 Di Gruttola Giardino Nicola
- s303395 Durando Luca

# Description
This project is a CUDA implementation of Super Sampling algorithm. It is a technique used to reduce the aliasing effect in computer graphics. It is based on the idea of sampling the image multiple times with offsets and then averaging the results. The result is a smoother image with less aliasing artifacts.
It is related to aerospace research, where it is used to reduce the noise in radar images.

# How to compile
The project is developed in C++ and CUDA. To compile it, you need to have installed the CUDA Toolkit.

This is the command to compile the project:

> nvcc main.cu cpulib/cpu.cpp gpulib/gpu.cu imglib/img.cpp 
