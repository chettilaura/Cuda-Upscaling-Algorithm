# Cuda_Upscaling Project

# TEAM
- s294767 Marchetti Laura
- s298846 Di Gruttola Giardino Nicola
- s303395 Durando Luca

# Description
This project is a CUDA implementation of Upscaling algorithm. It is a technique used to resize an image while reducing the aliasing effect in computer graphics. It is based on the idea of repeating each pixel N-times and then convoluting the resulting matrix with a kernel previously loaded. The result is a smoother and bigger image with less aliasing artifacts.

# How to compile
The project is developed in C++ and CUDA. To compile it, you need to have installed the CUDA Toolkit.

This is the command to compile the project:

> nvcc main.cu cpulib/cpu.cpp gpulib/gpu.cu imglib/img.cpp 
