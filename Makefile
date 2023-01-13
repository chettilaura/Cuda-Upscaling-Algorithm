main: main.cu cpulib/cpulib.c gpulib/gpulib.cu imglib/img.cpp
	nvcc -o upsCu main.cu cpulib/cpu.cpp gpulib/gpu.cu imglib/img.cpp