main: main.cu cpulib/cpu.cpp gpulib/gpu.cu imglib/img.cpp
	nvcc -o upsCu main.cu cpulib/cpu.cpp gpulib/gpu.cu imglib/img.cpp