// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// PNG include
#include <png.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128 // number of threads in each block
#endif

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 4096 // Width and height of output image
#endif

#define DATASET_SIZE (IMAGE_SIZE * IMAGE_SIZE)


typedef png_color* colorp;


uint8_t hEscapeNumber[IMAGE_SIZE * IMAGE_SIZE];
double  hComplexArray[2 * IMAGE_SIZE * IMAGE_SIZE];

void write_palette(colorp palette, uint16_t const palette_size)
{
    uint8_t const Ored   = 244;
    uint8_t const Ogreen = 172;
    uint8_t const Oblue  = 123;
    uint8_t const Bred   = 14;
    uint8_t const Bgreen = 59;
    uint8_t const Bblue  = 92;
    for(size_t i = 0; i < palette_size; i++)
    {
        double scale = (double)i / (double)palette_size;
        double Cred   = 0;
        double Cgreen = 0;
        double Cblue  = 0;
        if(i >= 64 && i < 192) // Orange to blue
        {
            scale  = (double)(i - 64) / (double)128;
            Cred   = (1.0 - scale) * Ored   + scale * Bred;
            Cgreen = (1.0 - scale) * Ogreen + scale * Bgreen;
            Cblue  = (1.0 - scale) * Oblue  + scale * Bblue;
        } else // Blue to orange
        {
            if (i < 64)
            {
                scale = (double)(i + 64) / (double)128;
            } else
            {
                scale = (double)(i - 192) / (double)128;
            }
            Cred   = (1.0 - scale) * Bred   + scale * Ored;
            Cgreen = (1.0 - scale) * Bgreen + scale * Ogreen;
            Cblue  = (1.0 - scale) * Bblue  + scale * Oblue;
        }
        colorp col = &palette[i];
        col->red   = (uint8_t) Cred;
        col->green = (uint8_t) Cgreen;
        col->blue  = (uint8_t) Cblue;
    }
}

int mandelbrot_to_png(char const * const fileName, uint8_t const * const escapeArray)
{
    FILE *fp = fopen(fileName, "wb");
    if (fp == NULL)
    {
        return (1);
    }
    png_structp png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL,
                                                   NULL, NULL);
    if (png_ptr == NULL)
    {
        fclose(fp);
        return (1);
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, NULL);
        return (1);
    }
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, IMAGE_SIZE, IMAGE_SIZE,
                 8, PNG_COLOR_TYPE_PALETTE, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,  PNG_FILTER_TYPE_DEFAULT);

    uint16_t const palette_size = 256;
    assert(palette_size <= PNG_MAX_PALETTE_LENGTH);

    png_colorp palette = (png_colorp)png_malloc(png_ptr, palette_size * sizeof (png_color));
    write_palette(palette, palette_size);
    png_set_PLTE(png_ptr, info_ptr, palette, palette_size);
    png_write_info(png_ptr, info_ptr);
    for (size_t i = 0; i < IMAGE_SIZE; i++)
    {
        png_write_row(png_ptr, (png_const_bytep)&escapeArray[i*IMAGE_SIZE]);
    }
    png_write_flush(png_ptr);
    png_write_end(png_ptr, NULL);
    png_free(png_ptr, palette);
    palette=NULL;
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return (0);
}


__global__ void generate_complex_array (double const centerReal, double const centerImaginary,
                                        double const scale,      double * const complexArray)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if( gid < DATASET_SIZE )
    {
        int row               = gid / IMAGE_SIZE;
        int column            = gid % IMAGE_SIZE;
        complexArray[2*gid]   = scale * (column - IMAGE_SIZE/2 ) + centerReal;      // Real
        complexArray[2*gid+1] = scale * (IMAGE_SIZE/2 - row ) + centerImaginary; // Imaginary
    }
}

__global__ void brot_escape( uint8_t const        threshold,    uint32_t const  limit,
                             double const * const complexArray, uint8_t * const escapeNumber)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    if( gid < DATASET_SIZE )
    {
        double const x0 = complexArray[2 * gid];
        double const y0 = complexArray[2 * gid + 1];
        double x2 = 0;
        double y2 = 0;
        double x = 0;
        double y = 0;
        uint16_t iteration = 0;
        while (x2 + y2 < threshold && iteration < limit)
        {
            y = 2 * x * y + y0;
            x = x2 - y2 + x0;
            x2 = x * x;
            y2 = y * y;
            iteration++;
        }
        double palette_index = pow(((double)iteration / (double)limit * 255.0), 1.5); // 2^8 - 1 for 8 bit palette
        escapeNumber[gid] = (uint8_t) palette_index % 255;
    }
}

int
main(int argc, char *argv[])
{
    // int dev = findCudaDevice(argc, (const char **)argv);
    int returnValue = 0;
    if(argc >= 7)
    {
        double      const centerReal      = strtod(argv[1], NULL);
        double      const centerImaginary = strtod(argv[2], NULL);
        uint8_t     const threshold       = atoi(argv[3]);
        uint32_t    const limit           = atoi(argv[4]);
        double      const scale           = strtod(argv[5], NULL);
        char const* const filename        = argv[6];

        // printf("Center at %lf + %lfi, threshold: %d, limit: %d, scale: %lf\n", centerReal, centerImaginary, threshold, limit, scale);

        // allocate device memory:
        double  *dComplexArray;
        uint8_t *dEscapeNumber;

        cudaError_t status;
        status = cudaMalloc( (void **)(&dComplexArray), sizeof(hComplexArray) );
        checkCudaErrors(status);

        status = cudaMalloc( (void **)(&dEscapeNumber), sizeof(hEscapeNumber) );
        checkCudaErrors(status);


        // copy host memory to the device:
        status = cudaMemcpy( dEscapeNumber, hEscapeNumber, sizeof(hEscapeNumber), cudaMemcpyHostToDevice );
        checkCudaErrors(status);

        // setup the execution parameters:
        dim3 grid(DATASET_SIZE / THREADS_PER_BLOCK, 1, 1 );
        dim3 threads(THREADS_PER_BLOCK, 1, 1 );

        // execute the kernel:
        generate_complex_array<<< grid, threads >>>(centerReal, centerImaginary,
                                                    scale,      dComplexArray);
        status = cudaMemcpy( hComplexArray, dComplexArray, sizeof(hComplexArray), cudaMemcpyDeviceToHost );
        checkCudaErrors(status);

        brot_escape<<< grid, threads >>>(threshold,  limit,
                                         dComplexArray, dEscapeNumber);

        // copy result from the device to the host:
        status = cudaMemcpy( hEscapeNumber, dEscapeNumber, sizeof(hEscapeNumber), cudaMemcpyDeviceToHost );
        checkCudaErrors(status);

        status = cudaDeviceSynchronize();
        checkCudaErrors(status);

        // clean up:
        status = cudaFree( dComplexArray );
        checkCudaErrors(status);

        status = cudaFree( dEscapeNumber );
        checkCudaErrors(status);

        return mandelbrot_to_png(filename, hEscapeNumber);
    } else
    {
        fprintf(stderr, "Arguments: center real, center imaginary, threshold, limit, scale, filename\n");
        returnValue = 1;
    }
    return returnValue;
}
