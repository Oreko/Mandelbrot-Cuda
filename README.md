# Mandelbrot-Cuda
Generate mandelbrot images and gifs using cuda

This is a simple project written to relearn basic cuda programming.

To use the gif generator, you need to have ffpmeg and python installed.
Basic usage is
```
gif.py -0.4997425 0.523556 4
^ Name of script
       ^ x coordinate (real)
                  ^ y coordinate (imaginary)
                           ^ Threshold (limit norm)
```
