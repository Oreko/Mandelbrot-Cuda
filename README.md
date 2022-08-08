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

Similarly, to run the static image generator
```
./mandelbrot -0.4997425 0.523556 4 10000 out.png
^ Name of executable
             ^ x coordinate (real)
                        ^ y coordinate (imaginary)
                                 ^ Threshold (limit norm)
                                   ^ Limit (iteration limit)
                                         ^ Output Filename
```
![Simple Static](https://i.imgur.com/uFXB9Fm.jpg)