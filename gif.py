#!/usr/bin/python3

from tempfile import TemporaryDirectory
import shutil
import os
import math
import sys



def main():
    if(len(sys.argv) != 4):
        print("Usage: {sys.argv[0]} x y threshold")
        return(1)

    cwd = os.getcwd()
    with TemporaryDirectory() as tempdirpath:
        for iteration in range(1, 1201):
            scale = 1.0 / ( 1.1 ** (2.0 * math.sqrt(iteration * 10 + 400)))
            limit = 500 + (1.00915 * math.tanh(0.002247 * iteration)) * (8000)
            args = [os.path.join(cwd, 'mandelbrot'), sys.argv[1], sys.argv[2],
                    sys.argv[3], str(limit), str(scale),
                    os.path.join(tempdirpath, f'{iteration:04d}.png')]
            cmd = ' '.join(args)
            os.system(cmd)

        os.chdir(tempdirpath)
        output = os.path.join(cwd, 'out.gif')
        cmd = 'ffmpeg -f image2 -framerate 30 -i %04d.png {output}'
        os.system(cmd)
        os.chdir(cwd)

    return(0)

if __name__ == '__main__':
    main()

