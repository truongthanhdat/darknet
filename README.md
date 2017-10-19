# Darknet

## Introduction

+ This project clones from [darknet](http://github.com/pjreddie/darknet).

+ This is an open source neural network framework written in C and CUDA.

+ Project's author: pjreddie

## My new adding features:

+ Adding converting Numpy array RGB to Image format of darknet.

```python
import darknet
from skimage import io

img = io.imread('path/to/image')
im_ = darknet.convertImage(img.ctypes.data, img.shape[0], img.shape[1], img.shape[2])
```

+ It easily uses darknet with python.

## Building

+ If you want to build with GPU, you need to edit Makefile. You set GPU=1 and CUDNN=1

+ Type make to build. After building finished, you need to copy libdarknet.so to python folder.
