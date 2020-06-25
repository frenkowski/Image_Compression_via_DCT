# Methods of Scientific Calculation Project 2

# Image Compression via DCT

## Team

Ali Manan (_mohammadalimanan+github@gmail.com_)  
Francesco Porto (_francescoporto97+github@gmail.com_)  
Francesco Stranieri (_frenkowski+github@gmail.com_)

## Abstract

The DCT is a Fourier-related transform, that expresses a finite se- quence of values in terms of a sum of cosine functions oscillating at dif- ferent frequencies. It is vastly used in the image processing field, mostly for compression related tasks, such as in _JPEG_, a lossy format. Lossy compression permits reconstruction of an approximation of the original data, but reduces file size when compared to lossless formats, such as _BMP_ (bitmap).

## Introduction

In this report we discuss our implementation of the **Discrete Cosine Transform** (DCT) in one and two dimension, called DCT1 and DCT2 respectively. We then compare our implementations in terms of _execution time_ to the ones provided by Scipy, an open-source Python library. Finally, we present our GUI that compresses a greyscale bitmap image via a DCT2-based algorithm, while also providing a side-by-side comparison with the original image.
