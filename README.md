# FBRLF
An Efficient and Geometric-distortion-free Binary Robust Local Feature (FBRLF)

## Abstract
An efficient and geometric-distortion-free approach, namely Fast Binary Robust Local Feature (FBRLF), is proposed. The FBRLF searches the stable features from an image with the proposed Multiscale Adaptive AGAST (MAAGAST) to yield optimum threshold value based on AGAST. To overcome the image noise, the Gaussian template is applied, which is efficiently boosted by the adoption of integral image. The feature matching is conducted by incorporating the voting mechanism and lookup table method to achieve high accuracy with low computational complexity. The experimental results clearly demonstrate the superiority of the proposed method compared with the former schemes regarding local stable features performance and processing efficiency.

## Dependencies
1. OpenCV 3.0

## Model Usage

In the main.cpp, there are two parameters(main (int argc, char** argv)). The argc means the target image and the argv is the number of comparison in the dataset.

## Disclaimer
The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so the use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use.
