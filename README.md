# Non-Deep Learning Grocery Product Classification

This software uses OpenCV's feature detection and description to perform classification of grocery products. To use this, you first need to extract and describe keypoints using [this repo](https://github.com/andikarachman/Keypoints-Extraction-for-Grocery-Product-Classification). Please see the demo of the software below.  

<div style="text-align:center; margin:20px" >
  <video width="640" height="480" controls>
    <source src="demo.mp4" type="video/mp4">
  </video>
</div>

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions
1. Clone this repo.
2. Put the directory of the build file of your OpenCV in the `CMakeLists.txt`.
3. Put all of the products detected keypoints files (the ones with `txt` filetype) in `ref/keypoints/` folder. Some examples are provided for references. 
4. Put all of the products keypoints descriptors files (the ones with `xml` filetype) in `ref/descriptors/` folder. Some examples are provided for references. 
5. Make a build directory in the top level directory: `mkdir build && cd build`
6. Compile: `cmake .. && make`
7. Run it: `./product_classification`.