This is the software package of our ECCV paper:

C. Liu, J. Yuen, A. Torralba, J. Sivic and W. T. Freeman. SIFT flow: dense correspondence across different scenes. ECCV 2008.

Please cite our paper if you use our code for your research paper.


There is a big change compared to the original paper. We have a coarse-to-fine implementation of SIFT flow matching which runs much faster than the original algorithm presented in the paper.

Please go to "mex" subfolder and follow readme.txt to compile the cpp files. After the compilation is done, copy the dll to the current folder (unless you add the mex folder into MATLAB path). A precompiled dll mexDiscreteFlow.mexw64 for Winndows Vista x64 is included. But in general, compilation is needed.

------------------------- Important -------------------------

You must change one line in order to compile correctly. On line 5 of project.h, you should comment this line if you are compiling using visual studio in windows, or uncomment if you are in linux or mac os.

-------------------------------------------------------------

Run demo.m in MATLAB and you will see how SIFT flow works.

Enjoy!


Ce Liu
celiu@mit.edu
CSAIL MIT
Jan 2009