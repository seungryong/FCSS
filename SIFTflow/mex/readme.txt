The *.cpp and *.h files in this folder are used to mex a matlab dll file for SIFT flow matching. Use the following command to compile in Matlab (tested on version 7.6 or later)

mex mexDiscreteFlow.cpp BPFlow.cpp Stochastic.cpp

It has been tested in Windows Vista x64, Linux x64 and Mac OS 10.5. A precompiled version in Windows Vista x64 is included in the folder.


------------------------- Important -------------------------

You must change one line in order to compile correctly. On line 5 of project.h, you should comment this line if you are compiling using visual studio in windows, or uncomment if you are in linux or mac os.

-------------------------------------------------------------

Please contact Ce Liu (celiu@mit.edu) should you encounter any problems in compiling or find any bugs.