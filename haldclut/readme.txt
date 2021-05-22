haldclut
========

These files are the fundamets of the color vision deficiency si-
mulation. Basically the files contain the color vison defect RGB
counterpart of any given RGB color.


About Hald CLUTs in General
===========================

Any color correction can be expressed as a Color LookUp Table or
CLUT (some times written as "Color LUT").  This a 3D dimensional
table where all colors are represented in color space.  For each
color in the  color lookup table  there is  a destination  color
value that corresponds to what the particular color becomes when
it is corrected using the CLUT. These tables are by nature 3-di-
mensional  (Red Green and Blue)  and therefore special file for-
mats are used to store them.  Hald CLUTs  however have been con-
verted to a 2D space and since tables  store colors the CLUT can
be be saved as a image, in any non destructive image format.

More info: http://www.quelsolaar.com/technology/clut.html


About These Hald CLUTs
======================

All Hald CLUTs provided  are lossless 4096x4096 pixel size PNGs,
and all images were reduced in size by OptiPNG 0.7.7:

http://optipng.sourceforge.net/


identity.png                        
                Normal vision. An identity CLUT is a CLUT that
                doesn't affect the image it is applied to. 
