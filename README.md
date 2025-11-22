# Low Poly Generator

Configuration and further instructions are in the file itself. Should be self-
explanatory.

## [lowpoly_gen.py](lowpoly_gen.py)

Takes in an SVG (with outlines) and a reference image.
Outputs an SVG that has been tesselated w/triangles and colored based on the
reference image. The input images must be in the same directory that the python
file is run in.

## [col_avg.py](col_avg.py)

Takes in an SVG and reference image and colors polygons based on what they
enclose in the reference image. Made for triangles (i.e. other polygons have not
been tested) and is not being worked on.