# YART
YART is a just Yet Another Ray Tracing renderer boosted by OpenCL
based on lessons from http://scratchapixel.com and http://pbr-book.org

The main goal is nothing but to learn ray tracing algorithms and implement
C code which can be compiled on host and simultaneously compiled and loaded
by OpenCL framework.

# HOWTO

## Example 1

![Alt text](https://i.imgur.com/NAM2ZMnl.png)

```
# Creates 4 different spheres on 1 plane. Camera is set to the position
# with pitch and yaw angles
$ ./yart --sphere r=0.3,pos=-3,1,0,Ks=0.05  \
         --sphere r=0.5,pos=-2,1,0,Ks=0.08  \
		 --sphere r=0.7,pos=-0.5,1,0,Ks=0.1 \
		 --sphere r=0.9,pos=1.4,1,0,Ks=0.4  \
		 \
		 --pitch -10 --yaw -55 --pos 10,3,7 \
		 \
		 ./models/plane.geo
```
