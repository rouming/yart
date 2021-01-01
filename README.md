# YART
YART is a just Yet Another Ray Tracer renderer boosted by OpenCL
based on lessons from http://scratchapixel.com and http://pbr-book.org

The main goal is nothing but to learn ray tracing algorithms and
implement C code which can be compiled on host and simultaneously
compiled and loaded by OpenCL framework.

# HOWTO

Compilation is very simple: do just make.

By default YART loads itself (yart.c source) as OpenCL program and the
whole ray tracing is executed on GPU. In order to disable OpenCL
accelaration --no-opencl option can be provided.

When the scene is loaded and the window is opened you can observe with
the mouse and walk with WASD keys.

## Example 1

![Alt text](https://i.imgur.com/paKdHNp.gif)

```
# Creates 4 different spheres on 1 plane under distant light with
# specified direction. Camera is set to the position with pitch and
# yaw angles
$ ./yart --object type=sphere,radius=0.3,pos=-3,1,0,Ks=0.05  \
         --object type=sphere,radius=0.5,pos=-2,1,0,Ks=0.08  \
         --object type=sphere,radius=0.7,pos=-0.5,1,0,Ks=0.1 \
         --object type=sphere,radius=0.9,pos=1.4,1,0,Ks=0.4  \
         \
         --object type=mesh,file=./models/plane.geo \
         \
         --light dir=-0.4,-0.8,0.3,type=distant \
         \
         --backcolor 3cacd7 \
         --pitch -10 --yaw -55 --pos 10,3,7
```

## Example 2

![Alt text](https://i.imgur.com/qLc9wzUl.png)

```
# Creates 1 sphere under 3 different point lights
$ ./yart --object type=sphere,radius=0.8,pos=0,1,0,Ks=0.05 \
         \
         --object type=mesh,file=./models/plane.geo \
         \
         --light pos=-2,2,1,type=point,intensity=200,color=ff0000 \
         --light pos=0,4,1,type=point,intensity=190,color=00ff00  \
         --light pos=2,5,1,type=point,intensity=420,color=0000ff  \
         \
         --backcolor 3cacd7 \
         --pos=0,1,8
```
