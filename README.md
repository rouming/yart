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

![Alt text](https://imgur.com/OkVEmeOl.png)

```
# Four different spheres on a plane under a distant light
# (inspired by scratchapixel.com)
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

![Alt text](https://imgur.com/hjSiEJhl.png)

```
# One sphere under three different point lights
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

## Example 3

![Alt text](https://imgur.com/af1RwmMl.png)

```
# Four glasses of water (inspired by scratchapixel.com)
$ ./yart --object type=mesh,file=./models/glasses.geo,material=reflect-refract,ior=1.3,smooth-shading=1,scale=0.1 \
         --object type=mesh,file=./models/backdrop1.geo,albedo=0.18,smooth-shading=1,scale=0.1,pattern=line[scale=0.06,angle=45] \
         \
         --light type=distant,dir=0,-1,-0.5,intensity=1.6 \
         \
         --pos 0,3,8  --pitch -13 --backcolor 3cacd7
```

## Example 4

![Alt text](https://imgur.com/8XFS3tdl.png)

```
# Just a cow (model taken from scratchapixel.com)
./yart --object type=mesh,file=./models/cow.obj,smooth-shading=0 \
       \
       --light dir=-1,0,-1,type=distant \
       \
       --pos 29,23,38 --pitch -19 --yaw -37.5 \
       --fov 27 --backcolor 3cacd7
```

## Example 5

![Alt text](https://imgur.com/5XI2wfdl.png)

```
# Smooth-shaded cyborg (model taken from learnopengl.com)
./yart --object type=mesh,file=./models/nanosuit.obj,smooth-shading=1 \
       \
       --light dir=-1,0,-1,type=distant \
       \
       --pos 29,23,38 --pitch -19 --yaw -37.5
```

## Example 6

![Alt text](https://imgur.com/6Dc1fPPl.png)

```
# One glass on a reflective surface (inspired by scratchapixel.com)
./yart --object type=mesh,file=./models/cylinder.geo,material=reflect-refract,ior=1.3,smooth-shading=1,scale=0.1,0.3,0.1 \
       --object type=mesh,file=./models/backdrop1.geo,albedo=0.18,r=0.1,smooth-shading=1,scale=0.1,pattern=check[scale=0.1,angle=45] \
       \
       --light type=distant,dir=0,-1,-0.5,intensity=1.6 \
       \
       --pos 0,3,8 --pitch -13 --backcolor 3cacd7
```

## Example 7

![Alt text](https://imgur.com/MfAQdyil.png)

```
# Glass of water with a pen (inspired by scratchapixel.com)
./yart --object type=mesh,file=./models/backdrop.geo,smooth-shading=1,pattern=line[scale=0.1,angle=-45] \
       --object type=mesh,file=./models/cylinder.geo,smooth-shading=1,material=reflect-refract,ior=1.5 \
       --object type=mesh,file=./models/pen.geo \
       \
       --light type=distant,dir=-0.5,-1,0.5,intensity=2 \
       \
        --pos 7,22,-30 --pitch -27 --yaw -164 --backcolor 3cacd7
```

## Example 8

![Alt text](https://imgur.com/7Bkbgzzl.png)

```
# Red ball, check plane under a point light
./yart --object type=sphere,radius=0.3,pos=0,0.3,0,Ks=0.1,Kd=0.9,0.01,0.01 \
       --object type=plane,Ks=0.05,Kd=0.8,pattern=check \
       \
       --light pos=1,2,0,type=point,intensity=400 \
       \
       --pitch -20 --yaw -5 --pos 0,2,5
```

## Example 9

![Alt text](https://imgur.com/I0n9iksl.png)

```
# Two metal shiny balls under four point lights on a reflective surface
./yart --object type=plane,Ks=0.05,Kd=0.8,r=0.4,pattern=check \
       --object type=sphere,radius=1,pos=0,1,0,Ks=0.5,Kd=0.6,r=0.2,n=500 \
       --object type=sphere,radius=0.5,pos=-1,0.5,-1.5,Ks=0.5,Kd=0.6,r=0.2,n=500 \
       \
       --light type=point,pos=-2,2.5,0,intensity=300,color=b2994c \
       --light type=point,pos=1.5,2.5,-1.5,intensity=200,color=727f99 \
       --light type=point,pos=1.5,2.5,1.5,intensity=100,color=b2ccb2 \
       --light type=point,pos=0,3.5,0,intensity=300,color=4c4c4c \
       \
       --pitch -16 --yaw -91 --pos 8,3,-1
```
