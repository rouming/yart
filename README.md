# YART
YART is a Yet Another Ray Tracer renderer that can be accelerated by
OpenCL or CUDA, with black hole simulation as a bonus (see examples
below).

Based on lessons from http://scratchapixel.com and http://pbr-book.org

The main goal is to learn ray tracing algorithms and implement C code that can be
compiled on the host and simultaneously compiled and executed by an OpenCL or CUDA
GPU framework.

# Dependencies

The following packages are required on Ubuntu/Debian:

```
sudo apt install build-essential libsdl2-dev libsdl2-ttf-dev libassimp-dev xxd
```

- `build-essential` -- GCC and standard build tools
- `libsdl2-dev` -- SDL2 for window creation and input handling
- `libsdl2-ttf-dev` -- SDL2 TTF extension for on-screen text (FPS counter, etc.)
- `libassimp-dev` -- Assimp for loading 3D model files (.obj, .geo, etc.)
- `xxd` -- used to embed the OpenCL kernel source as a C header at build time

For OpenCL builds, also install the ICD loader and a platform runtime:

```
sudo apt install ocl-icd-opencl-dev
```

A platform-specific runtime is also needed: NVIDIA and AMD GPUs are covered by
their proprietary drivers; for Intel GPUs install `intel-opencl-icd`; for a
CPU-only OpenCL runtime install `pocl-opencl-icd`.

For CUDA builds, install the CUDA toolkit from https://developer.nvidia.com/cuda-downloads
(provides `nvcc` and `cuda_runtime.h`). The CUDA toolkit is not available via apt.

# Building

Three build targets are available, all producing a binary named `yart`.

**CUDA (default):**
```
make
```
or explicitly:
```
make yart-cuda
```
Requires CUDA toolkit (`nvcc`) and an NVIDIA GPU.

**OpenCL:**
```
make yart-opencl
```
Requires an OpenCL-capable GPU and the OpenCL runtime (`libOpenCL`).

**CPU (no GPU acceleration):**
```
make yart-cpu
```
Runs entirely on the CPU. No GPU required.

# Running

By default YART uses GPU acceleration. To disable it and fall back to the CPU
renderer, pass `--no-accel`.

When the scene is loaded and the window is opened you can look around with the
mouse and walk with the WASD keys.

## Rendering Modes

YART implements two fundamentally different rendering algorithms selectable at
runtime.

### Whitted Ray Tracing (default)

The default mode implements the algorithm introduced by Turner Whitted in 1980.
It is a deterministic, recursive ray tracer:

- At each surface hit, it shoots **shadow rays** to every light source to
  determine visibility (hard shadows).
- Reflective surfaces spawn one **mirror reflection ray**.
- Transparent surfaces spawn one **refraction ray** (and one reflection ray
  weighted by the Fresnel term).
- Shading follows the **Phong model**: diffuse + specular highlights from each
  light, summed analytically.

Because every decision is deterministic, a single sample per pixel converges
immediately to a clean image.  The trade-off is physical accuracy: Whitted
tracing only models specular (mirror-like) light transport.  Soft shadows,
color bleeding between surfaces, and realistic diffuse inter-reflections are
not possible.  The scene must have at least one explicit light source
(`--light`).

### Path Tracing (--path)

`--path` enables a physically based Monte Carlo renderer inspired by
*Ray Tracing in One Weekend* (RTIOW).  Instead of making deterministic
shading decisions, it traces **random walks**:

- At each hit, one scatter direction is sampled stochastically from the
  material's BRDF (cosine-weighted hemisphere for Lambertian surfaces,
  perturbed mirror reflection for metals, stochastic Fresnel for glass).
- The ray bounces until it escapes to the sky or is absorbed.  The sky color
  (`--backcolor` / `--backcolor-horizon`) **is the light source** -- no
  explicit lights are needed or used.
- Many paths are averaged per pixel (`--samples-per-pixel`) to reduce noise.
  Russian Roulette terminates low-contribution paths early to keep runtime
  bounded.

Because every light-transport effect emerges from the same random walk, path
tracing naturally produces **soft shadows**, **color bleeding** (global
illumination), **caustics**, and **depth of field** (`--defocus-angle` /
`--focus-dist`).  The cost is that it requires many samples to converge:
expect noise at low sample counts and clean images only after 32-256+ samples
per pixel depending on scene complexity.

| Property | Whitted (default) | Path Tracing (--path) |
|---|---|---|
| Shadows | Hard (shadow rays) | Soft (emerges from sampling) |
| Diffuse inter-reflection | No | Yes |
| Color bleeding | No | Yes |
| Depth of field | No | Yes |
| Light source | Explicit --light | --backcolor sky |
| Samples needed | 1 | 32-256+ |
| Convergence | Instant | Noisy then clean |

## Example 1: Black Hole (whitted tracing)

Three colored spheres (blue, green, red) are placed far behind a black hole,
with a checkered plane as the ground.  From the camera's viewpoint the black
hole sits almost exactly in front of the blue sphere, producing an
**Einstein ring**: rays that would normally miss the blue sphere are bent by
gravity around all sides of the black hole and redirected toward the camera,
so the blue sphere appears as a blue ring encircling the dark event
horizon. The green and red spheres are off-axis, flipped, and appear as ordinary
lensed objects.

The event horizon itself is visible as a black disc -- rays that cross the
Schwarzschild radius (RS = 2*mass) are absorbed and return no color.
Gravitational redshift dims the blue and green channels of light that passes
close to the horizon, so objects seen through the inner lensing region shift
toward red.

![Black Hole](https://i.imgur.com/LYOr6LL.png)

```
./yart --object type=sphere,radius=0.6,pos=-12,1.3,-13,Kd=0,0,1,Ks=0.05 \
       --object type=sphere,radius=0.6,pos=-12,1.3,-10,Kd=0,1,0,Ks=0.05 \
       --object type=sphere,radius=0.6,pos=-12,1.3,-16,Kd=1,0,0,Ks=0.05 \
       \
       --object type=blackhole,pos=-3,2,-5,mass=0.2,step=0.05,maxsteps=1000,escape=80,colorshift=10 \
       \
       --object type=plane,Ks=0.05,Kd=0.8,pattern=check \
       \
       --light dir=-0.4,-0.8,0.3,type=distant,intensity=7 \
       \
       --backcolor 3cacd7 \
       --pitch -5 --yaw -49 --pos 6,3,3
```

## Example 2: RTIOW Final Scene (path tracing)

The classic [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
final scene: ~480 randomly placed small spheres of mixed materials on a ground
sphere, three hero spheres, sky gradient, and camera depth of field.  Each run
randomizes the small sphere placement.  Requires `bash` and `python3`.

![RTIOW scene rendered with YART path tracer](https://i.imgur.com/tWfHoF5.png)

```bash
#!/usr/bin/env bash
# RTIOW final scene: ground sphere + 3 hero spheres + 22x22 random small spheres.
# Usage: bash YART-COMMANDS.txt
#
# Python generates one --object argument pair per line (flag then value) for
# each small sphere.  mapfile reads them into an array so the shell passes
# them as individual argv entries -- no eval, no word-splitting surprises.
# No fixed seed: every run produces a different arrangement, matching RTIOW.

mapfile -t objects < <(python3 - <<'PYEOF'
import random, math

for a in range(-11, 11):
    for b in range(-11, 11):
        cx = a + 0.9 * random.random()
        cz = b + 0.9 * random.random()
        cy = 0.2
        # Skip spheres too close to the right hero sphere, as in RTIOW
        if math.sqrt((cx - 4)**2 + cz**2) <= 0.9:
            continue
        m = random.random()
        if m < 0.8:
            # Lambertian: albedo = random * random (darker, more varied)
            r = random.random() * random.random()
            g = random.random() * random.random()
            b = random.random() * random.random()
            print('--object')
            print(f'type=sphere,radius=0.2,pos={cx:.3f},{cy},{cz:.3f}'
                  f',material=lambertian,Kd={r:.3f},{g:.3f},{b:.3f}')
        elif m < 0.95:
            # Mirror: albedo in [0.5, 1], fuzz in [0, 0.5]
            r = 0.5 + 0.5 * random.random()
            g = 0.5 + 0.5 * random.random()
            b = 0.5 + 0.5 * random.random()
            fz = 0.5 * random.random()
            print('--object')
            print(f'type=sphere,radius=0.2,pos={cx:.3f},{cy},{cz:.3f}'
                  f',material=mirror,Kd={r:.3f},{g:.3f},{b:.3f},fuzz={fz:.3f}')
        else:
            # Dielectric glass
            print('--object')
            print(f'type=sphere,radius=0.2,pos={cx:.3f},{cy},{cz:.3f}'
                  f',material=dielectric,ior=1.5')
PYEOF
)

./yart --path \
  --width 1200 --height 675 --fov 20 \
  --pos 13,2,3 --pitch -8 --yaw -77 \
  --backcolor 80b3ff --backcolor-horizon ffffff \
  --defocus-angle 0.6 --focus-dist 10.0 \
  --ray-depth 8 --samples-per-pixel 32 \
  --object type=sphere,radius=1000,pos=0,-1000,0,Kd=0.5,0.5,0.5 \
  --object type=sphere,radius=1,pos=0,1,0,material=dielectric,ior=1.5 \
  --object type=sphere,radius=1,pos=-4,1,0,material=lambertian,Kd=0.4,0.2,0.1 \
  --object type=sphere,radius=1,pos=4,1,0,material=mirror,Kd=0.7,0.6,0.5 \
  "${objects[@]}"
```

### Example 3 (whitted tracing)

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

### Example 4 (whitted tracing)

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

### Example 5 (whitted tracing)

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

### Example 6 (whitted tracing)

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

### Example 7 (whitted tracing)

![Alt text](https://imgur.com/5XI2wfdl.png)

```
# Smooth-shaded cyborg (model taken from learnopengl.com)
./yart --object type=mesh,file=./models/nanosuit.obj,smooth-shading=1 \
       \
       --light dir=-1,0,-1,type=distant \
       \
       --pos 29,23,38 --pitch -19 --yaw -37.5
```

### Example 8 (whitted tracing)

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

### Example 9 (whitted tracing)

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

### Example 10 (whitted tracing)

![Alt text](https://imgur.com/7Bkbgzzl.png)

```
# Red ball, checkered plane under a point light
./yart --object type=sphere,radius=0.3,pos=0,0.3,0,Ks=0.1,Kd=0.9,0.01,0.01 \
       --object type=plane,Ks=0.05,Kd=0.8,pattern=check \
       \
       --light pos=1,2,0,type=point,intensity=400 \
       \
       --pitch -20 --yaw -5 --pos 0,2,5
```

### Example 11 (whitted tracing)

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
