CC = gcc
DEFINES = -D_GNU_SOURCE
CFLAGS = -g -O3 -std=gnu89 -Wall

NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets

PREP = ./render-opencl-preprocessed.cl
GEN  = ./render-opencl.h
# Exclude $GEN header
DEPS = $(filter-out $(GEN), $(shell find . -name '*.h'))

CUDA_SOURCES := render-cuda.cu
CUDA_OBJ = $(CUDA_SOURCES:.cu=.o)

ifeq ("$(origin V)", "command line")
  VERBOSE = $(V)
endif
ifndef VERBOSE
  VERBOSE = 0
endif

ifeq ($(VERBOSE),1)
  Q =
else
  Q = @
endif

.PHONY: yart-cuda yart-cpu yart-opencl clean

# Default: CUDA build
yart: YARTFLAGS = -DUSE_CUDA -I/usr/local/cuda/include
yart: yart.o bvh.o $(CUDA_OBJ)
ifneq ($(VERBOSE),1)
	@echo " LD $@"
endif
	$(Q)$(NVCC) $(NVCCFLAGS) -o $@ yart.o bvh.o $(CUDA_OBJ) -lm -lSDL2_ttf -lSDL2 -lassimp

yart-cuda: yart

yart-cpu: YARTFLAGS = -DUSE_CPU
yart-cpu: yart.o bvh.o
ifneq ($(VERBOSE),1)
	@echo " LD $@"
endif
	$(Q)$(CC) -o yart yart.o bvh.o -lm -lSDL2_ttf -lSDL2 -lassimp

yart-opencl: yart.o bvh.o $(GEN)
ifneq ($(VERBOSE),1)
	@echo " LD $@"
endif
	$(Q)$(CC) -o yart yart.o bvh.o -lm -lSDL2_ttf -lSDL2 -lassimp -lOpenCL

ifneq (,$(filter yart-opencl,$(MAKECMDGOALS)))
yart.o: $(GEN)
endif
yart.o: yart.c $(DEPS)
ifneq ($(VERBOSE),1)
	@echo " CC $@"
endif
	$(Q)$(CC) $(CFLAGS) $(YARTFLAGS) -c -o $@ $<

bvh.o: bvh.c $(DEPS)
ifneq ($(VERBOSE),1)
	@echo " CC $@"
endif
	$(Q)$(CC) $(CFLAGS) $(YARTFLAGS) -c -o $@ $<

$(CUDA_OBJ): $(CUDA_SOURCES) $(DEPS)
ifneq ($(VERBOSE),1)
	@echo " NVCC $@"
endif
	$(Q)$(NVCC) $(NVCCFLAGS) -c -o $@ $(CUDA_SOURCES)

$(GEN): $(PREP)
ifneq ($(VERBOSE),1)
	@echo HDR $@
endif
	$(Q) xxd -i $< $@

# The thing is that (I don't know why, but the fact is) NVIDIA OpenCL
# online compiler is not able to detect properly changes in includes
# (probably compiler caches somewhere are stored?) and the old code
# is used, even main render ray-tracing headers we changed heavily.
# So use gcc to output preprocessed source.
$(PREP): render-opencl.cl $(DEPS)
ifneq ($(VERBOSE),1)
	@echo "GEN $@"
endif
	$(Q)$(CC) -E -std=c99 -x c -D__OPENCL__ -o $@ $<

clean:
	$(Q)rm -f yart core $(GEN) $(PREP)
	$(Q)find . \( -name \*.o -or -name \*~ \) -delete
