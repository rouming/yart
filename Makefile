CC = gcc
DEFINES = -D_GNU_SOURCE
CFLAGS = -g -O3 -std=gnu89 -Wall

PREP = ./render-opencl-preprocessed.cl
GEN  = ./render-opencl.h
# Exclude $GEN header
DEPS = $(filter-out $(GEN), $(shell find . -name '*.h'))
SOURCES:= yart.c
OBJ = $(SOURCES:.c=.o)

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

%.o: %.c $(DEPS) $(GEN)
ifneq ($(VERBOSE),1)
	@echo " CC $@"
endif
	$(Q)$(CC) -c -o $@ $< $(CFLAGS)

yart: $(OBJ) $(GEN)
ifneq ($(VERBOSE),1)
	@echo " LD $@"
endif
	$(Q)$(CC) -o $@ $(OBJ) -lm -lSDL2_ttf -lSDL2 -lassimp -lOpenCL

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

.PHONY: clean

clean:
	$(Q)rm -f yart core $(GEN) $(PREP)
	$(Q)find . \( -name \*.o -or -name \*~ \) -delete
