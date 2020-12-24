CC = gcc
DEFINES = -D_GNU_SOURCE
CFLAGS = -g -O3 -Wall

DEPS = $(shell find . -name '*.h')
SOURCES:= $(shell find . -name '*.c')
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

%.o: %.c $(DEPS)
ifneq ($(VERBOSE),1)
	@echo CC $@
endif
	$(Q)$(CC) -c -o $@ $< $(CFLAGS)

yart: $(OBJ)
ifneq ($(VERBOSE),1)
	@echo LD $@
endif
	$(Q)$(CC) -o $@ $^ -lm -lOpenCL

.PHONY: clean

clean:
	$(Q)rm -f yart core
	$(Q)find . \( -name \*.o -or -name \*~ \) -delete
