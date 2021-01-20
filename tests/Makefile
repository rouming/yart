CC = gcc
DEFINES = -D_GNU_SOURCE
CFLAGS = -g -O3 -std=gnu89 -Wall

DEPS = $(shell find ../ -name '*.h')
SOURCES:= $(shell find ./ -name '*.c')
TESTS = $(SOURCES:.c=)

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

all: $(TESTS)

%:%.c $(DEPS)
ifneq ($(VERBOSE),1)
	@echo "CC $@"
endif
	$(Q)$(CC) -o $@ $< $(CFLAGS)

.PHONY: clean all

clean:
	$(Q)rm -f $(OBJS) $(TESTS) *~
