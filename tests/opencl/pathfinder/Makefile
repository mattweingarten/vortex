ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := pathfinder

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

timing.h: $(SRC_DIR)/timing.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl timing.h

OPTS ?= -r100 -c100 -h1
CXXFLAGS += -I$(ROOT_DIR)/runtime/include

include ../common.mk
