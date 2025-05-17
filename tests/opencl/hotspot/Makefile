ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := hotspot

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/hotspot.cpp

hotspot_kernel.cl: $(SRC_DIR)/hotspot_kernel.cl
	cp $< $@
	
KERNEL_SRCS := hotspot_kernel.cl

OPTS ?= -n64

include ../common.mk
