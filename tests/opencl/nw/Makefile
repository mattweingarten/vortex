ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := nw
TARGET := nw

SRC_DIR := $(realpath .)
SRCS := $(SRC_DIR)/nw.cc
KERNEL_SRCS := nw.cl

# Declare that input.txt is needed (but no need to copy it)
EXTRA_SRCS := input.txt

# Benchmark arguments (with -n64 at the end)
OPTS ?=
ARGS := 128 10 input.txt -n64

include ../common.mk

BINARY := ./$(TARGET)

# Override the run target after common.mk is fully loaded
define override_run_rtlsim
run-rtlsim:
	LD_LIBRARY_PATH=$$(LD_LIBRARY_PATH) \
	VORTEX_DRIVER=$$(VORTEX_DRIVER) \
	$$(BINARY) $$(ARGS)
endef

$(eval $(override_run_rtlsim))
