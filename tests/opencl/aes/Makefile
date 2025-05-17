# ROOT_DIR := $(realpath ../../..)
# include $(ROOT_DIR)/config.mk

# PROJECT := aes

# SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

# SRCS := $(SRC_DIR)/main.cc

# common.h: $(SRC_DIR)/common.h
# 	cp $< $@

# kernel.cl: $(SRC_DIR)/kernel.cl
# 	cp $< $@
	
# KERNEL_SRCS := kernel.cl common.h

# OPTS ?= -n64

# include ../common.mk

ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := aes

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)
COMMON_DIR := $(VORTEX_HOME)/tests/opencl/common

# # Just the filenames for local build
# LOCAL_SRCS := \
# 	main.cc \
# 	aes_command_line_options.cc \
# 	aes_cl12_benchmark.cc \
# 	benchmark_runner.cc \
# 	cl_benchmark.cc \
# 	cl_error.cc \
# 	cl_file.cc \
# 	cl_profiler.cc \
# 	cl_runtime.cc \
# 	cl_util.cc \
# 	benchmark_command_line_options.cc \
# 	command_line_option.cc \
# 	option_parser_impl.cc \
# 	option_setting_help_printer.cc \
# 	option_setting_impl.cc \
# 	time_keeper_impl.cc \
# 	time_keeper_summary_printer.cc \
# 	time_measurement_impl.cc \
# 	timer_impl.cc

# Full paths for compilation
SRCS := \
	$(SRC_DIR)/main.cc \
	$(SRC_DIR)/aes_command_line_options.cc \
	$(SRC_DIR)/aes_cl12_benchmark.cc \
	$(COMMON_DIR)/benchmark/benchmark_runner.cc \
	$(COMMON_DIR)/cl_util/cl_benchmark.cc \
	$(COMMON_DIR)/cl_util/cl_error.cc \
	$(COMMON_DIR)/cl_util/cl_file.cc \
	$(COMMON_DIR)/cl_util/cl_profiler.cc \
	$(COMMON_DIR)/cl_util/cl_runtime.cc \
	$(COMMON_DIR)/cl_util/cl_util.cc \
	$(COMMON_DIR)/command_line_option/benchmark_command_line_options.cc \
	$(COMMON_DIR)/command_line_option/command_line_option.cc \
	$(COMMON_DIR)/command_line_option/option_parser_impl.cc \
	$(COMMON_DIR)/command_line_option/option_setting_help_printer.cc \
	$(COMMON_DIR)/command_line_option/option_setting_impl.cc \
	$(COMMON_DIR)/time_measurement/time_keeper_impl.cc \
	$(COMMON_DIR)/time_measurement/time_keeper_summary_printer.cc \
	$(COMMON_DIR)/time_measurement/time_measurement_impl.cc \
	$(COMMON_DIR)/time_measurement/timer_impl.cc

OBJS := $(LOCAL_SRCS:.cc=.o)

INCLUDES := \
	-I$(COMMON_DIR)/benchmark \
	-I$(COMMON_DIR)/cl_util \
	-I$(COMMON_DIR)/command_line_option \
	-I$(COMMON_DIR)/memory \
	-I$(COMMON_DIR)/time_measurement \
	-I$(SRC_DIR)

CXXFLAGS += $(INCLUDES)

# Object file rules
%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(COMMON_DIR)/benchmark/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(COMMON_DIR)/cl_util/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(COMMON_DIR)/command_line_option/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: $(COMMON_DIR)/time_measurement/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Kernel and header files
common.h: $(SRC_DIR)/common.h
	cp $< $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl common.h

OPTS ?= -n64

TARGET := aes

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(VORTEX_HOME)/build/runtime -lvortex -L$(HOME)/tools/pocl/lib -lOpenCL

include ../common.mk
