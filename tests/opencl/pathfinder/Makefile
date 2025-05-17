VORTEX_RT_PATH := $(realpath ../../../runtime)

ARGS           ?= -r 512 -c 512 -h 16

CC             := g++
CFLAGS         := -std=c++17 -g -Wall -Wextra -pedantic -Wfatal-errors

CXXFLAGS      += -I$(VORTEX_RT_PATH)/include \
                  -I$(realpath ../../../hw/rtl) \
                  -I$(VORTEX_RT_PATH)/tests/util \
                  -I/home/pranavipuvvadi/vortex/tests/util

LDFLAGS        := -L$(VORTEX_RT_PATH)/lib -lvortex -lOpenCL

SRC            := $(realpath $(CURDIR)/../../../../tests/opencl/pathfinder/main.cc)
OBJ            := main.o
BIN            := pathfinder

.PHONY: all clean run run-rtlsim

all: $(BIN)

$(OBJ): $(SRC)
	@echo "Compiling host → $@"
	$(CC) $(CFLAGS) $(CXXFLAGS) -c $(SRC) -o $@

$(BIN): $(OBJ)
	@echo "Linking → $@"
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "LDFLAGS used: $(LDFLAGS)"

clean:
	@rm -f $(OBJ) $(BIN)

run: all
	@echo "Running $(BIN) with: $(ARGS)"
	@./$(BIN) $(ARGS)

run-rtlsim: all
	@echo "Running $(BIN) under rtlsim with: $(ARGS)"
	@VORTEX_DRIVER=rtlsim ./$(BIN) $(ARGS)
