# Since CcT is being used as a backend, need to configure it
include ../../CaffeConTroll/.config

# SHADJIS TODO: The Makefile below has a lot in common with the CaffeConTroll Makefile
# Can just include that here instead, or refactor common parts into Makefile.common
# In particular the main source will be different but otherwise the Makefiles are similar.

# Some commands may depend on the shell
UNAME := $(shell uname)

# Libraries to include
# SHADJIS TODO: config++ should not be needed anymore
LIBS=lmdb glog zmq config++ $(BLAS_LIBS) 
LD_BASE=$(foreach l, $(LIBS), -l$l)

# Header directories
INCLUDE_DIRS=$(BOOST_INCLUDE) $(GLOG_INCLUDE) \
	     $(LMDB_INCLUDE) $(BLAS_INCLUDE) ./src ../../CaffeConTroll/src/parser/
INCLUDE_STR=$(foreach d, $(INCLUDE_DIRS), -I$d)

# Library directories
LIB_DIRS=$(BOOST_LIB_DIR) $(GLOG_LIB_DIR) \
	 $(LMDB_LIB_DIR) $(BLAS_LIB_DIR) /usr/local/lib
LIB_STR=$(foreach d, $(LIB_DIRS), -L$d)

# For Mac OS X 10.10 x86_64 Yosemite
ifeq ($(UNAME), Darwin)
  CFLAGS = -Wall -std=c++11 -fsanitize-undefined-trap-on-error -fsanitize=integer-divide-by-zero
  LDFLAGS = $(LD_BASE) -lboost_program_options-mt -lboost_serialization -lpthread
  NVCCFLAGS = -D_GPU_TARGET -D_INCLUDE_GPUDRIVER -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options-mt -lboost_serialization -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
# For Ubuntu x86_64
else ifeq ($(UNAME), Linux)
  CFLAGS = -Wall -Wl,--no-as-needed -std=c++11
  NVCCFLAGS = -D_GPU_TARGET -D_INCLUDE_GPUDRIVER -std=c++11 $(LD_BASE) -lcublas -lcuda -lboost_program_options -lboost_serialization -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
  LDFLAGS = $(LD_BASE) -lrt -lboost_program_options -lboost_serialization -lpthread 
endif

# BLAS includes
CFLAGS += $(BLAS_DEFS)

DIR_PARAMS=$(INCLUDE_STR) $(LIB_STR)
PRODUCT_FLAGS = -Ofast -D_FASTPOW

# Protobuf variables
PROTOBUF_LIB = -lprotobuf
PROTO_SRC_DIR=../../CaffeConTroll/src/parser/
PROTO_CC=protoc --cpp_out=.
PROTO_SRC=cnn.proto
PROTO_COMPILED_SRC=$(PROTO_SRC_DIR)cnn.pb.cc

# Source file for main program
TARGET = size_util
SRC = ../../CaffeConTroll/src/DeepNetConfig.cpp ../../CaffeConTroll/src/util.cpp ../../CaffeConTroll/src/timer.cpp ../../CaffeConTroll/src/sched/DeviceDriver_CPU.cpp size_util.cpp
OBJ_FILES = $(patsubst %.cpp,%.o,$(SRC))

# CUDA Includes
ifdef NVCC
MAIN_CUDA_SOURCES = ../../CaffeConTroll/src/sched/DeviceDriver_GPU.cu
MAIN_CUDA_OBJ_FILES = $(patsubst %.cu,%.o,$(MAIN_CUDA_SOURCES))
CFLAGS += -D_INCLUDE_GPUDRIVER  -I $(CUDA_INCLUDE) -L $(CUDA_LIB)
endif

# Linker flags
LINKCC = $(CC)
LINKFLAG = $(CFLAGS) $(LDFLAGS)

ifdef NVCC
LINKFLAG += -lcurand -lcublas -lcudart
NVCC_LINK = dlink.o
endif

# See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html 
.PHONY: all clean

# Default goal (target) is make all (equivalent to make because it is first)
# Dependencies: main.cpp, protobuf, cuda
all: CFLAGS += $(PRODUCT_FLAGS) 
all: LINKFLAG += $(PRODUCT_FLAGS) 
all: $(OBJ_FILES) cnn.pb.o $(MAIN_CUDA_OBJ_FILES)
ifdef NVCC
	$(NVCC) -dlink $^ -o $(NVCC_LINK)
endif
	$(LINKCC) $^ $(NVCC_LINK) -o $(TARGET) $(LINKFLAG) $(DIR_PARAMS) $(LDFLAGS) $(PROTOBUF_LIB)
	rm cnn.pb.o size_util.o

# Compile C++ sources
%.o: %.cpp $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $< -o $@

# Compile CUDA sources
%.o: %.cu $(PROTO_COMPILED_SRC)
	$(NVCC) -O3 $(BLAS_DEFS) $(NVCCFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) -dc $< -o $@

# Compile protobuf sources
cnn.pb.o: $(PROTO_COMPILED_SRC)
	$(CC) $(CFLAGS) $(INCLUDE_STR) $(TEST_BLASFLAGS) $(PROTOBUF) -c $(PROTO_COMPILED_SRC)

$(PROTO_COMPILED_SRC): $(PROTO_SRC_DIR)$(PROTO_SRC)
	cd $(PROTO_SRC_DIR); $(PROTO_CC) $(PROTO_SRC); cd -

# Clean target
clean:
	rm -f $(TARGET)
	rm -f $(PROTO_SRC_DIR)*.pb.*
	rm -f $(OBJ_FILES)
	rm -f $(MAIN_CUDA_OBJ_FILES)

