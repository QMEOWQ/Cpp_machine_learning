CC=g++
SRC := $(MNIST_ML_ROOT)/KMEANS/src 
CFLAGS := -std=c++11
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/kmeans.cpp
	$(CC) $(CFLAGS) $(SRC)/kmeans.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm main