CC=g++
SRC := $(MNIST_ML_ROOT)/KNN/src 
CFLAGS := -std=c++11 -DEUCLID 
INCLUDE_DIR := $(PWD)/include/*

all: main

main: $(SRC)/knn.cpp
	$(CC) $(CFLAGS) $(SRC)/knn.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata

clean:
	rm main