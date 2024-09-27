CC=g++
#SRC := $(MNIST_ML_ROOT)/NEURAL_NETWORK/src 
SRC := $(PWD)/src
CFLAGS := -std=c++11 -g
#CFLAGS := -std=c++11 -g -DMNIST
INCLUDE_DIR := $(PWD)/include/*

all: main

main:	obj $(SRC)/main.cpp obj/network.o obj/layer.o obj/neuron.o
	$(CC) $(CFLAGS) $(SRC)/main.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) \ 
	-I$(MNIST_ML_ROOT)/include -ldata ./obj/*.o

obj:
	if [ ! -d "./obj" ]; then mkdir obj; fi

obj/network.o : $(SRC)/network.cpp
	$(CC) -c $(CFLAGS) $(SRC)/network.cpp -o obj/network.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include 

obj/layer.o : $(SRC)/layer.cpp
	$(CC) -c $(CFLAGS) $(SRC)/layer.cpp -o obj/layer.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include 

obj/neuron.o : $(SRC)/neuron.cpp
	$(CC) -c $(CFLAGS) $(SRC)/neuron.cpp -o obj/neuron.o -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include 

clean:
	rm -r main obj