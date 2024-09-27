#ifndef __NETWORK_HPP
#define __NETWORK_HPP

#include "data.hpp"
#include "neuron.hpp"
#include "input_layer.hpp"
#include "layer.hpp"
#include "hidden_layer.hpp"
#include "output_layer.hpp"
#include "common.hpp"

class Network : public common_data
{
public:
    std::vector<Layer *> layers;
    double learningRate;
    double testPerformance;

    Network(std::vector<int> spec, int, int, double);
    ~Network();
    std::vector<double> fprop(data *d);                        // feedforward
    double activate(std::vector<double>, std::vector<double>); // dot product
    double transfer(double);
    double transferDerivative(double); // used for backprop
    void bprop(data *d);
    void updateWeights(data *d);
    int predict(data *d); // return index of max value in output array
    void train(int);
    void validate();
    double test();
};

// class Network : public common_data
// {
// private:
//     InputLayer *inputLayer;
//     OutputLayer *outputLayer;
//     std::vector<HiddenLayer *> hiddenLayers;
//     double eta; // learning rate

// public:
//     Network(std::vector<int> hiddenLayerSpec, int, int);
//     ~Network();

//     void fprop(data *d);
//     void bprop(data *d);
//     void updateWeights();
//     void train();
//     void validate();
//     void test();
// };

#endif