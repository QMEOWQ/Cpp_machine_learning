#ifndef __NEURON_HPP
#define __NEURON_HPP

#include <vector>
#include <cmath>

class Neuron
{
public:
    double output;
    double delta;
    std::vector<double> weights;

    Neuron(int, int);
    void initializeWeights(int);
};

// class Neuron
// {
//     std::vector<double> weights;
//     double preActivation;
//     double activatedOutput;
//     double outputDerivative;
//     double error; // diff
//     double alpha; // used in activation funcs

// public:
//     Neuron(int, int);
//     ~Neuron();

//     void initializeWeights(int previousLayerSize, int currentLayerSize);
//     void setError(double);
//     void setWeight(double, int);

//     double calculatePreActivation(std::vector<double>);
//     double activate();
//     double calculateOutputDerivate();

//     // 激活函数
//     double sigmoid();
//     double relu();
//     double leakyRelu();
//     double inverseSqrtRelu();

//     double getOutput();
//     double getOutputDerivative();
//     double getError();
//     std::vector<double> getWeights();
// };

#endif