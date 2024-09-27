#ifndef __LAYER_HPP
#define __LAYER_HPP

#include "neuron.hpp"
#include <vector>
#include <stdint.h>

static int layerId = 0;

class Layer
{
public:
    int currentLayerSize;
    std::vector<Neuron *> neurons;
    std::vector<double> layerOutputs;

    Layer(int, int);
};

// class Layer
// {
// public:
//     int currentLayerSize;
//     std::vector<Neuron *> neurons;
//     std::vector<double> layerOutput;

//     Layer(int, int);
//     ~Layer();

//     std::vector<double> getLayerOutputs();
//     int getSize();
// };

#endif