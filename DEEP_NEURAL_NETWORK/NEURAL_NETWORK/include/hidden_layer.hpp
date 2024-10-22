#ifndef __HIDDEN_LAYER_HPP
#define __HIDDEN_LAYER_HPP

#include "layer.hpp"

class HiddenLayer : public Layer
{
public:
    HiddenLayer(int prev, int current) : Layer(prev, current) {}
    void feedForward(Layer prev);
    void backProp(Layer next);
    void updateWeights(double, Layer *);
};

#endif