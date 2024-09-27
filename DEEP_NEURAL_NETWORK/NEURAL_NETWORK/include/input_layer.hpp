#ifndef __INPUT_LAYER_HPP
#define __INPUT_LAYER_HPP

#include "layer.hpp"
#include "data.hpp"

class InputLayer : public Layer
{
public:
    InputLayer(int prev, int current) : Layer(prev, current) {}
    void setLayerPutputs(data *d);
};

#endif