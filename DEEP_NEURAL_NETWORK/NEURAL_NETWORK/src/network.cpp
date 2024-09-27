#include "network.hpp"
#include "layer.hpp"
#include "data_handler.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int inputSize, int numClasses, double learningRate)
{
    for (int i = 0; i < spec.size(); i++)
    {
        if (i == 0)
        {
            layers.push_back(new Layer(inputSize, spec.at(i)));
        }
        else
        {
            layers.push_back(new Layer(layers.at(i - 1)->neurons.size(), spec.at(i)));
        }
    }
    layers.push_back(new Layer(layers.at(layers.size() - 1)->neurons.size(), numClasses));
    this->learningRate = learningRate;
}

Network::~Network()
{
    // just free memory
}

double Network::activate(std::vector<double> weights, std::vector<double> input) // dot product
{
    double activation = weights.back(); // bias term
    for (int i = 0; i < weights.size() - 1; i++)
    {
        activation += weights[i] * input[i];
    }
    return activation;
}

double Network::transfer(double activation)
{
    return 1.0 / (1.0 + exp(-activation));
}

double Network::transferDerivative(double output) // used for backprop
{
    return output * (1 - output);
}

std::vector<double> Network::fprop(data *d) // feedforward
{
    std::vector<double> input = *d->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++)
    {
        Layer *layer = layers.at(i);
        std::vector<double> newInput;
        for (Neuron *n : layer->neurons)
        {
            double activation = this->activate(n->weights, input);
            n->output = this->transfer(activation);
            newInput.push_back(n->output);
        }
        input = newInput;
    }
    return input;
}

void Network::bprop(data *d)
{
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if (i != layers.size() - 1)
        {
            for (int j = 0; i < layer->neurons.size(); j++)
            {
                double error = 0.0;
                for (Neuron *n : layers.at(i + 1)->neurons)
                {
                    error += (n->weights.at(j) * n->delta);
                }
                errors.push_back(error);
            }
        }
        else
        {
            // output layer
            for (int j = 0; j < layer->neurons.size(); j++)
            {
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double)d->get_class_vector()->at(j) - n->output); // expect - actual
            }
        }

        for (int j = 0; j < layer->neurons.size(); j++)
        {
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transferDerivative(n->output); // grad
        }
    }
}

void Network::updateWeights(data *d)
{
    std::vector<double> inputs = *d->get_normalized_feature_vector();
    for (int i = 0; i < layers.size(); i++)
    {
        if (i != 0)
        {
            for (Neuron *n : layers.at(i - 1)->neurons)
            {
                inputs.push_back(n->output);
            }
        }

        for (Neuron *n : layers.at(i)->neurons)
        {
            for (int j = 0; j < inputs.size(); j++)
            {
                n->weights.at(j) += this->learningRate * n->delta * inputs.at(j);
            }
            n->weights.back() += this->learningRate * n->delta;
        }
        inputs.clear();
    }
}

int Network::predict(data *d) // return index of max value in output array
{
    std::vector<double> outputs = fprop(d);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

// epoch: number of times to iterate over the training data(one fprop + one bprop)
void Network::train(int numEpochs)
{
    for (int i = 0; i < numEpochs; i++)
    {
        double sumError = 0.0;
        for (data *d : *this->training_data)
        {
            std::vector<double> outputs = fprop(d);
            std::vector<int> expected = *d->get_class_vector();
            double tmpError = 0.0;
            for (int j = 0; j < outputs.size(); j++)
            {
                tmpError += pow((double)expected.at(j) - outputs.at(j), 2);
            }
            sumError += tmpError;
            bprop(d);
            updateWeights(d);
        }
        printf("Iteration: %d \t Error = %.4f\n", i, sumError);
    }
}

void Network::validate()
{
    double numCorrect = 0.0;
    double cnt = 0.0;
    for (data *d : *this->test_data)
    {
        cnt++;
        int idx = predict(d);
        if (d->get_class_vector()->at(idx) == 1)
        {
            numCorrect++;
        }
    }
    printf("Validation Performace: %.4f\n", numCorrect / cnt);
}

double Network::test()
{
    double numCorrect = 0.0;
    double cnt = 0.0;
    for (data *d : *this->test_data)
    {
        cnt++;
        int idx = predict(d);
        if (d->get_class_vector()->at(idx) == 1)
        {
            numCorrect++;
        }
    }

    double testPerformace = numCorrect / cnt;
    return testPerformance;
}