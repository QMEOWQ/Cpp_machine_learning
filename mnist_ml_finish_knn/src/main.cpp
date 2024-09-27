#include "data.hpp"
#include "data_handler.hpp"

int main()
{
    // test data & data handler: g++ -std=c++11 -I./include/ -o test_data ./src/* & test_data.exe
    data_handler *dh = new data_handler();
    dh->read_feature_vector("./train-images.idx3-ubyte");
    dh->read_feature_labels("./train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();
}