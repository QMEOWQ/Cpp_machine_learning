#! /bin/bash

if [[ -z $MNIST_ML_ROOT ]]; then
    echo "Please define in MNIST_ML_ROOT"
    exit 1
fi

dir=$(echo "$@" | tr a-z A-Z) #makes input all uppercase
model_name_lower=$(echo "$@" | tr A-Z a-z)

mkdir -p $MNIST_ML_ROOT/$dir/include $MNIST_ML_ROOT/$dir/src
touch $MNIST_ML_ROOT/$dir/Makefile
touch $MNIST_ML_ROOT/$dir/include/"model_name_lower.hpp"
touch $MNIST_ML_ROOT/$dir/src/"model_name_lower.cpp"


# chmod +x add_new_model.sh
# ./add_new_model.sh MyNewModel