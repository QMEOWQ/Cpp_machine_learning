#include "knn.hpp"
#include "data_handler.hpp"
#include "stdint.h"
#include <cmath>
#include <limits>
#include <map>

knn::knn()
{
    // nothing to do here
}

knn::knn(int val)
{
    k = val;
}

knn::~knn()
{
    // nothing to do here, just auto free memory
}

void knn::find_knearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double prev_min = min;
    int idx = 0;

    for (int i = 0; i < training_data->size(); i++)
    {
        if (i == 0)
        {
            for (int j = 0; j < training_data->size(); j++)
            {
                double distance = calculate_distance(query_point, training_data->at(j));
                training_data->at(j)->set_distance(distance);
                if (distance < min)
                {
                    min = distance;
                    idx = j;
                }
            }
            neighbors->push_back(training_data->at(idx));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
        else
        {
            for (int j = 0; j < training_data->size(); j++)
            {
                // double distance = calculate_distance(query_point, training_data->at(j));
                double distance = training_data->at(j)->get_distance();
                if (distance > prev_min && distance < min)
                {
                    min = distance;
                    idx = j;
                }
            }
            neighbors->push_back(training_data->at(idx));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}

// void knn::set_training_data(std::vector<data *> *vect)
// {
//     training_data = vect;
// }

// void knn::set_test_data(std::vector<data *> *vect)
// {
//     test_data = vect;
// }

// void knn::set_validation_data(std::vector<data *> *vect)
// {
//     validation_data = vect;
// }

void knn::set_k(int val)
{
    k = val;
}

int knn::predict()
{
    std::map<uint8_t, int> class_freq;
    for (int i = 0; i < neighbors->size(); i++)
    {
        if (class_freq.find(neighbors->at(i)->get_label()) == class_freq.end())
        {
            class_freq[neighbors->at(i)->get_label()] = 1;
        }
        else
        {
            class_freq[neighbors->at(i)->get_label()]++;
        }
    }

    int max_freq = 0;
    int best_class = 0;
    for (auto kv : class_freq)
    {
        if (kv.second > max_freq)
        {
            max_freq = kv.second;
            best_class = kv.first;
        }
    }

    // delete neighbors;
    neighbors->clear();
    return best_class;
}

double knn::calculate_distance(data *query_point, data *input)
{
    double distance = 0.0;

    if (query_point->get_feature_vector_size() != input->get_feature_vector_size())
    {
        std::cout << "Error: vector size mismatch." << std::endl;
        exit(1);
    }

#ifdef EUCLID

    for (unsigned i = 0; i < query_point->get_feature_vector_size(); i++)
    {
        distance += pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);

    return distance;

#elif defined MANHATTAN

#endif

    // return distance;
}

double knn::test_performance()
{
    double current_performance = 0.0;
    int cnt = 0;
    // int data_idx = 0;
    for (data *query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            cnt++;
        }
        // data_idx++;
        // printf("Current performance = %.3f %%.\n", (cnt * 100.0) / ((double)data_idx));
    }
    current_performance = (cnt * 100.0) / ((double)test_data->size());
    printf("Validation performance for k = %d: %.3f %%.\n", k, current_performance);
    return current_performance;
}

double knn::validation_performance()
{
    double current_performance = 0.0;
    int cnt = 0;
    int data_idx = 0;
    for (data *query_point : *validation_data)
    {
        find_knearest(query_point);
        int prediction = predict();
        if (prediction == query_point->get_label())
        {
            cnt++;
        }
        data_idx++;
        printf("Current performance = %.3f %%.\n", (cnt * 100.0) / ((double)data_idx));
    }
    current_performance = (cnt * 100.0) / ((double)validation_data->size());
    printf("Validation performance for k = %d: %.3f %%.\n", k, current_performance);
    return current_performance;
}
