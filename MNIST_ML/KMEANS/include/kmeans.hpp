#ifndef __KMEANS_HPP
#define __KMEANS_HPP

#include "common.hpp"
#include "data_handler.hpp"
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>

typedef struct cluster
{
    std::vector<double> *centroid;
    std::vector<data *> *cluster_points;
    std::map<int, int> class_cnts;
    int most_frequent_class;

    // initialize cluster
    cluster(data *initial_point)
    {
        centroid = new std::vector<double>;
        cluster_points = new std::vector<data *>;

        for (auto value : *(initial_point->get_feature_vector()))
        {
            // value : distance defined in data.hpp
            centroid->push_back(value);
        }

        cluster_points->push_back(initial_point);
        class_cnts[initial_point->get_label()] = 1;
        most_frequent_class = initial_point->get_label();
    }

    void add_to_cluster(data *point)
    {
        int prev_size = cluster_points->size();
        cluster_points->push_back(point);

        for (int i = 0; i < centroid->size() - 1; i++)
        {
            double value = centroid->at(i);
            value *= prev_size;
            value += point->get_feature_vector()->at(i);
            value /= (double)cluster_points->size();
            centroid->at(i) = value;
        }

        if (class_cnts.find(point->get_label()) == class_cnts.end())
        {
            class_cnts[point->get_label()] = 1;
        }
        else
        {
            class_cnts[point->get_label()]++;
        }
        set_most_frequent_class();
    }

    void set_most_frequent_class()
    {
        int best_class;
        int freq = 0;
        for (auto kv : class_cnts)
        {
            if (kv.second > freq)
            {
                freq = kv.second;
                best_class = kv.first;
            }
        }
        most_frequent_class = best_class;
    }

} cluster_t;

class kmeans : public common_data
{
    int num_clusters;
    std::vector<cluster_t *> *clusters;
    std::unordered_set<int> *used_indices;

public:
    kmeans(int k);
    void init_clusters();
    void init_clusters_for_each_class();
    void train();
    double euclidean_distance(std::vector<double> *, data *);
    double validation();
    double test();
};

#endif