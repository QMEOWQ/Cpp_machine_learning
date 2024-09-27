#include "../include/kmeans.hpp"

kmeans::kmeans(int k)
{
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_indices = new std::unordered_set<int>;
}

void kmeans::init_clusters()
{
    for (int i = 0; i < num_clusters; i++)
    {
        int idx = rand() % training_data->size();
        while (used_indices->find(idx) != used_indices->end())
        {
            idx = rand() % training_data->size();
        }
        clusters->push_back(new cluster(training_data->at(idx)));
        used_indices->insert(idx);
    }
}

void kmeans::init_clusters_for_each_class()
{
    std::unordered_set<int> classes_used;
    for (int i = 0; i < training_data->size(); i++)
    {
        if (classes_used.find(training_data->at(i)->get_label()) == classes_used.end())
        {
            clusters->push_back(new cluster_t(training_data->at(i)));
            classes_used.insert(training_data->at(i)->get_label());
            used_indices->insert(i);
        }
    }
}

void kmeans::train()
{
    // a speed trick
    int idx = 0;

    while (used_indices->size() < training_data->size())
    {
        // int idx = rand() % training_data->size();
        // while (used_indices->find(idx) != used_indices->end())
        // {
        //     idx = rand() % training_data->size();
        // }
        while (used_indices->find(idx) != used_indices->end())
        {
            idx++;
        }

        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < clusters->size(); j++)
        {
            double current_dist = euclidean_distance(clusters->at(j)->centroid, training_data->at(idx));
            if (current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        clusters->at(best_cluster)->add_to_cluster(training_data->at(idx));
        used_indices->insert(idx);
    }
}

double kmeans::euclidean_distance(std::vector<double> *centroid, data *point)
{
    double dist = 0.0;
    for (int i = 0; i < centroid->size(); i++)
    {
        dist += pow(centroid->at(i) - point->get_feature_vector()->at(i), 2);
    }
    dist = sqrt(dist);
    return dist;
}

double kmeans::validation()
{
    double num_correct = 0.0;
    for (auto query_point : *validation_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < clusters->size(); j++)
        {
            double current_dist = euclidean_distance(clusters->at(j)->centroid, query_point);
            if (current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label())
        {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / (double)validation_data->size());
}

double kmeans::test()
{
    double num_correct = 0.0;
    for (auto query_point : *test_data)
    {
        double min_dist = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for (int j = 0; j < clusters->size(); j++)
        {
            double current_dist = euclidean_distance(clusters->at(j)->centroid, query_point);
            if (current_dist < min_dist)
            {
                min_dist = current_dist;
                best_cluster = j;
            }
        }
        if (clusters->at(best_cluster)->most_frequent_class == query_point->get_label())
        {
            num_correct++;
        }
    }
    return 100.0 * (num_correct / (double)test_data->size());
}
