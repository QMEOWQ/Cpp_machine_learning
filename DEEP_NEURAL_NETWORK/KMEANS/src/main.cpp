#include "kmeans.hpp"
#include "data_handler.hpp"
#include "data.hpp"

int main()
{
    data_handler *dh = new data_handler();
    dh->read_feature_vector("../train-images.idx3-ubyte");
    dh->read_feature_labels("../train-labels.idx1-ubyte");
    dh->split_data();
    dh->count_classes();

    double performance = 0.0;
    double best_performance = 0.0;
    int best_k = 1;
    for (int k = dh->get_class_counts(); k < dh->get_training_data()->size() * 0.1; k++)
    {
        kmeans *km = new kmeans(k);

        km->set_training_data(dh->get_training_data());
        km->set_test_data(dh->get_test_data());
        km->set_validation_data(dh->get_validation_data());

        km->init_clusters();
        km->train();
        performance = km->validation();
        printf("Current performance @ k = %d : %.3f %%.\n", k, performance);
        if (performance > best_performance)
        {
            best_performance = performance;
            best_k = k;
        }
    }

    // test
    kmeans *km = new kmeans(best_k);

    km->set_training_data(dh->get_training_data());
    km->set_test_data(dh->get_test_data());
    km->set_validation_data(dh->get_validation_data());

    km->init_clusters();
    km->train();
    performance = km->test();
    printf("Test performance @ k = %d : %.3f %%.\n", best_k, performance);
    // if (performance > best_performance)
    // {
    //     best_performance = performance;
    //     best_k = k;
    // }

    return 0;
}