#include "data_handler.hpp"

data_handler::data_handler()
{
    data_array = new std::vector<data *>;
    training_data = new std::vector<data *>;
    test_data = new std::vector<data *>;
    validation_data = new std::vector<data *>;
}

data_handler::~data_handler()
{
    // free memory
}

void data_handler::read_feature_vector(std::string path)
{
    uint32_t header[4]; // magic num | num images | row_size | col_size
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    // FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 4; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Done getting input file Header.\n";

        int image_size = header[2] * header[3]; // row_size * col_size
        for (int i = 0; i < header[1]; i++)
        {
            data *d = new data();
            uint8_t element[1];
            for (int j = 0; j < image_size; j++)
            {
                if (fread(element, sizeof(element), 1, f))
                {
                    d->append_to_feature_vector(element[0]);
                }
                else
                {
                    std::cout << "Error reading from file.\n";
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
    }
    else
    {
        std::cout << "Could not find file.\n";
        exit(1);
    }
}

void data_handler::read_feature_labels(std::string path)
{
    uint32_t header[2]; // magic num | num images
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "rb");
    // FILE *f = fopen(path.c_str(), "r");
    if (f)
    {
        for (int i = 0; i < 2; i++)
        {
            if (fread(bytes, sizeof(bytes), 1, f))
            {
                header[i] = convert_to_little_endian(bytes);
            }
        }
        std::cout << "Done getting label file Header.\n";

        for (int i = 0; i < header[1]; i++)
        {
            uint8_t element[1];

            if (fread(element, sizeof(element), 1, f))
            {
                data_array->at(i)->set_label(element[0]);
            }
            else
            {
                std::cout << "Error reading from file.\n";
                exit(1);
            }
        }
        printf("Successfully read and stored %lu labels.\n", data_array->size());
    }
    else
    {
        std::cout << "Could not find file.\n";
        exit(1);
    }
}

void data_handler::split_data()
{
    std::unordered_set<int> used_indices;
    int train_size = data_array->size() * TRAIN_SET_PERCENT;
    int test_size = data_array->size() * TEST_SET_PERCENT;
    int validation_size = data_array->size() * VALIDATION_SET_PERCENT;

    // training data

    int cnt = 0;
    while (cnt < train_size)
    {
        int rand_index = rand() % data_array->size(); // 0 ~ size - 1
        if (used_indices.find(rand_index) == used_indices.end())
        {
            training_data->push_back(data_array->at(rand_index));
            used_indices.insert(rand_index);
            cnt++;
        }
    }
    // printf("Training data size: %lu.\n", training_data->size());

    // test data

    cnt = 0;
    while (cnt < test_size)
    {
        int rand_index = rand() % data_array->size(); // 0 ~ size - 1
        if (used_indices.find(rand_index) == used_indices.end())
        {
            test_data->push_back(data_array->at(rand_index));
            used_indices.insert(rand_index);
            cnt++;
        }
    }
    // printf("Test data size: %lu.\n", test_data->size());

    // validation data

    cnt = 0;
    while (cnt < validation_size)
    {
        int rand_index = rand() % data_array->size(); // 0 ~ size - 1
        if (used_indices.find(rand_index) == used_indices.end())
        {
            validation_data->push_back(data_array->at(rand_index));
            used_indices.insert(rand_index);
            cnt++;
        }
    }
    // printf("Validation data size: %lu.\n", validation_data->size());

    printf("Training data size: %lu.\n", training_data->size());
    printf("Test data size: %lu.\n", test_data->size());
    printf("Validation data size: %lu.\n", validation_data->size());
}

void data_handler::count_classes()
{
    int cnt = 0;
    for (unsigned i = 0; i < data_array->size(); i++)
    {
        if (class_map.find(data_array->at(i)->get_label()) == class_map.end())
        {
            class_map[data_array->at(i)->get_label()] = cnt;
            data_array->at(i)->set_enumerated_label(cnt);
            cnt++;
        }
    }
    num_classes = cnt;
    printf("Successfully extracted %d unique classed.\n", num_classes);
}

// 小端模式存储
uint32_t data_handler::convert_to_little_endian(const unsigned char *bytes)
{
    return (uint32_t)((bytes[0] << 24) |
                      (bytes[1] << 16) |
                      (bytes[2] << 8) |
                      (bytes[3]));
}

int data_handler::get_class_counts()
{
    return num_classes;
}

std::vector<data *> *data_handler::get_training_data()
{
    return training_data;
}

std::vector<data *> *data_handler::get_test_data()
{
    return test_data;
}

std::vector<data *> *data_handler::get_validation_data()
{
    return validation_data;
}