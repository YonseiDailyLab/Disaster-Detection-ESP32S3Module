#ifndef MODEL_H
#define MODEL_H

#include <aifes.h>

class AutoEncoder {
public:
    AutoEncoder(uint16_t input_size, uint16_t embedding_size, uint16_t batch_size);

    // Initialize the models
    void initialize();

    // Train the autoencoder
    void train(float* input_data, float* target_data, uint32_t total_data_size);

    // Perform inference
    void infer(float* input_data, float* output_data, uint32_t total_data_size);

    // Get embedding vector
    void getEmbedding(float* input_data, float* embedding_output, uint32_t total_data_size);

    // Get weights
    void getWeights(float* weights[]);

    // Get biases
    void getBiases(float* biases[]);

    // Set weights
    void setWeights(const float* weights[]);

    // Set biases
    void setBiases(const float* biases[]);

private:
    uint8_t input_size;
    uint8_t embedding_size;
    uint16_t batch_size;

    // Encoder layers
    ailayer_input_f32_t encoder_input_layer;
    ailayer_dense_f32_t encoder_dense_1;
    ailayer_tanh_f32_t encoder_tanh_1;
    ailayer_dense_f32_t encoder_dense_2;
    ailayer_tanh_f32_t encoder_tanh_2;
    ailayer_dense_f32_t encoder_dense_3;
    ailayer_tanh_f32_t encoder_tanh_3;

    // Decoder layers
    ailayer_input_f32_t decoder_input_layer;
    ailayer_dense_f32_t decoder_dense_1;
    ailayer_tanh_f32_t decoder_tanh_1;
    ailayer_dense_f32_t decoder_dense_2;
    ailayer_tanh_f32_t decoder_tanh_2;
    ailayer_dense_f32_t decoder_dense_3;
    ailayer_tanh_f32_t decoder_tanh_3;

    // Models
    aimodel_t encoder_model;
    aimodel_t decoder_model;

    // Optimizer
    aiopti_adam_f32_t adam_opti;
    aiopti_t* optimizer;

    // Tensors
    aitensor_t input_tensor;
    aitensor_t target_tensor;
    aitensor_t output_tensor;
    aitensor_t embedding_tensor;

    // Memory pointers
    byte* parameter_memory;
    byte* training_memory;

    // Error handling functions
    void handleTrainingError(int8_t error);
    void handleInferenceError(int8_t error);

    void print_aitensor(const aitensor_t* tensor);
};

#endif // MODEL_H
