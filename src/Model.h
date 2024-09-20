#ifndef MODEL_H
#define MODEL_H

#include <aifes.h>

class AutoEncoder {
public:
    /**
     * @brief Construct a new AutoEncoder object
     *
     * @param individual_input_size input size of individual data(eg. 8 for SensorData)
     * @param window_size sliding window size
     * @param embedding_size size of embedding layer
     * @param batch_size size of batch
     */
    AutoEncoder(uint16_t individual_input_size, uint16_t window_size, uint16_t embedding_size, uint16_t batch_size);

    /**
     * @brief Destroy the AutoEncoder object
     */
    ~AutoEncoder();

    /**
     * @brief Initialize the AutoEncoder model
     */
    void init();

    /**
     * @brief Train the AutoEncoder model
     *
     * @param input_data
     * @param target_data
     * @param total_data_size
     * @param epochs
     */
    void train(float* input_data, float* target_data, uint32_t total_data_size, uint16_t epochs);

    /**
     * @brief Infer the AutoEncoder model
     *
     * @param input_data
     * @param output_data
     */
    void infer(float* input_data, float* output_data);

    /**
     * @brief Get the Embedding object
     *
     * @param input_data
     * @param embedding_output outparam for embedding output
     */
    void getEmbedding(float* input_data, float* embedding_output);

    /**
     * @brief Get the Weights object
     *
     * @param weights outparam for weights
     */
    void getWeights(float* weights[]) const;

    /**
     * @brief Get the Biases object
     *
     * @param biases outparam for biases
     */
    void getBiases(float* biases[]) const;

    /**
     * @brief Set the Weights object
     *
     * @param weights
     */
    void setWeights(const float* weights[]) const;

    /**
     * @brief Set the Biases object
     *
     * @param biases
     */
    void setBiases(const float* biases[]) const;

private:
    uint16_t individual_input_size;
    uint16_t window_size;
    uint16_t step_size;

    uint8_t input_size;
    uint8_t embedding_size;
    uint16_t batch_size;

    // Encoder layers
    ailayer_input_f32_t encoder_input_layer{};
    ailayer_dense_f32_t encoder_dense_1{};
    ailayer_tanh_f32_t encoder_tanh_1{};
    ailayer_dense_f32_t encoder_dense_2{};
    ailayer_tanh_f32_t encoder_tanh_2{};
    ailayer_dense_f32_t encoder_dense_3{};
    ailayer_tanh_f32_t encoder_tanh_3{};

    // Decoder layers
    ailayer_input_f32_t decoder_input_layer{};
    ailayer_dense_f32_t decoder_dense_1{};
    ailayer_tanh_f32_t decoder_tanh_1{};
    ailayer_dense_f32_t decoder_dense_2{};
    ailayer_tanh_f32_t decoder_tanh_2{};
    ailayer_dense_f32_t decoder_dense_3{};
    ailayer_tanh_f32_t decoder_tanh_3{};

    // Models
    aimodel_t encoder_model{};
    aimodel_t decoder_model{};

    // Optimizer
    aiopti_adam_f32_t adam_opti{};
    aiopti_t* optimizer{};

    // Tensors
    aitensor_t input_tensor{};
    aitensor_t target_tensor{};
    aitensor_t output_tensor{};
    aitensor_t embedding_tensor{};

    // Memory pointers
    byte* parameter_memory;
    byte* training_memory;

    // Error handling functions
    static void handleTrainingError(int8_t error);
    static void handleInferenceError(int8_t error);

    static void print_aitensor(const aitensor_t* tensor);
};

#endif // MODEL_H
