#include "Model.h"
#include <Arduino.h>

AutoEncoder::AutoEncoder(uint16_t individual_input_size, uint16_t window_size, uint16_t embedding_size, uint16_t batch_size)
        : individual_input_size(individual_input_size), window_size(window_size), embedding_size(embedding_size),
          batch_size(batch_size), parameter_memory(nullptr), training_memory(nullptr) {
    input_size = window_size * individual_input_size;
    step_size = window_size / 2; // 겹치는 데이터 수 (예: 16)
}

AutoEncoder::~AutoEncoder() {
    free(parameter_memory);
    free(training_memory);
}

void AutoEncoder::init() {
    log_d("AutoEncoder initializing...");

    // Initialize random seed
    srand(analogRead(A4));

    // --- Encoder Model Initialization ---
    uint16_t encoder_input_shape[] = {1, input_size};
    encoder_input_layer = AILAYER_INPUT_F32_A(input_size, encoder_input_shape);
    encoder_dense_1 = AILAYER_DENSE_F32_A(/*neurons=*/ 128);
    encoder_tanh_1 = AILAYER_TANH_F32_A();
    encoder_dense_2 = AILAYER_DENSE_F32_A(/*neurons=*/ 64);
    encoder_tanh_2 = AILAYER_TANH_F32_A();
    encoder_dense_3 = AILAYER_DENSE_F32_A(/*neurons=*/ embedding_size); // embedding_size = 32
    encoder_tanh_3 = AILAYER_TANH_F32_A();

    // Build Encoder Model
    ailayer_t *x;
    encoder_model.input_layer = ailayer_input_f32_default(&encoder_input_layer);
    x = ailayer_dense_f32_default(&encoder_dense_1, encoder_model.input_layer);
    x = ailayer_tanh_f32_default(&encoder_tanh_1, x);
    x = ailayer_dense_f32_default(&encoder_dense_2, x);
    x = ailayer_tanh_f32_default(&encoder_tanh_2, x);
    x = ailayer_dense_f32_default(&encoder_dense_3, x);
    x = ailayer_tanh_f32_default(&encoder_tanh_3, x);
    encoder_model.output_layer = x;

    // --- Decoder Model Initialization ---
    uint16_t decoder_input_shape[] = {1, embedding_size};
    decoder_input_layer = AILAYER_INPUT_F32_A(embedding_size, decoder_input_shape);
    decoder_dense_1 = AILAYER_DENSE_F32_A(/*neurons=*/ 64);
    decoder_tanh_1 = AILAYER_TANH_F32_A();
    decoder_dense_2 = AILAYER_DENSE_F32_A(/*neurons=*/ 128);
    decoder_tanh_2 = AILAYER_TANH_F32_A();
    decoder_dense_3 = AILAYER_DENSE_F32_A(/*neurons=*/ input_size);
    decoder_tanh_3 = AILAYER_TANH_F32_A();

    // Build Decoder Model
    decoder_model.input_layer = ailayer_input_f32_default(&decoder_input_layer);
    x = ailayer_dense_f32_default(&decoder_dense_1, decoder_model.input_layer);
    x = ailayer_tanh_f32_default(&decoder_tanh_1, x);
    x = ailayer_dense_f32_default(&decoder_dense_2, x);
    x = ailayer_tanh_f32_default(&decoder_tanh_2, x);
    x = ailayer_dense_f32_default(&decoder_dense_3, x);
    x = ailayer_tanh_f32_default(&decoder_tanh_3, x);
    decoder_model.output_layer = x;

    // Loss function
    ailoss_mse_t mse_loss_encoder;
    encoder_model.loss = ailoss_mse_f32_default(&mse_loss_encoder, encoder_model.output_layer);

    ailoss_mse_t mse_loss_decoder;
    decoder_model.loss = ailoss_mse_f32_default(&mse_loss_decoder, decoder_model.output_layer);

    // Compile models
    aialgo_compile_model(&encoder_model);
    aialgo_compile_model(&decoder_model);

    // Allocate parameter memory for both models
    uint32_t encoder_param_memory_size = aialgo_sizeof_parameter_memory(&encoder_model);
    uint32_t decoder_param_memory_size = aialgo_sizeof_parameter_memory(&decoder_model);
    uint32_t total_param_memory_size = encoder_param_memory_size + decoder_param_memory_size;

    log_w("Required memory for parameters: %d bytes \n", total_param_memory_size);

    parameter_memory = (byte*)ps_malloc(total_param_memory_size);
    if (!parameter_memory) {
        log_e("Failed to allocate memory for parameters.");
        ESP.restart();
    }

    // Distribute parameter memory
    aialgo_distribute_parameter_memory(&encoder_model, parameter_memory, encoder_param_memory_size);
    aialgo_distribute_parameter_memory(&decoder_model, parameter_memory + encoder_param_memory_size, decoder_param_memory_size);

    // Initialize weights and biases for encoder
    aimath_f32_default_init_glorot_uniform(&encoder_dense_1.weights);
    aimath_f32_default_init_zeros(&encoder_dense_1.bias);
    aimath_f32_default_init_glorot_uniform(&encoder_dense_2.weights);
    aimath_f32_default_init_zeros(&encoder_dense_2.bias);
    aimath_f32_default_init_glorot_uniform(&encoder_dense_3.weights);
    aimath_f32_default_init_zeros(&encoder_dense_3.bias);

    // Initialize weights and biases for decoder
    aimath_f32_default_init_glorot_uniform(&decoder_dense_1.weights);
    aimath_f32_default_init_zeros(&decoder_dense_1.bias);
    aimath_f32_default_init_glorot_uniform(&decoder_dense_2.weights);
    aimath_f32_default_init_zeros(&decoder_dense_2.bias);
    aimath_f32_default_init_glorot_uniform(&decoder_dense_3.weights);
    aimath_f32_default_init_zeros(&decoder_dense_3.bias);

    // Optimizer
    adam_opti = AIOPTI_ADAM_F32(/*learning rate=*/ 0.01f, /*beta_1=*/ 0.9f, /*beta_2=*/ 0.999f, /*eps=*/ 1e-7);
    optimizer = aiopti_adam_f32_default(&adam_opti);

    // Allocate training memory for both models
    uint32_t encoder_training_memory_size = aialgo_sizeof_training_memory(&encoder_model, optimizer);
    uint32_t decoder_training_memory_size = aialgo_sizeof_training_memory(&decoder_model, optimizer);
    uint32_t total_training_memory_size = encoder_training_memory_size + decoder_training_memory_size;

    log_w("Required memory for training: %d bytes \n", total_training_memory_size);

    training_memory = (byte*)ps_malloc(total_training_memory_size);
    if (!training_memory) {
        log_e("Failed to allocate memory for training.");
        ESP.restart();
    }

    // Schedule training memory
    aialgo_schedule_training_memory(&encoder_model, optimizer, training_memory, encoder_training_memory_size);
    aialgo_schedule_training_memory(&decoder_model, optimizer, training_memory + encoder_training_memory_size, decoder_training_memory_size);

    // Initialize models for training
    aialgo_init_model_for_training(&encoder_model, optimizer);
    aialgo_init_model_for_training(&decoder_model, optimizer);

    log_d("AutoEncoder initialized.");
}

void AutoEncoder::train(float* input_data, float* target_data, uint32_t total_data_size, uint16_t epochs){

    uint32_t num_windows = (total_data_size - window_size) / step_size + 1;

    uint16_t input_shape[] = {1, input_size};
    uint16_t embedding_shape[] = {1, embedding_size};
    uint16_t output_shape[] = {1, input_size};

    input_tensor = AITENSOR_2D_F32(input_shape, nullptr);
    target_tensor = AITENSOR_2D_F32(output_shape, nullptr);

    float* embedding_data = (float*)ps_malloc(embedding_size * sizeof(float));
    embedding_tensor = AITENSOR_2D_F32(embedding_shape, embedding_data);

    float* output_data = (float*)ps_malloc(input_size * sizeof(float));
    output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    uint16_t print_interval = 10;
    float loss;

    log_d("Start training");

    for (uint16_t epoch = 0; epoch < epochs; epoch++) {
        for (uint32_t window = 0; window < num_windows; window++) {

            uint32_t start_idx = window * step_size * individual_input_size;

            if (start_idx + input_size > total_data_size * individual_input_size) {
                break;
            }
            input_tensor.data = &input_data[start_idx];
            target_tensor.data = &target_data[start_idx];

            aialgo_inference_model(&encoder_model, &input_tensor, &embedding_tensor);
            aialgo_train_model(&decoder_model, &embedding_tensor, &target_tensor, optimizer, 1);
            aialgo_train_model(&encoder_model, &input_tensor, &embedding_tensor, optimizer, 1);
        }

        if (epoch % print_interval == 0) {
            float total_loss = 0.0f;
            for (uint32_t window = 0; window < num_windows; window++) {
                uint32_t start_idx = window * step_size * individual_input_size;

                if (start_idx + input_size > total_data_size * individual_input_size) {
                    break;
                }

                input_tensor.data = &input_data[start_idx];
                target_tensor.data = &target_data[start_idx];

                aialgo_inference_model(&encoder_model, &input_tensor, &embedding_tensor);
                aialgo_inference_model(&decoder_model, &embedding_tensor, &output_tensor);

                aialgo_calc_loss_model_f32(&decoder_model, &embedding_tensor, &target_tensor, &loss);

                total_loss += loss;
            }

            float average_loss = total_loss / (float)num_windows;
            log_i("Epoch: %d Loss: %f", epoch, average_loss);
        }
    }

    free(embedding_data);
    free(output_data);

    log_d("Training completed.");
}

void AutoEncoder::infer(float* input_data, float* output_data) {
    uint16_t input_shape[] = {batch_size, input_size};
    uint16_t embedding_shape[] = {batch_size, embedding_size};
    uint16_t output_shape[] = {batch_size, input_size};

    input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    embedding_tensor = AITENSOR_2D_F32(embedding_shape, (float*)ps_malloc(batch_size * embedding_size * sizeof(float)));
    output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    // Forward pass through encoder
    aialgo_inference_model(&encoder_model, &input_tensor, &embedding_tensor);

    // Forward pass through decoder
    aialgo_inference_model(&decoder_model, &embedding_tensor, &output_tensor);

    free(embedding_tensor.data);
}

void AutoEncoder::getEmbedding(float* input_data, float* embedding_output) {
    uint16_t input_shape[] = {batch_size, input_size};
    uint16_t embedding_shape[] = {batch_size, embedding_size};

    input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    embedding_tensor = AITENSOR_2D_F32(embedding_shape, embedding_output);

    // Perform inference with encoder model to get embedding
    aialgo_inference_model(&encoder_model, &input_tensor, &embedding_tensor);
}

void AutoEncoder::getWeights(float* weights[]) const {
    // Encoder layers
    weights[0] = (float*)encoder_dense_1.weights.data;
    weights[1] = (float*)encoder_dense_2.weights.data;
    weights[2] = (float*)encoder_dense_3.weights.data;

    // Decoder layers
    weights[3] = (float*)decoder_dense_1.weights.data;
    weights[4] = (float*)decoder_dense_2.weights.data;
    weights[5] = (float*)decoder_dense_3.weights.data;
}

void AutoEncoder::getBiases(float* biases[]) const {
    // Encoder layers
    biases[0] = (float*)encoder_dense_1.bias.data;
    biases[1] = (float*)encoder_dense_2.bias.data;
    biases[2] = (float*)encoder_dense_3.bias.data;

    // Decoder layers
    biases[3] = (float*)decoder_dense_1.bias.data;
    biases[4] = (float*)decoder_dense_2.bias.data;
    biases[5] = (float*)decoder_dense_3.bias.data;
}

void AutoEncoder::setWeights(const float* weights[]) const {
    // Encoder layers
    memcpy(encoder_dense_1.weights.data, weights[0], sizeof(float) * encoder_dense_1.weights.shape[0] * encoder_dense_1.weights.shape[1]);
    memcpy(encoder_dense_2.weights.data, weights[1], sizeof(float) * encoder_dense_2.weights.shape[0] * encoder_dense_2.weights.shape[1]);
    memcpy(encoder_dense_3.weights.data, weights[2], sizeof(float) * encoder_dense_3.weights.shape[0] * encoder_dense_3.weights.shape[1]);

    // Decoder layers
    memcpy(decoder_dense_1.weights.data, weights[3], sizeof(float) * decoder_dense_1.weights.shape[0] * decoder_dense_1.weights.shape[1]);
    memcpy(decoder_dense_2.weights.data, weights[4], sizeof(float) * decoder_dense_2.weights.shape[0] * decoder_dense_2.weights.shape[1]);
    memcpy(decoder_dense_3.weights.data, weights[5], sizeof(float) * decoder_dense_3.weights.shape[0] * decoder_dense_3.weights.shape[1]);
}

void AutoEncoder::setBiases(const float* biases[]) const {
    // Encoder layers
    memcpy(encoder_dense_1.bias.data, biases[0], sizeof(float) * encoder_dense_1.bias.shape[0]);
    memcpy(encoder_dense_2.bias.data, biases[1], sizeof(float) * encoder_dense_2.bias.shape[0]);
    memcpy(encoder_dense_3.bias.data, biases[2], sizeof(float) * encoder_dense_3.bias.shape[0]);

    // Decoder layers
    memcpy(decoder_dense_1.bias.data, biases[3], sizeof(float) * decoder_dense_1.bias.shape[0]);
    memcpy(decoder_dense_2.bias.data, biases[4], sizeof(float) * decoder_dense_2.bias.shape[0]);
    memcpy(decoder_dense_3.bias.data, biases[5], sizeof(float) * decoder_dense_3.bias.shape[0]);
}

void AutoEncoder::handleTrainingError(int8_t error) {
    if (error == 0) return;
    log_e("Training Error: %d", error);
    switch(error){
        case 0:
            //Serial.println(F("No Error :)"));
            break;
        case -1:
            log_e("ERROR! Tensor dtype");
            break;
        case -2:
            log_e("ERROR! Tensor shape: Data Number");
            break;
        case -3:
            log_e("ERROR! Input tensor shape does not correspond to ANN inputs");
            break;
        case -4:
            log_e("ERROR! Output tensor shape does not correspond to ANN outputs");
            break;
        case -5:
            log_e("ERROR! Use the crossentropy as loss for softmax");
            break;
        case -6:
            log_e("ERROR! learn_rate or sgd_momentum negative");
            break;
        case -7:
            log_e("ERROR! Init uniform weights min - max wrong");
            break;
        case -8:
            log_e("ERROR! batch_size: min = 1 / max = Number of training data");
            break;
        case -9:
            log_e("ERROR! Unknown activation function");
            break;
        case -10:
            log_e("ERROR! Unknown loss function");
            break;
        case -11:
            log_e("ERROR! Unknown init weights method");
            break;
        case -12:
            Serial.println(F("ERROR! Unknown optimizer"));
            break;
        case -13:
            Serial.println(F("ERROR! Not enough memory"));
            break;
        default :
            Serial.println(F("Unknown error"));
    }
}

void AutoEncoder::handleInferenceError(int8_t error) {
    if (error == 0) return;
    log_e("Inference Error: %d", error);
    switch(error){
        case 0:
            //Serial.println(F("No Error :)"));
            break;
        case -1:
            log_e("ERROR! Tensor dtype");
            break;
        case -2:
            log_e("ERROR! Tensor shape: Data Number");
            break;
        case -3:
            log_e("ERROR! Input tensor shape does not correspond to ANN inputs");
            break;
        case -4:
            log_e("ERROR! Output tensor shape does not correspond to ANN outputs");
            break;
        case -5:
            log_e("ERROR! Unknown activation function");
            break;
        case -6:
            log_e("ERROR! Not enough memory");
            break;
        default :
            log_e("ERROR! Unknown error");
    }
}

void AutoEncoder::print_aitensor(const aitensor_t* tensor) {
    // Implement tensor data print function
    float* data = (float*)tensor->data;
    uint32_t size = 1;
    for (uint16_t i = 0; i < tensor->dim; i++) {
        size *= tensor->shape[i];
    }
    for (uint32_t i = 0; i < size; i++) {
        Serial.print(data[i]);
        Serial.print(" ");
    }
    Serial.println();
}
