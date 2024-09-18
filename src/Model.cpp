#include "Model.h"

AutoencoderModel::AutoencoderModel() : optimizer(nullptr), parameter_memory(nullptr), training_memory(nullptr), loss(0.0f), initialized(false) {
    // 생성자
}

AutoencoderModel::~AutoencoderModel() {
    // 소멸자
    freeMemory();
}

void AutoencoderModel::initialize() {
    if (initialized) return;

    Serial.println("Initializing AIfES model...");

    // 모델 레이어 정의
    const int input_dim = InputDim;
    const int encoding_dim1 = 64;
    const int encoding_dim2 = 32;

    ailayer_input_f32_t input_layer = AILAYER_INPUT_F32_A(input_dim, (uint16_t[]){input_dim});
    ailayer_dense_f32_t dense_layer_1 = AILAYER_DENSE_F32_A(encoding_dim1); // 인코더 레이어 1
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_A();
    ailayer_dense_f32_t dense_layer_2 = AILAYER_DENSE_F32_A(encoding_dim2); // 인코더 레이어 2 (병목)
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_A();
    ailayer_dense_f32_t dense_layer_3 = AILAYER_DENSE_F32_A(encoding_dim1); // 디코더 레이어 1
    ailayer_sigmoid_f32_t sigmoid_layer_3 = AILAYER_SIGMOID_F32_A();
    ailayer_dense_f32_t dense_layer_4 = AILAYER_DENSE_F32_A(input_dim); // 디코더 레이어 2 (출력층)
    ailayer_sigmoid_f32_t sigmoid_layer_4 = AILAYER_SIGMOID_F32_A();

    ailoss_mse_t mse_loss;

    // 모델 구조 연결
    model.input_layer = ailayer_input_f32_default(&input_layer);
    ailayer_t *x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    x = ailayer_dense_f32_default(&dense_layer_3, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_3, x);
    x = ailayer_dense_f32_default(&dense_layer_4, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_4, x);
    model.output_layer = x;

    // 손실 함수 설정
    model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);

    // 모델 컴파일
    aialgo_compile_model(&model);

    // 파라미터 메모리 할당
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    parameter_memory = (byte *) ps_malloc(parameter_memory_size); // PSRAM 사용

    if(parameter_memory == nullptr){
        Serial.println(F("ERROR: Not enough memory (RAM) available for parameters!"));
        while(1);
    }

    // 파라미터 메모리 분배
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    // 가중치 초기화 (Glorot 균등 초기화)
    aimath_f32_default_init_glorot_uniform(&((ailayer_dense_f32_t*)dense_layer_1.base)->weights);
    aimath_f32_default_init_zeros(&((ailayer_dense_f32_t*)dense_layer_1.base)->bias);

    aimath_f32_default_init_glorot_uniform(&((ailayer_dense_f32_t*)dense_layer_2.base)->weights);
    aimath_f32_default_init_zeros(&((ailayer_dense_f32_t*)dense_layer_2.base)->bias);

    aimath_f32_default_init_glorot_uniform(&((ailayer_dense_f32_t*)dense_layer_3.base)->weights);
    aimath_f32_default_init_zeros(&((ailayer_dense_f32_t*)dense_layer_3.base)->bias);

    aimath_f32_default_init_glorot_uniform(&((ailayer_dense_f32_t*)dense_layer_4.base)->weights);
    aimath_f32_default_init_zeros(&((ailayer_dense_f32_t*)dense_layer_4.base)->bias);

    // 옵티마이저 설정 (ADAM)
    aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.001f, 0.9f, 0.999f, 1e-7f);
    optimizer = aiopti_adam_f32_default(&adam_opti);

    // 훈련 메모리 할당
    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    training_memory = (byte *) ps_malloc(memory_size); // PSRAM 사용

    if(training_memory == nullptr){
        Serial.println(F("ERROR: Not enough memory (RAM) available for training! Try to use another optimizer (e.g. SGD) or make your net smaller."));
        while(1);
    }

    // 훈련 메모리 분배
    aialgo_schedule_training_memory(&model, optimizer, training_memory, memory_size);

    // 모델 훈련 초기화
    aialgo_init_model_for_training(&model, optimizer);

    initialized = true;

    Serial.println("AIfES model initialized successfully.");
}

void AutoencoderModel::train(float** data, uint32_t data_size) {
    if (!initialized) initialize();

    if (data_size < 1) {
        Serial.println("Not enough data to train.");
        return;
    }

    Serial.println("Starting training...");

    // 입력 및 타겟 데이터 배열 정의 (입력 데이터를 그대로 타겟으로 사용)
    uint16_t input_shape[] = {data_size, InputDim}; // 입력 데이터
    uint16_t target_shape[] = {data_size, InputDim}; // 타겟 데이터

    // PSRAM 사용하여 데이터 배열 할당
    float (*input_data)[InputDim] = (float (*)[InputDim]) ps_malloc(data_size * InputDim * sizeof(float));
    float (*target_data)[InputDim] = (float (*)[InputDim]) ps_malloc(data_size * InputDim * sizeof(float));

    if(input_data == nullptr || target_data == nullptr) {
        Serial.println("Memory allocation failed for training data.");
        return;
    }

    for(uint32_t i = 0; i < data_size; i++){
        for(int j = 0; j < InputDim; j++){
            input_data[i][j] = data[i][j];
            target_data[i][j] = data[i][j]; // 입력 데이터를 그대로 타겟으로 사용
        }
    }

    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);

    // 배치 사이즈 및 에포크 설정
    uint32_t batch_size = data_size;
    uint16_t epochs = 10;
    uint16_t print_interval = 5;

    for(uint16_t epoch = 0; epoch < epochs; epoch++) {
        aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);

        if(epoch % print_interval == 0){
            aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
            Serial.print(F("Epoch: "));
            Serial.print(epoch);
            Serial.print(F(" Loss: "));
            Serial.println(loss);
        }
    }

    Serial.println("Training completed.");

    // 메모리 해제
    free(input_data);
    free(target_data);
}

void AutoencoderModel::evaluate(float** data, uint32_t data_size) {
    if (!initialized) {
        Serial.println("Model is not initialized.");
        return;
    }

    if (data_size < 1) {
        Serial.println("No data to evaluate.");
        return;
    }

    // 평가를 위한 데이터 준비
    uint16_t eval_input_shape[] = {data_size, InputDim};

    // PSRAM 사용하여 데이터 배열 할당
    float (*eval_input_data)[InputDim] = (float (*)[InputDim]) ps_malloc(data_size * InputDim * sizeof(float));
    float (*eval_output_data)[InputDim] = (float (*)[InputDim]) ps_malloc(data_size * InputDim * sizeof(float));

    if(eval_input_data == nullptr || eval_output_data == nullptr) {
        Serial.println("Memory allocation failed for evaluation data.");
        return;
    }

    for(uint32_t i = 0; i < data_size; i++){
        for(int j = 0; j < InputDim; j++){
            eval_input_data[i][j] = data[i][j];
        }
    }

    aitensor_t eval_input_tensor = AITENSOR_2D_F32(eval_input_shape, eval_input_data);
    aitensor_t eval_output_tensor = AITENSOR_2D_F32(eval_input_shape, eval_output_data);

    aialgo_inference_model(&model, &eval_input_tensor, &eval_output_tensor);

    // 재구성 오차 계산
    float total_loss = 0.0f;
    for(uint32_t i = 0; i < data_size; i++){
        float sample_loss = 0.0f;
        for(int j = 0; j < InputDim; j++){
            float error = eval_input_data[i][j] - eval_output_data[i][j];
            sample_loss += error * error;
        }
        sample_loss /= InputDim;
        total_loss += sample_loss;

        Serial.print("Sample ");
        Serial.print(i);
        Serial.print(" Reconstruction Loss: ");
        Serial.println(sample_loss);
    }

    float average_loss = total_loss / data_size;
    Serial.print("Average Reconstruction Loss: ");
    Serial.println(average_loss);

    // 메모리 해제
    free(eval_input_data);
    free(eval_output_data);
}

void AutoencoderModel::getWeights() {
    if (!initialized) {
        Serial.println("Model is not initialized.");
        return;
    }

    Serial.println("Retrieving model weights...");

    // 예상 버퍼 크기 계산 (레이어당 크기: 가중치 + 바이어스)
    size_t buffer_size = aialgo_sizeof_parameter_memory(&model);
    byte* buffer = (byte*)ps_malloc(buffer_size); // PSRAM 사용
    if(buffer == nullptr){
        Serial.println("Failed to allocate buffer for model parameters.");
        return;
    }

    size_t offset = 0;
    ailayer_t* layer = model.input_layer->next; // 첫 번째 Dense 레이어부터 시작

    while(layer != nullptr) {
        serializeLayerWeights(layer, buffer, offset);
        layer = layer->next;
    }

    // 가중치 데이터를 활용하는 코드 추가 가능
    // 예를 들어, 시리얼 모니터에 출력하거나 파일에 저장하는 등의 작업을 수행할 수 있습니다.

    // 예시로 가중치 데이터의 일부를 시리얼 모니터에 출력
    Serial.println("First 100 bytes of model weights:");
    for(size_t i = 0; i < 100 && i < buffer_size; i++) {
        Serial.print(buffer[i], HEX);
        Serial.print(" ");
    }
    Serial.println();

    // 메모리 해제
    free(buffer);
}

// 기타 private 함수들 구현
void AutoencoderModel::allocateMemory() {
    // 필요한 경우 메모리 할당 관련 로직 추가
}

void AutoencoderModel::freeMemory() {
    // 메모리 해제 로직
    if(parameter_memory != nullptr) {
        free(parameter_memory);
        parameter_memory = nullptr;
    }
    if(training_memory != nullptr) {
        free(training_memory);
        training_memory = nullptr;
    }
}

void AutoencoderModel::serializeTensor(aitensor_t* tensor, byte* buffer) {
    uint16_t* shape = (uint16_t*)tensor->shape;
    uint16_t dims = tensor->dim;
    buffer[0] = dims;
    memcpy(buffer + 1, shape, dims * sizeof(uint16_t));
    memcpy(buffer + 1 + dims * sizeof(uint16_t), tensor->data, tensor->size * sizeof(float));
}

void AutoencoderModel::serializeLayerWeights(ailayer_t* layer, byte* buffer, size_t& offset) {
    // 현재는 Dense 레이어만 처리
    if (layer->type == AILAYER_DENSE_F32) {
        ailayer_dense_f32_t* dense = (ailayer_dense_f32_t*)layer;
        serializeTensor(&dense->weights, buffer + offset);
        offset += 1 + dense->weights.dim * sizeof(uint16_t) + dense->weights.size * sizeof(float);
        serializeTensor(&dense->bias, buffer + offset);
        offset += 1 + dense->bias.dim * sizeof(uint16_t) + dense->bias.size * sizeof(float);
    }
}
