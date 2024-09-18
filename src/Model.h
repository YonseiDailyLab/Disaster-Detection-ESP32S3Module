#ifndef MODEL_H
#define MODEL_H

#include <Arduino.h>
#include <aifes.h>

#ifndef MODEL_H
#define MODEL_H

#include <Arduino.h>
#include <aifes.h>

class AutoencoderModel {
public:
    AutoencoderModel();
    ~AutoencoderModel();

    void initialize();
    void train(float** data, uint32_t data_size);
    void evaluate(float** data, uint32_t data_size);
    void getWeights();

private:
    static const int InputDim = 128; // 입력 차원

    aimodel_t model;
    aiopti_t *optimizer;
    byte *parameter_memory;
    byte *training_memory;
    float loss;
    bool initialized;

    void allocateMemory();
    void freeMemory();
    void serializeTensor(aitensor_t* tensor, byte* buffer);
    void serializeLayerWeights(ailayer_t* layer, byte* buffer, size_t& offset);
};

#endif // MODEL_H
