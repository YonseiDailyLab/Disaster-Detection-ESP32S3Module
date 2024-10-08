#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <pm2008_i2c.h>
#include <Adafruit_AM2320.h>
#include <Adafruit_BMP085.h>
#include <MQUnifiedsensor.h>
#include <PubSubClient.h>
#include <aifes.h>
#include <Adafruit_NeoPixel.h>
#include <ArduinoJson.h>

#include "dataQueue.h"
#include "utils.h"
#include "Model.h"

// Definitions for MQ-7
#define placa "ESP32S3FEATHER"
#define Voltage_Resolution 3.3
#define pin A0
#define type "MQ-7"
#define ADC_Bit_Resolution 12
#define RatioMQ7CleanAir 27.5

#define LED_PIN 33
#define NUMPIXELS 1

// Hyperparameters
#define MAX_DATA_POINTS (1<<14)
#define SensorCount 8

#define DEVICE_ID "ESP32S3FEATHER2349080"

const char *ssid = "Yonsei-IoT-2G";
const char *password = "yonseiiot209";
const char *mqtt_server = "mqtt-dashboard.com";

JsonDocument doc;
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_NeoPixel pixels(NUMPIXELS, LED_PIN, NEO_GRB + NEO_KHZ800);
AutoEncoder model(SensorCount, 32, 32, 4);

// Sensor objects
PM2008_I2C pm2008_i2c;
Adafruit_BMP085 bmp;
Adafruit_AM2320 am2320 = Adafruit_AM2320();
MQUnifiedsensor MQ7(placa, Voltage_Resolution, ADC_Bit_Resolution, pin, type);

// Data queue
DataQueue<float*> dataQueue = DataQueue<float*>(MAX_DATA_POINTS);

void setup_wifi() {
    delay(10);
    Serial.println();
    Serial.print("Connecting to ");
    Serial.println(ssid);

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    randomSeed(micros());

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}

void reconnect() {
    while (!client.connected()) {
        Serial.print("Attempting MQTT connection...");
        set_pixels_mode(pixels, CONNECTING);
        // Attempt to connect
        if (client.connect("")) {
            Serial.println("connected");
            client.subscribe("gsi-aiot/getSensorData");  // Subscribe to the required topic
        } else {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" try again in 5 seconds");
            // Wait 5 seconds before retrying
            set_pixels_blank(pixels, WARNING, 500, 5);
        }
    }
    set_pixels_blank(pixels, SUCSESSED, 250, 2);
}

void updateDataArray() {
    auto *sensorData = (float *)ps_malloc(SensorCount * sizeof(float));
    if (sensorData == nullptr) {
        Serial.println("Memory allocation failed! Could not allocate sensor data.");
        return;
    }
    Serial.println("Memory allocated for sensor data.");

    // 센서 데이터 업데이트
    uint8_t ret = pm2008_i2c.read();
    if (ret != 0) {
        Serial.println("Failed to read PM2008 sensor data.");
    }
    sensorData[0] = pm2008_i2c.pm1p0_tsi;
    sensorData[1] = pm2008_i2c.pm2p5_tsi;
    sensorData[2] = pm2008_i2c.pm10_tsi;
    sensorData[3] = MQ7.readSensor();
    sensorData[4] = am2320.readHumidity();
    sensorData[5] = am2320.readTemperature();
    sensorData[6] = bmp.readTemperature();
    sensorData[7] = (float)bmp.readPressure();
    Serial.println("Sensor data updated.");

    if (dataQueue.isFull()) {
        Serial.println("Data queue is full. Dequeueing the oldest data.");
        free(dataQueue.dequeue());
    }
    Serial.println("Sensor data enqueuing.");
    dataQueue.enqueue(sensorData);
    Serial.println("Sensor data enqueued.");
    for (int i = 0; i < SensorCount; i++) {
        Serial.print(sensorData[i]);
        Serial.print(" ");
    }
    Serial.println();
}

void pubMQTT(const char *topic) {
    String message;
    serializeJson(doc, message);

    Serial.println("Publishing message:");
    Serial.println(message);

    if (!client.connected()) {
        reconnect();
    }

    if (!client.publish(topic, message.c_str())) {
        Serial.println("Failed to publish MQTT message.");
    } else {
        Serial.println("MQTT message published successfully.");
    }
}

void callback(char *topic, byte *payload, unsigned int length) {
    if (strcmp(topic, "gsi-aiot/getSensorData") == 0) {
        updateDataArray();
        Serial.println("Sensor data updated and enqueued.");

        // 딥러닝 모델
        float *data[dataQueue.size()];
        int data_cnt = dataQueue.peekAll(data);
        log_i("Data peek all count: %d, Queue size: %d", data_cnt, dataQueue.size());

        // 모델 입력 데이터 생성
        float *input_data = (float *) ps_malloc(SensorCount * data_cnt * sizeof(float));
        if (input_data == nullptr) {
            log_e("Memory allocation failed! Could not allocate input data.");
            return;
        }
        for (int i = 0; i < data_cnt; i++) {
            for (int j = 0; j < SensorCount; j++) {
                input_data[i * SensorCount + j] = data[i][j];
            }
        }
        log_d("Input data created.");

        // 모델 학습
        model.train(input_data, input_data, data_cnt, 100);
        log_d("Model trained.");

        // 가중치 및 바이어스, 임베딩 MQTT 전송

        aitensor weightsArray[6];
        aitensor biasesArray[6];

        JsonArray weights = doc["weights"].to<JsonArray>();
        JsonArray biases = doc["biases"].to<JsonArray>();
        model.getWeights(weightsArray);
        model.getBiases(biasesArray);
        for(auto& i : weightsArray) {
            for(int j = 0; j < (int)i.shape; j++) {
                weights.add(((float*)i.data)[j]);
            }
        }
        for(auto& i : biasesArray) {
            for(int j = 0; j < (int)i.shape; j++) {
                biases.add(((float*)i.data)[j]);
            }
        }
        doc["commend"] = "Server update";
        pubMQTT("gsi-aiot/update2server");

        // 메모리 해제
        free(input_data);
    }
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    pixels.begin();
    model.init();
    int t = 5; // delay time
    int step = 256; // step size for each color transition

    for (int i = 0; i < 8; i++) { // 7 color transitions
        for (int j = 0; j < step; j++) {
            switch (i) {
                case 0: // Red increase
                    pixels.setPixelColor(0, pixels.Color(j, 0, 0));
                    break;
                case 1: // Green increase
                    pixels.setPixelColor(0, pixels.Color(255, j, 0));
                    break;
                case 2: // Red decrease
                    pixels.setPixelColor(0, pixels.Color(255 - j, 255, 0));
                    break;
                case 3: // Blue increase
                    pixels.setPixelColor(0, pixels.Color(0, 255, j));
                    break;
                case 4: // Green decrease
                    pixels.setPixelColor(0, pixels.Color(0, 255 - j, 255));
                    break;
                case 5: // Red increase
                    pixels.setPixelColor(0, pixels.Color(j, 0, 255));
                    break;
                case 6: // Green increase
                    pixels.setPixelColor(0, pixels.Color(255, j, 255));
                    break;
                case 7: // White to Black
                    pixels.setPixelColor(0, pixels.Color(255 - j, 255 - j, 255 - j));
                    break;
            }
            pixels.show();
            delay(t);
        }
    }
    if (psramFound()) {
        Serial.println("PSRAM is initialized and available.");

        // PSRAM의 사용 가능한 메모리 크기 확인
        size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        Serial.print("Free PSRAM: ");
        Serial.print(free_psram);
        Serial.println(" bytes");

        // PSRAM의 최대 연속 사용 가능한 메모리 블록 크기 확인
        size_t largest_free_block = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
        Serial.print("Largest free PSRAM block: ");
        Serial.print(largest_free_block);
        Serial.println(" bytes");
    } else {
        Serial.println("PSRAM not found or not enabled.");
    }

    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);

    // PM2008 init
    set_pixels_mode(pixels, INIT);
    pm2008_i2c.begin();
    pm2008_i2c.command();
    set_pixels_blank(pixels, SUCSESSED, 100, 3);
    Serial.println("PM2008 init");

    // BMP180 init
    set_pixels_mode(pixels, INIT);
    if (!bmp.begin()) {
        Serial.println("Could not find a valid BMP180 sensor, check wiring!");
        set_pixels_blank(pixels, ERROR, 250, 10);
        ESP.restart();  // Optionally restart to recover from error
    }
    set_pixels_blank(pixels, SUCSESSED, 100, 3);

    // AM2320 init
    set_pixels_mode(pixels, INIT);

    set_pixels_mode(pixels, INIT);
    if (!am2320.begin()) {
        Serial.println("AM2320 sensor not found");
        set_pixels_blank(pixels, ERROR, 250, 10);
        ESP.restart();
    }
    set_pixels_blank(pixels, SUCSESSED, 250, 5);

    // MQ-7 init
    set_pixels_mode(pixels, INIT);
    MQ7.setRegressionMethod(1);
    MQ7.setA(99.042);
    MQ7.setB(-1.518);
    MQ7.init();
    set_pixels_blank(pixels, SUCSESSED, 250, 5);

    Serial.print("Calibrating please wait.");
    set_pixels_mode(pixels, INIT);
    float calcR0 = 0;
    for (int i = 1; i <= 10; i++) {
        MQ7.update();
        calcR0 += MQ7.calibrate(RatioMQ7CleanAir);
        Serial.print(".");
    }
    MQ7.setR0(calcR0 / 10);
    Serial.println("  done!.");

    if (isinf(calcR0)) {
        Serial.println("Warning: Connection issue, R0 is infinite (Open circuit detected).");
        set_pixels_blank(pixels, ERROR, 250, 10);
        ESP.restart();
    }
    if (calcR0 == 0) {
        Serial.println("Warning: Connection issue found, R0 is zero (Analog pin shorts to ground).");
        set_pixels_blank(pixels, ERROR, 250, 10);
        ESP.restart();
    }
    MQ7.serialDebug(true);
    set_pixels_blank(pixels, SUCSESSED, 250, 5);

    for (int i = 0; i < MAX_DATA_POINTS; i++) {

        set_pixels_mode(pixels, TRANSMIT);
        updateDataArray();
        set_pixels_blank(pixels, RECV, 100, 5);
        delay(1000);
    }
    set_pixels_mode(pixels, INIT);
    doc["device_id"] = DEVICE_ID;
    doc["sensor_count"] = SensorCount;
    doc["data_count"] = MAX_DATA_POINTS;
    doc["model"] = "AutoEncoder";
    doc["commend"] = "init";
    delay(1000);
    set_pixels_blank(pixels, SUCSESSED, 250, 5);
    set_pixels_mode(pixels, INIT);
    pubMQTT("gsi-aiot/init");
    delay(100);
    set_pixels_blank(pixels, SUCSESSED, 250, 5);
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
}
