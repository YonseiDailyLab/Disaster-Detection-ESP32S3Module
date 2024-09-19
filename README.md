# Anomaly Detection for Disaster Using ESP32S3Feather

## Introduction

This project focuses on anomaly detection in disaster environments using an **ESP32S3 Feather with 2MB PSRAM**.

The **ESP32** sends weights and biases to a **Jetson** to facilitate **federated learning** where model updates are processed on the Jetson side.
 
This repository contains only the **ESP32** code that handles sensor data collection, model weight updates, and communication with the Jetson.

## Getting Started

### Requirements
- **ESP32S3 Feather (2MB PSRAM)**
- **PlatformIO** (for development)
- **Arduino Framework**
- **IDE** (VSCode, CLion, etc.)
### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YonseiDailyLab/Disaster-Detection-ESP32S3Module.git
   cd Disaster-Detection-ESP32S3Module
    ```
   
2. Install the necessary libraries:
    ```bash
   platformio lib install
    ```
   
3. Build the project:
    ```bash
    platformio run
     ```
   
4. Upload the firmware to the ESP32:
    ```bash
    platformio run --target upload
    ```


## Folder Structure

```

├── src
│   ├── main.cpp
│   ├── Model.h
│   ├── Model.cpp
│   ├── dataQueue.h
│   ├── dataQueue.tpp
│   ├── utils.h
│   └── utils.cpp
│
├── lib
│   └── README
│
├── include
│   └── README
│
├── test
│   └── README
│
├── platformio.ini
│
├── .gitignore
│
└── README.md

```
