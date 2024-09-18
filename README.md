# Anomaly Detection for Disaster Using ESP32S3Feather

## Introduction

This project focuses on detecting anomalies in disaster environments using **ESP32S3 Feather with 2MB PSRAM**. The ESP32 sends weights and biases to a **Jetson Platform** to facilitate **federated learning**, where the model updates are processed on the Jetson side. This repository only contains the **ESP32** code that handles sensor data collection, model weight updates, and communication with the Jetson.

## Getting Started

### Requirements
- **ESP32S3 Feather (2MB PSRAM)**
- **PlatformIO** (for development)
- **Arduino Framework**
- **IDE** (VSCode, CLion, etc.)
### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
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