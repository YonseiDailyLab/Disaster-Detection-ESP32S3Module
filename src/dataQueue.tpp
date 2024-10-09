#include "dataQueue.h"

template <typename T>
DataQueue<T>::DataQueue(std::size_t capacity) : capacity(capacity) {
    arr.reserve(capacity);
}

template <typename T>
DataQueue<T>::~DataQueue() {
    while (!this->isEmpty()) {
        T data = dequeue();
        cleanup(data);
    }
}

template <typename T>
void DataQueue<T>::enqueue(T data) {
    if (this->isFull()) {
        throw std::runtime_error("Queue is full, cannot enqueue.");
    }
    Serial.printf("Enqueueing data: %d \n", data);
    arr.push_back(data);
}

template <typename T>
T DataQueue<T>::dequeue() {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }

    T data = arr.front();
    arr.erase(arr.begin());
    return data;
}

template <typename T>
T DataQueue<T>::peek() {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }

    return arr.front();
}

template <typename T>
bool DataQueue<T>::isEmpty() const {
    return arr.empty();
}

template <typename T>
bool DataQueue<T>::isFull() const {
    return arr.size() == capacity;
}

template <typename T>
std::size_t DataQueue<T>::size() const {
    return arr.size();
}

template <typename T>
std::size_t DataQueue<T>::getCapacity() const {
    return capacity;
}

template <typename T>
void DataQueue<T>::cleanup(T data) {
    if (std::is_pointer<T>::value) {
        free(data);
    }
}

template <typename T>
int DataQueue<T>::peekAll(T* data) {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }

    for (std::size_t i = 0; i < arr.size(); ++i) {
        data[i] = arr[i];
    }
    return arr.size();
}
