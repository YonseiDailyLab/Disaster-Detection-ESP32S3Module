#ifndef DATAQUEUE_H
#define DATAQUEUE_H

#include <stdexcept>
#include <Arduino.h>
#include <type_traits>

template <typename T, std::size_t Size>
class DataQueue {
private:
    T* arr;
    int capacity;
    int front;
    int rear;
    int count;

public:
    DataQueue();

    [[noreturn]] ~DataQueue();

    void enqueue(T data);
    T dequeue();
    T peek();
    bool isEmpty();
    bool isFull();
    int size();
    int getCapacity();
    T* peekAll();

private:
    void cleanup(T data);
};

#include "dataQueue.tpp"

#endif // DATAQUEUE_H