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

    ~DataQueue();

    /**
     * @brief Enqueue data to the queue
     * @param data
     */
    void enqueue(T data);

    /**
     * @brief Dequeue the front data in the queue
     * @return T data
     */
    T dequeue();

    /**
     * @brief Peek the front data in the queue
     * @return T data
     */
    T peek();

    /**
     * @brief Check if the queue is empty
     * @return true if empty
     */
    bool isEmpty();

    /**
     * @brief Check if the queue is full
     * @return true if full
     */
    bool isFull();

    /**
     * @brief Get the size of the queue
     * @return int size
     */
    int size();

    /**
     * @brief Get the Capacity object
     * @return int capacity
     */
    int getCapacity();

    /**
     * @brief Peek all data in the queue
     * @param data outparam for data
     * @return int number of data in the queue
     */
    int peekAll(T* data);

private:
    void cleanup(T data);
};

#include "dataQueue.tpp"

#endif // DATAQUEUE_H