#ifndef DATAQUEUE_H
#define DATAQUEUE_H

#include <stdexcept>
#include <Arduino.h>
#include <vector>

template <typename T>
class DataQueue {
private:
    std::vector<T> arr;
    std::size_t capacity;

public:
    DataQueue(std::size_t capacity);

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
    bool isEmpty() const;

    /**
     * @brief Check if the queue is full
     * @return true if full
     */
    bool isFull() const;

    /**
     * @brief Get the size of the queue
     * @return int size
     */
    std::size_t size() const;

    /**
     * @brief Get the Capacity object
     * @return int capacity
     */
    std::size_t getCapacity() const;

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
