template <typename T, std::size_t Size>
DataQueue<T, Size>::DataQueue() : capacity(Size), front(0), rear(-1), count(0) {}

template <typename T, std::size_t Size>
DataQueue<T, Size>::~DataQueue() {
    while (!this->isEmpty()) {
        T data = dequeue();
        cleanup(data);
    }
}

template <typename T, std::size_t Size>
void DataQueue<T, Size>::enqueue(T data) {
    if (this->isFull()) {
        throw std::runtime_error("Queue is full, cannot enqueue.");
    }
    rear = (rear + 1) % capacity;
    arr[rear] = data;
    count++;
}

template <typename T, std::size_t Size>
T DataQueue<T, Size>::dequeue() {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }

    T data = arr[front];
    front = (front + 1) % capacity;
    count--;

    return data;
}

template <typename T, std::size_t Size>
T DataQueue<T, Size>::peek() {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }

    return arr[front];
}

template <typename T, std::size_t Size>
bool DataQueue<T, Size>::isEmpty() {
    return (count == 0);
}

template <typename T, std::size_t Size>
bool DataQueue<T, Size>::isFull() {
    return (count == capacity);
}

template <typename T, std::size_t Size>
int DataQueue<T, Size>::size() {
    return count;
}

template <typename T, std::size_t Size>
int DataQueue<T, Size>::getCapacity() {
    return capacity;
}

template <typename T, std::size_t Size>
void DataQueue<T, Size>::cleanup(T data) {
    if (std::is_pointer<T>::value) {
        free(data);
    }
}

template <typename T, std::size_t Size>
int DataQueue<T, Size>::peekAll(T* data) {
    if (this->isEmpty()) {
        throw std::runtime_error("Queue is empty");
    }
    else{
        for (int i = 0; i < count; i++) {
            data[i] = arr[(front + i) % capacity];
        }
        return count;
        }
}
