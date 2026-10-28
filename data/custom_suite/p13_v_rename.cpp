class CircularQueue {
    int buffer[100];
    int head = 0, tail = 0, size = 0;
public:
    void enqueue(int item) { buffer[tail] = item; tail = (tail + 1) % 100; size++; }
    int dequeue() { int item = buffer[head]; head = (head + 1) % 100; size--; return item; }
    bool isEmpty() { return size == 0; }
};
