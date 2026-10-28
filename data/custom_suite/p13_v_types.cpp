template<typename T>
class GenericQueue {
    T data[100];
    int front = 0, rear = 0, count = 0;
public:
    void enqueue(T x) { data[rear] = x; rear = (rear + 1) % 100; count++; }
    T dequeue() { T val = data[front]; front = (front + 1) % 100; count--; return val; }
    bool isEmpty() { return count == 0; }
};
