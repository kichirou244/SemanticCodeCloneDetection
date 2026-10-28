class ArrayQueue {
    int data[100];
    int front = 0, rear = 0, count = 0;
public:
    void enqueue(int x) { data[rear] = x; rear = (rear + 1) % 100; count++; }
    int dequeue() { int val = data[front]; front = (front + 1) % 100; count--; return val; }
    bool isEmpty() { return count == 0; }
};
