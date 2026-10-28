#include <vector>
using namespace std;
class ResizableQueue {
    vector<int> data = vector<int>(4);
    int front = 0, rear = 0, size = 0;
public:
    void enqueue(int x) {
        if (size == (int)data.size()) {
            vector<int> next(data.size() * 2);
            for (int i = 0; i < size; i++) next[i] = data[(front + i) % data.size()];
            data = next; front = 0; rear = size;
        }
        data[rear] = x; rear = (rear + 1) % data.size(); size++;
    }
    int dequeue() { int val = data[front]; front = (front + 1) % data.size(); size--; return val; }
    bool isEmpty() { return size == 0; }
};
