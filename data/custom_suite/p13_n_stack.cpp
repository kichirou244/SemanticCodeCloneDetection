class ArrayStack {
    int data[100];
    int top = -1;
public:
    void enqueue(int x) { data[++top] = x; } // hanh vi la push
    int dequeue() { return data[top--]; }    // hanh vi la pop
    bool isEmpty() { return top == -1; }
};
