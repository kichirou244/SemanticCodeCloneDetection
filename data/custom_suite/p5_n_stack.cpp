struct Node { int val; Node* next; Node(int v, Node* n): val(v), next(n) {} };
class LinkedStack {
    Node* top = nullptr;
public:
    void push(int v) { top = new Node(v, top); }
    int pop() {
        int v = top->val; Node* old = top; top = top->next; delete old;
        return v;
    }
    int peek() { return top->val; }
};
