struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class LinkedQueue {
    Node* head = nullptr; Node* tail = nullptr;
public:
    void enqueue(int x) {
        Node* n = new Node(x);
        if (!tail) { head = tail = n; }
        else { tail->next = n; tail = n; }
    }
    int dequeue() {
        int v = head->val; Node* old = head; head = head->next;
        if (!head) tail = nullptr;
        delete old; return v;
    }
    bool isEmpty() { return head == nullptr; }
};
