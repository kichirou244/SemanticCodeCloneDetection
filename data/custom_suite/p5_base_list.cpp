struct Node { int val; Node* next; Node(int v): val(v), next(nullptr) {} };
class List {
    Node* head = nullptr;
public:
    void add(int v) {
        Node* n = new Node(v);
        if (!head) { head = n; return; }
        Node* cur = head;
        while (cur->next) cur = cur->next;
        cur->next = n;
    }
    bool contains(int t) {
        Node* cur = head;
        while (cur) { if (cur->val == t) return true; cur = cur->next; }
        return false;
    }
    int sum() {
        int total = 0; Node* cur = head;
        while (cur) { total += cur->val; cur = cur->next; }
        return total;
    }
};
